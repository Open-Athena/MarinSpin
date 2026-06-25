# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the model's next-flip position distribution to the exact BKL transition rates.

This is the most direct test of what the model learned: for a *fixed* spin config at temperature T,
BKL selects the next site to flip with probability ``p_i = rate_i / Σ_j rate_j`` where
``rate_i = exp(-max(0, ΔE_i)/T)`` (Metropolis single-spin-flip; see ``generate_trajectories._init_rates``).
The transformer, given the same config as context, emits ``softmax`` over the 256 position tokens — its
estimate of that very distribution, in a single forward pass (no rollout, no error accumulation).

We evaluate on the checkerboard plus a few BKL-evolved configs (to span the coarsening process) and
report KL(BKL‖model), total-variation distance, Pearson/Spearman correlation, and top-1 site agreement,
with side-by-side spatial heatmaps and a per-site scatter.

    uv run python -m marin_spin.probs_vs_bkl \
        --checkpoint scratch/ckpt/step-49400 --condition-T 1.7 --outdir scratch/probs_T1.7
"""

from __future__ import annotations

import argparse
import os

import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from scipy.stats import pearsonr, spearmanr

from marin_spin.compare_bkl import ac, checkerboard
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L


def bkl_pos_prob(spins: np.ndarray, T: float) -> np.ndarray:
    """Exact BKL next-flip distribution over the N sites: rate_i / Σ rate_j (row-major flat order)."""
    r = ac._init_rates(spins, T).astype(np.float64)
    return r / r.sum()


def model_pos_probs(model, tok, configs, T: float) -> np.ndarray:
    """Model's softmax over the 256 pos tokens for each config (batched). Returns (B, N)."""
    ctx = np.stack([tok.encode(T, c, np.zeros(0, np.int32), np.zeros(0, np.float64)) for c in configs])
    t_tok = tok.T_id(T)
    seq = jnp.concatenate([jnp.asarray(ctx), jnp.full((len(configs), 1), t_tok, jnp.int32)], axis=1)
    logits = np.asarray(model.logits(seq)[:, -1, :])  # (B, V)
    pos = logits[:, tok.POS_OFFSET:tok.POS_OFFSET + tok.N].astype(np.float64)
    pos -= pos.max(axis=1, keepdims=True)
    p = np.exp(pos)
    return p / p.sum(axis=1, keepdims=True)


def bkl_dt_binned(spins: np.ndarray, T: float, tok) -> tuple[np.ndarray, float]:
    """BKL waiting-time distribution Exp(R) binned into the tokenizer's dt tokens, plus mean 1/R.

    R = Σ rate_i is the total rate; Δt ~ Exp(R). Token layout (see IsingTokenizer): index 0 = underflow
    [0, edges[0]); index 1+k = [edges[k], edges[k+1]) for k=0..n_dt_bins-1; index n_dt_tokens-1 = overflow
    [edges[-1], ∞). Bin mass via the exponential CDF F(t)=1−e^(−R t).
    """
    R = float(ac._init_rates(spins, T).sum())
    e = tok.dt_edges
    surv = np.exp(-R * e)  # P(Δt ≥ edge)
    p = np.empty(tok.n_dt_tokens)
    p[0] = 1.0 - surv[0]                       # underflow
    p[1:1 + tok.n_dt_bins] = surv[:-1] - surv[1:]  # interior bins
    p[tok.n_dt_tokens - 1] = surv[-1]          # overflow
    return p, 1.0 / R


def model_dt_probs(model, tok, configs, T: float, pos_sites) -> np.ndarray:
    """Model's softmax over dt tokens, conditioned on [T_bin][config×2][T_bin][pos]. Returns (B, n_dt_tokens).

    pos_sites[b] is the position token fed before reading dt (BKL Δt is independent of the site, so we
    feed BKL's modal site as a representative choice).
    """
    t_tok = tok.T_id(T)
    seqs = []
    for c, site in zip(configs, pos_sites):
        ctx = tok.encode(T, c, np.zeros(0, np.int32), np.zeros(0, np.float64))
        seqs.append(np.concatenate([ctx, [t_tok, tok.POS_OFFSET + int(site)]]))
    seq = jnp.asarray(np.stack(seqs))
    logits = np.asarray(model.logits(seq)[:, -1, :])
    dt = logits[:, tok.DT_OFFSET:tok.DT_OFFSET + tok.n_dt_tokens].astype(np.float64)
    dt -= dt.max(axis=1, keepdims=True)
    p = np.exp(dt)
    return p / p.sum(axis=1, keepdims=True)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Model next-flip position distribution vs exact KMC rates")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=1.7)
    ap.add_argument("--evolve-events", type=int, nargs="*", default=[0, 200, 600, 1200],
                    help="Event counts (from a BKL run on the checkerboard) at which to grab test configs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    T, L = args.condition_T, LATTICE_L
    outdir = args.outdir or f"scratch/probs_T{T}"
    os.makedirs(outdir, exist_ok=True)

    # Build the test configs: a single BKL trajectory from the checkerboard, sampled at the requested
    # event counts (snapshot_every=1 so any event count is reachable; config 0 is the checkerboard).
    cb = checkerboard(L)
    max_ev = max(args.evolve_events)
    n_windows = max(1, (max_ev + 49) // 50)
    cfgs_all, *_ = ac.bkl_rollout(cb, T, n_windows, 50, np.random.default_rng(args.seed + 7), snapshot_every=1)
    configs = [cfgs_all[min(e, len(cfgs_all) - 1)] for e in args.evolve_events]
    labels = [("checkerboard" if e == 0 else f"{e} events") for e in args.evolve_events]

    tok = build_tokenizer()
    p_bkl = np.stack([bkl_pos_prob(c, T) for c in configs])
    pos_sites = p_bkl.argmax(axis=1)  # BKL modal site, fed as the representative pos before reading dt
    bkl_dt = np.stack([bkl_dt_binned(c, T, tok)[0] for c in configs])
    bkl_meandt = np.array([bkl_dt_binned(c, T, tok)[1] for c in configs])
    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        p_model = model_pos_probs(model, tok, configs, T)
        p_model_dt = model_dt_probs(model, tok, configs, T, pos_sites)

    print(f"\nNext-flip position distribution: model vs exact BKL  (T={T})\n")
    print(f"{'config':>14} | {'KL(BKL‖model)':>13} | {'TV':>6} | {'Pearson':>7} | {'Spearman':>8} | {'top1':>5}")
    rows = []
    for i, lab in enumerate(labels):
        pm, pb = p_model[i], p_bkl[i]
        kl, tv = _kl(pb, pm), _tv(pb, pm)
        pr = pearsonr(pb, pm)[0]
        sr = spearmanr(pb, pm)[0]
        top1 = "yes" if int(pm.argmax()) == int(pb.argmax()) else "no"
        rows.append((lab, kl, tv, pr, sr, top1))
        print(f"{lab:>14} | {kl:>13.4f} | {tv:>6.3f} | {pr:>7.3f} | {sr:>8.3f} | {top1:>5}")

    # Figure: rows = configs, cols = [BKL heatmap, model heatmap, scatter]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import BoundaryNorm
    n = len(configs)
    # Discretized LOG color scale, fixed across all temperatures (T=1.5, 3.5, …) so panels share colors.
    # log p spans the full dynamic range — suppressed-bulk (~1e-3), uniform (~1/256≈0.004), peaks (~0.1) —
    # with even resolution per decade. Values below VMIN clip to black; the scatter is log-log to match.
    nlev = 14
    VMIN, VMAX = 1e-4, 0.20  # fixed log range across temperatures
    levels = np.logspace(np.log10(VMIN), np.log10(VMAX), nlev + 1)
    cmap = plt.get_cmap("magma", nlev).copy()
    cmap.set_under("black"); cmap.set_over(cmap(cmap.N - 1))
    norm = BoundaryNorm(levels, cmap.N)
    # 3 columns per row: [spin config (±1)] [BKL p(flip)] [model p(flip)] — the config makes the
    # rate field self-explanatory (flips concentrate on domain walls of the shown configuration).
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    im = None
    for i, lab in enumerate(labels):
        acfg, a0, a1 = axes[i]
        acfg.imshow(np.asarray(configs[i]).reshape(L, L), cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
        acfg.set_title(f"{lab}\nspin config (±1)", fontsize=9); acfg.axis("off")
        im = a0.imshow(p_bkl[i].reshape(L, L), cmap=cmap, norm=norm)
        a0.set_title("KMC  p(flip site)", fontsize=9); a0.axis("off")
        a1.imshow(p_model[i].reshape(L, L), cmap=cmap, norm=norm)
        a1.set_title(f"model  p(flip site)\nKL={rows[i][1]:.3f}  TV={rows[i][2]:.3f}  r={rows[i][3]:.2f}",
                     fontsize=9); a1.axis("off")
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=0.04, pad=0.02, extend="both",
                 label="p(flip site) — LOG scale, fixed across all T")
    fig.suptitle(f"Next-flip position distribution: model vs exact KMC rates  (T={T})", fontsize=11)
    png = os.path.join(outdir, f"probs_vs_bkl_T{T}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png}")

    # ---- dt (waiting-time) comparison: model dt-token dist vs binned Exp(R) ----
    centers = tok.dt_centers  # real seconds per dt token (calibrated edges)
    model_meandt = (p_model_dt * centers).sum(axis=1)
    print(f"\nWaiting-time (dt) distribution: model vs binned KMC Exp(R)  (T={T})\n")
    print(f"{'config':>14} | {'KL(BKL‖model)':>13} | {'TV':>6} | {'mean Δt model':>13} | {'1/R (BKL)':>10}")
    dt_rows = []
    for i, lab in enumerate(labels):
        kl, tv = _kl(bkl_dt[i], p_model_dt[i]), _tv(bkl_dt[i], p_model_dt[i])
        dt_rows.append((lab, kl, tv, model_meandt[i], bkl_meandt[i]))
        print(f"{lab:>14} | {kl:>13.4f} | {tv:>6.3f} | {model_meandt[i]:>13.4g} | {bkl_meandt[i]:>10.4g}")

    fig2, axes2 = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.6))
    binx = np.arange(tok.n_dt_tokens)
    for i, lab in enumerate(labels):
        ax = axes2[i]
        ax.step(binx, bkl_dt[i], where="mid", color="r", label="KMC Exp(R)")
        ax.step(binx, p_model_dt[i], where="mid", color="b", label="model")
        ax.set_title(f"{lab}\nKL={dt_rows[i][1]:.3f}", fontsize=9)
        ax.set_xlabel("dt token (bin)"); ax.set_ylabel("prob"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axc = axes2[n]  # calibration: model mean Δt vs BKL 1/R
    axc.loglog(bkl_meandt, model_meandt, "o", ms=7)
    lo = min(bkl_meandt.min(), model_meandt.min()) * 0.7
    hi = max(bkl_meandt.max(), model_meandt.max()) * 1.4
    axc.loglog([lo, hi], [lo, hi], "k--", lw=0.8)
    for i, lab in enumerate(labels):
        axc.annotate(lab.split()[0], (bkl_meandt[i], model_meandt[i]), fontsize=7)
    axc.set_xlabel("KMC mean Δt = 1/R"); axc.set_ylabel("model mean Δt")
    axc.set_title("timescale calibration", fontsize=9); axc.grid(True, which="both", alpha=0.3)
    fig2.suptitle(f"Waiting-time: model dt-tokens vs binned KMC Exp(R)  (T={T})", fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    png2 = os.path.join(outdir, f"dt_vs_bkl_T{T}.png")
    fig2.savefig(png2, dpi=140)
    plt.close(fig2)
    print(f"wrote {png2}")

    np.savez(os.path.join(outdir, f"probs_vs_bkl_T{T}.npz"),
             p_model=p_model, p_bkl=p_bkl, p_model_dt=p_model_dt, bkl_dt=bkl_dt,
             model_meandt=model_meandt, bkl_meandt=bkl_meandt, labels=np.array(labels))


if __name__ == "__main__":
    main()
