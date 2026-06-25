# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directional correlation length (horizontal vs vertical) of model-generated vs BKL configs.

The Ising dynamics are isotropic, so coarsening domains should grow equally in both directions:
ξ_horizontal ≈ ξ_vertical for BKL. If the row-major raster encoding makes the model anisotropic, its
generated configs will show ξ_h ≠ ξ_v (domains elongated along one lattice axis). We roll out an
ensemble from the checkerboard at a cold T (where domains form), then measure the connected spin-spin
correlation C(r) along rows (left-right) and columns (up-down), and the integrated correlation length.

    uv run python -m marin_spin.corr_anisotropy --checkpoint scratch/ckpt/step-49400 --condition-T 1.5
"""

from __future__ import annotations

import argparse

import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import equilibrium_configs
from marin_spin.kv_decode import cached_rollout
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS


def corr_dir(spins: np.ndarray, axis: int) -> np.ndarray:
    """Connected spin-spin correlation C(r) along `axis` (1=vertical/up-down, 2=horizontal/left-right).

    spins: (B, L, L). Averages over the lattice and the ensemble; normalized so C(0)=1. Periodic BC.
    """
    L = spins.shape[-1]
    sc = spins.astype(np.float64) - spins.astype(np.float64).mean(axis=(1, 2), keepdims=True)
    C = np.array([(sc * np.roll(sc, -r, axis=axis)).mean(axis=(1, 2)).mean() for r in range(L // 2 + 1)])
    return C / C[0]


def xi_integrated(C: np.ndarray) -> float:
    """Integrated correlation length: Σ C(r) up to the first non-positive value (lattice units)."""
    s = 0.0
    for r in range(len(C)):
        if C[r] <= 0:
            break
        s += C[r]
    return s


def model_rollout_snaps(model, tok, spins0, T, n_windows, W, sample_temp, rng, burnin):
    """Batched KV-cached rollout; returns per-window snapshots after `burnin`, shape (B, n_keep, L, L)."""
    snaps, _times = cached_rollout(model, tok, spins0, T, n_windows, W, sample_temp=sample_temp, rng=rng)  # [nwin+1,B,L,L]
    keep = snaps[1 + burnin:]  # drop the initial config and the burn-in windows
    return np.transpose(keep, (1, 0, 2, 3))


def ratio_with_ci(snaps, nboot, rng):
    """ξ_v/ξ_h on all snapshots, plus 95% CI by bootstrapping over independent chains (axis 0)."""
    L = snaps.shape[-1]
    flat = snaps.reshape(-1, L, L)
    Cv, Ch = corr_dir(flat, 1), corr_dir(flat, 2)
    xv, xh = xi_integrated(Cv), xi_integrated(Ch)
    B = snaps.shape[0]
    boots = []
    for _ in range(nboot):
        bs = snaps[rng.integers(0, B, B)].reshape(-1, L, L)
        boots.append(xi_integrated(corr_dir(bs, 1)) / xi_integrated(corr_dir(bs, 2)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Cv, Ch, xv, xh, xv / xh, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Directional correlation length: model vs KMC")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=2.8,
                    help="Use a T with finite clusters (lowest hot 2.8 / highest cold 2.0), not the all-ordered cold limit")
    ap.add_argument("--n-chains", type=int, default=16, help="Model rollout chains (from equilibrium)")
    ap.add_argument("--bkl-chains", type=int, default=200, help="KMC equilibrium configs (cheap → many)")
    ap.add_argument("--n-windows", type=int, default=4)
    ap.add_argument("--burnin", type=int, default=1, help="Windows to discard before collecting snapshots")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--sample-temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h5-dir", default="/Users/yaelelmatad/Downloads")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    T, L, W = args.condition_T, LATTICE_L, WINDOW_EVENTS
    outdir = args.outdir or f"scratch/corr_T{T}"
    import os
    os.makedirs(outdir, exist_ok=True)

    tok = build_tokenizer()
    # Cluster-shape isotropy is measured on EQUILIBRIUM configs at T (finite clusters), not a quench:
    # the model is started from real equilibrium configs and evolved in-distribution; the KMC control is
    # the real equilibrium ensemble itself.
    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        init = equilibrium_configs(T, args.n_chains, args.h5_dir).astype(np.int8)
        print(f"model: {args.n_chains} chains from equilibrium × {args.n_windows} windows @ T={T}...", flush=True)
        m_snaps = model_rollout_snaps(model, tok, init, T, args.n_windows, W, args.sample_temp,
                                      np.random.default_rng(args.seed + 2), args.burnin)
    print(f"KMC: {args.bkl_chains} real equilibrium configs @ T={T}...", flush=True)
    b_snaps = equilibrium_configs(T, args.bkl_chains, args.h5_dir).astype(np.int8)[:, None]  # (B, 1, L, L)

    boot = np.random.default_rng(args.seed + 9)
    res = {}
    print(f"\n(configs: model {m_snaps.shape[0]}×{m_snaps.shape[1]}, BKL {b_snaps.shape[0]}×{b_snaps.shape[1]})\n")
    for name, snaps in [("model", m_snaps), ("KMC", b_snaps)]:
        Cv, Ch, xv, xh, ratio, lo, hi = ratio_with_ci(snaps, args.nboot, boot)
        res[name] = (Cv, Ch, xv, xh)
        print(f"{name:>6}:  ξ_v={xv:.2f}  ξ_h={xh:.2f}  ξ_v/ξ_h = {ratio:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")

    os.makedirs("scratch/data", exist_ok=True)
    np.savez(f"scratch/data/anisotropy_T{T}.npz", r=np.arange(L // 2 + 1),
             model_Cv=res["model"][0], model_Ch=res["model"][1], model_xi_v=res["model"][2], model_xi_h=res["model"][3],
             kmc_Cv=res["KMC"][0], kmc_Ch=res["KMC"][1], kmc_xi_v=res["KMC"][2], kmc_xi_h=res["KMC"][3])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(7.2, 4.6))
    r = np.arange(L // 2 + 1)
    colors = {"model": "#0F4C81", "KMC": "#C1440E"}
    for name, (Cv, Ch, xv, xh) in res.items():
        c = colors[name]
        a.plot(r, Cv, "-o", ms=4, color=c, label=rf"{name} vertical ($\xi$={xv:.2f})")
        a.plot(r, Ch, "--s", ms=4, color=c, mfc="white", label=rf"{name} horizontal ($\xi$={xh:.2f})")
    a.axhline(0, color="gray", lw=0.6)
    a.set_xlabel("separation $r$"); a.set_ylabel("$C(r)$")
    ratios = "   ".join(rf"{n}: $\xi_v/\xi_h$={xv/xh:.3f}" for n, (_, _, xv, xh) in res.items())
    a.set_title(f"Directional spin correlation: vertical vs horizontal  (T={T})\n{ratios}", fontsize=11)
    a.legend(fontsize=9, ncol=2); a.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(outdir, f"corr_anisotropy_T{T}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
