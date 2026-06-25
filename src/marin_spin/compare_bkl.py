# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side transformer-vs-BKL quench comparison for the marin-spin v1 (grug) model.

This reuses the original ``experiments/marin-spin/scripts/animate_comparison.py`` **verbatim** for
everything physics/rendering: the BKL ground-truth simulator (``bkl_rollout`` → numba ``_bkl_loop_jit``),
the coarsening observables (magnetization, domain walls, E/N, cluster count, S_max/N, correlation
length ξ), and the side-by-side ``make_animation``. The ONLY substitution is the transformer rollout's
forward pass: the original PyTorch ``IsingTransformer`` is replaced by the grug/base JAX ``Transformer``
loaded from our trained checkpoint, so we can compare *our* model against BKL.

The original module lives under a hyphenated directory (not an importable package name), so it is loaded
by file path; its top-level ``sys.path`` shim makes ``dataset``/``generate_trajectories`` importable.

    uv run python -m marin_spin.compare_bkl \
        --checkpoint gs://marin-eu-west4/grug/marin-spin-v1-c2e602/checkpoints/step-49400 \
        --condition-T 1.5 --n-windows 30 --output scratch/quench_T1.5/compare_bkl.gif
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib

import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS

# Load the original comparison module by file path and reuse its BKL sim + metrics + animation.
_AC_PATH = pathlib.Path(__file__).resolve().parents[2] / "marin-spin" / "scripts" / "animate_comparison.py"
_spec = importlib.util.spec_from_file_location("ising_animate_comparison", _AC_PATH)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


def checkerboard(L: int) -> np.ndarray:
    i, j = np.indices((L, L))
    return np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)


def equilibrium_configs(T: float, n: int, h5_dir: str) -> np.ndarray:
    """Load n distinct equilibrium spin configs at temperature T from the dataset (post-warmup initial_spins).

    Returns int8 [n, L, L]. Used for the same-temperature rollout: start in-distribution and check the
    model holds equilibrium (no energy drift) when run at its own T.
    """
    import glob

    from marin_spin.ising_tokenizer import open_h5
    matches = glob.glob(f"{h5_dir.rstrip('/')}/*ising_L{LATTICE_L}_T{T:.2f}.h5")
    if not matches:
        raise FileNotFoundError(f"no ising_L{LATTICE_L}_T{T:.2f}.h5 under {h5_dir}")
    with open_h5(matches[0]) as f:
        ntraj = int(f.attrs["n_traj"])
        cfgs = [f[f"trajectories/{i}/initial_spins"][:] for i in range(min(n, ntraj))]
    return np.stack(cfgs).astype(np.int8)


def _sample(logits: np.ndarray, lo: int, hi: int, temp: float, rng: np.random.Generator) -> int:
    """Constrained categorical sample over token ids [lo, hi). Mirrors the original pos/dt masking."""
    z = logits[lo:hi].astype(np.float64) / max(temp, 1e-6)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return lo + int(rng.choice(hi - lo, p=p))


def transformer_rollout_grug(model, tok, initial_spins, condition_T, n_windows, W, *,
                             sample_temp, snapshot_every, rng):
    """Grug-backed clone of ac.transformer_rollout: same protocol, same 8-tuple, JAX forward.

    Returns (configs, mags, dws, energies, n_clusters, largest_frac, xis, times) using the original
    module's pure-numpy observables so the transformer and BKL sides are scored identically.

    Uses a fixed-length PAD buffer (config context then PAD, events written in place) so the forward
    shape is constant and XLA compiles ONCE — appending tokens via concatenate instead recompiles the
    transformer at every sequence length (~100 compiles/window on CPU). Causal masking makes the
    trailing PAD invisible to the current position, so the in-place writes are exact.
    """
    snap = snapshot_every or W
    spins = initial_spins.copy()
    t_tok = tok.T_id(condition_T)
    POS, N, DT, NDT = tok.POS_OFFSET, tok.N, tok.DT_OFFSET, tok.n_dt_tokens
    ctxlen, native = 1 + 4 * N, 1 + 4 * N + 3 * W
    pad = tok.vocab_size  # PAD id (= base vocab size; model vocab = pad + 1)

    nc0, lf0, xi0 = ac.coarsening_metrics(spins)
    configs = [spins.copy()]
    mags = [ac.magnetization(spins)]
    dws = [ac.domain_walls(spins)]
    energies = [ac.energy_per_spin(spins)]
    n_clusters = [nc0]
    largest_frac = [lf0]
    xis = [xi0]
    times = [0.0]

    for w in range(n_windows):
        buf = np.full((1, native), pad, dtype=np.int32)
        buf[0, :ctxlen] = tok.encode(condition_T, spins, np.zeros(0, np.int32), np.zeros(0, np.float64))
        pos_out = np.empty(W, np.int32)
        dt_out = np.empty(W, np.float64)
        for k in range(W):
            base = ctxlen + 3 * k
            buf[0, base] = t_tok
            logits = np.asarray(model.logits(jnp.asarray(buf))[0, base, :])
            pos_tok = _sample(logits, POS, POS + N, sample_temp, rng)
            buf[0, base + 1] = pos_tok

            logits = np.asarray(model.logits(jnp.asarray(buf))[0, base + 1, :])
            dt_tok = _sample(logits, DT, DT + NDT, sample_temp, rng)
            buf[0, base + 2] = dt_tok

            pos_out[k] = pos_tok - POS
            dt_out[k] = tok.sample_dt(dt_tok, rng)  # real dt_edges (tokenizer_L16.json) → t is in true time units

        for s in range(0, W, snap):
            for idx in pos_out[s:s + snap]:
                spins.flat[idx] *= -1
            nc, lf, xi = ac.coarsening_metrics(spins)
            configs.append(spins.copy())
            mags.append(ac.magnetization(spins))
            dws.append(ac.domain_walls(spins))
            energies.append(ac.energy_per_spin(spins))
            n_clusters.append(nc)
            largest_frac.append(lf)
            xis.append(xi)
            times.append(times[-1] + dt_out[s:s + snap].sum())

        print(f"  [transformer] window {w + 1}/{n_windows}  |m|={abs(mags[-1]):.3f}  "
              f"dw={dws[-1]}  nc={n_clusters[-1]}  S_max={largest_frac[-1]:.2f}  xi={xis[-1]:.1f}", flush=True)

    return (configs, np.array(mags), np.array(dws), np.array(energies),
            np.array(n_clusters), np.array(largest_frac), np.array(xis), np.array(times))


def transformer_ensemble_grug(model, tok, initial_spins, condition_T, n_windows, W, *,
                              n_chains, sample_temp, snapshot_every, rng):
    """Batched grug rollout of n_chains independent chains from the same initial config.

    Returns (configs0, stacks) where configs0 is chain-0's lattice per snapshot (for the gif panel) and
    stacks is a dict of arrays shaped (n_chains, n_snapshots) for each observable. Metrics use the same
    ac.* numpy functions as the BKL side so both ensembles are scored identically.
    """
    snap = snapshot_every or W
    L, N, B = tok.L, tok.N, n_chains
    POS, DT, NDT = tok.POS_OFFSET, tok.DT_OFFSET, tok.n_dt_tokens
    t_tok = tok.T_id(condition_T)
    # initial_spins is either (L,L) [broadcast to all chains] or (n_chains,L,L) [per-chain, e.g. equilibrium].
    spins = (np.broadcast_to(initial_spins, (B, L, L)).copy() if initial_spins.ndim == 2 else initial_spins.copy())

    keys = ["mags", "dws", "energies", "nc", "lf", "xi"]
    stacks = {k: [] for k in keys}
    configs0 = [spins[0].copy()]

    def record():
        per = {k: [] for k in keys}
        for b in range(B):
            nc, lf, xi = ac.coarsening_metrics(spins[b])
            per["mags"].append(abs(ac.magnetization(spins[b])))
            per["dws"].append(ac.domain_walls(spins[b]))
            per["energies"].append(ac.energy_per_spin(spins[b]))
            per["nc"].append(nc)
            per["lf"].append(lf)
            per["xi"].append(xi)
        for k in keys:
            stacks[k].append(per[k])

    ctxlen, native = tok.ctxlen, tok.ctxlen + 3 * W
    pad = tok.vocab_size
    record()
    for w in range(n_windows):
        # Fixed-length PAD buffer (config context then PAD, events written in place) so the forward shape
        # is constant and XLA compiles once; causal masking hides the trailing PAD. Avoids the per-length
        # recompiles of growing concat (~100 compiles/window otherwise, brutal on CPU).
        buf = np.full((B, native), pad, dtype=np.int32)
        for b in range(B):
            buf[b, :ctxlen] = tok.encode(condition_T, spins[b], np.zeros(0, np.int32), np.zeros(0, np.float64))
        pos_all = np.empty((B, W), np.int32)
        for k in range(W):
            base = ctxlen + 3 * k
            buf[:, base] = t_tok
            logits = np.asarray(model.logits(jnp.asarray(buf))[:, base, :])
            pos_tok = _sample_batched(logits, POS, POS + N, sample_temp, rng)
            buf[:, base + 1] = pos_tok
            logits = np.asarray(model.logits(jnp.asarray(buf))[:, base + 1, :])
            dt_tok = _sample_batched(logits, DT, DT + NDT, sample_temp, rng)
            buf[:, base + 2] = dt_tok
            pos_all[:, k] = pos_tok - POS

        for s in range(0, W, snap):
            flat = spins.reshape(B, -1)
            for b in range(B):
                for idx in pos_all[b, s:s + snap]:
                    flat[b, idx] *= -1
            record()
            configs0.append(spins[0].copy())
        mm = np.mean(stacks["mags"][-1])
        print(f"  [transformer] window {w + 1}/{n_windows}  <|m|>={mm:.3f}  "
              f"<E/N>={np.mean(stacks['energies'][-1]):+.3f}  <ξ>={np.mean(stacks['xi'][-1]):.1f}", flush=True)

    return configs0, {k: np.array(v).T for k, v in stacks.items()}  # arrays (n_chains, n_snaps)


def _sample_batched(logits_bv, lo, hi, temp, rng):
    sub = logits_bv[:, lo:hi].astype(np.float64) / max(temp, 1e-6)
    sub -= sub.max(axis=1, keepdims=True)
    p = np.exp(sub)
    p /= p.sum(axis=1, keepdims=True)
    return np.array([lo + rng.choice(hi - lo, p=p[b]) for b in range(p.shape[0])], dtype=np.int32)


def bkl_ensemble(initial_spins, T, n_windows, W, *, n_chains, snapshot_every, rng):
    """Run n_chains independent BKL trajectories; return (configs0, stacks) matching the transformer side."""
    keys = ["mags", "dws", "energies", "nc", "lf", "xi"]
    cols = {k: [] for k in keys}
    configs0 = None
    for b in range(n_chains):
        c0 = initial_spins if initial_spins.ndim == 2 else initial_spins[b]
        cfgs, mags, dws, energies, nc, lf, xi, _t = ac.bkl_rollout(
            c0, T, n_windows, W, rng, snapshot_every=snapshot_every)
        cols["mags"].append(np.abs(mags))
        cols["dws"].append(dws)
        cols["energies"].append(energies)
        cols["nc"].append(nc)
        cols["lf"].append(lf)
        cols["xi"].append(xi)
        if b == 0:
            configs0 = cfgs
    return configs0, {k: np.stack(v) for k, v in cols.items()}  # arrays (n_chains, n_snaps)


def make_ensemble_animation(tr_cfg0, tr, bkl_cfg0, bkl, T, n_chains, output_path, fps, sample_temp):
    """Animated side-by-side: representative lattices (chain 0) + ensemble mean±1σ ribbons per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    n_frames = tr["mags"].shape[1]
    x = np.arange(n_frames)
    L = tr_cfg0[0].shape[0]
    max_dw = 2 * L * L

    fig = plt.figure(figsize=(13, 17))
    gs = fig.add_gridspec(5, 2, height_ratios=[2, 1, 1, 1, 1], hspace=0.5, wspace=0.3)
    ax_tr, ax_bkl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_m, ax_e, ax_nc, ax_xi = (fig.add_subplot(gs[i, :]) for i in range(1, 5))

    im_tr = ax_tr.imshow(tr_cfg0[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    im_bkl = ax_bkl.imshow(bkl_cfg0[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax_tr.axis("off"); ax_bkl.axis("off")

    panels = [
        (ax_m, "mags", r"$\langle|m|\rangle$", (-0.02, 1.05)),
        (ax_e, "energies", r"$\langle E/N\rangle$", (-2.1, 2.1)),
        (ax_nc, "nc", "n_clusters", (0, max(tr["nc"].max(), bkl["nc"].max()) * 1.05)),
        (ax_xi, "xi", r"$\xi = L/\sqrt{n_c}$", (0, L * 1.05)),
    ]
    state = {}
    for ax, key, ylabel, ylim in panels:
        tr_m, tr_s = tr[key].mean(0), tr[key].std(0)
        bk_m, bk_s = bkl[key].mean(0), bkl[key].std(0)
        lt, = ax.plot([], [], "b-", lw=1.5, label=f"Transformer (n={n_chains})")
        lb, = ax.plot([], [], "r-", lw=1.5, label=f"KMC (n={n_chains})")
        vl = ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlim(-0.2, n_frames - 1 + 0.2); ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
        state[key] = dict(ax=ax, lt=lt, lb=lb, vl=vl, tr_m=tr_m, tr_s=tr_s, bk_m=bk_m, bk_s=bk_s, fills=[])
    ax_xi.set_xlabel("snapshot")
    title = fig.suptitle("", fontsize=10)

    def update(frame):
        im_tr.set_data(tr_cfg0[frame]); im_bkl.set_data(bkl_cfg0[frame])
        ax_tr.set_title(f"Transformer (chain 0)\n<|m|>={tr['mags'][:, frame].mean():.3f}  "
                        f"<ξ>={tr['xi'][:, frame].mean():.1f}", fontsize=8)
        ax_bkl.set_title(f"KMC ground truth (chain 0)\n<|m|>={bkl['mags'][:, frame].mean():.3f}  "
                         f"<ξ>={bkl['xi'][:, frame].mean():.1f}", fontsize=8)
        xf = x[:frame + 1]
        for key, st in state.items():
            for c in st["fills"]:
                c.remove()
            st["fills"].clear()
            st["lt"].set_data(xf, st["tr_m"][:frame + 1])
            st["lb"].set_data(xf, st["bk_m"][:frame + 1])
            st["fills"].append(st["ax"].fill_between(
                xf, st["tr_m"][:frame + 1] - st["tr_s"][:frame + 1],
                st["tr_m"][:frame + 1] + st["tr_s"][:frame + 1], color="b", alpha=0.18))
            st["fills"].append(st["ax"].fill_between(
                xf, st["bk_m"][:frame + 1] - st["bk_s"][:frame + 1],
                st["bk_m"][:frame + 1] + st["bk_s"][:frame + 1], color="r", alpha=0.18))
            st["vl"].set_xdata([frame, frame])
        title.set_text(f"Ensemble (n={n_chains}) checkerboard quench  |  T={T}  |  sample_temp={sample_temp}  "
                       f"|  snapshot {frame}/{n_frames - 1}")
        return ()

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    if output_path.endswith(".mp4"):
        ani.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=80)
    else:
        ani.save(output_path, writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"\nSaved ensemble animation → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Transformer (grug) vs BKL side-by-side quench comparison")
    p.add_argument("--checkpoint", default="gs://marin-eu-west4/grug/marin-spin-v1-c2e602/checkpoints/step-49400")
    p.add_argument("--condition-T", type=float, default=1.5)
    p.add_argument("--n-ensemble", type=int, default=1, help="Number of independent chains; >1 → ensemble mean±σ gif")
    p.add_argument("--n-windows", type=int, default=30)
    p.add_argument("--window-size", type=int, default=WINDOW_EVENTS)
    p.add_argument("--snapshot-every", type=int, default=10, help="Snapshot every N events (must divide window-size)")
    p.add_argument("--sample-temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--init", choices=["checkerboard", "equilibrium"], default="checkerboard",
                   help="checkerboard = OOD quench; equilibrium = same-temperature in-distribution rollout "
                        "(start from real equilibrium configs at condition-T, test for energy drift)")
    p.add_argument("--h5-dir", default="/Users/yaelelmatad/Downloads",
                   help="Directory with ising_L16_T*.h5 (for --init equilibrium)")
    p.add_argument("--output", default="scratch/quench_T1.5/compare_bkl.gif")
    args = p.parse_args()

    L, W, T = LATTICE_L, args.window_size, args.condition_T
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.init == "equilibrium":
        n = max(args.n_ensemble, 1)
        initial_spins = equilibrium_configs(T, n, args.h5_dir)  # (n, L, L), per-chain
        e0 = float(np.mean([ac.energy_per_spin(c) for c in initial_spins]))
        print(f"Equilibrium init at T={T} ({n} real configs, <E/N>={e0:+.3f}) → same-T rollout, "
              f"{args.n_windows}×{W} events, n_ensemble={args.n_ensemble}, sample_temp={args.sample_temp}\n", flush=True)
    else:
        initial_spins = checkerboard(L)
        print(f"Checkerboard init (L={L}, dw={ac.domain_walls(initial_spins)}) → T={T}, "
              f"{args.n_windows}×{W} events, n_ensemble={args.n_ensemble}, sample_temp={args.sample_temp}\n", flush=True)

    tok = build_tokenizer()
    if args.n_ensemble <= 1:
        with set_mesh(compact_grug_mesh()):
            model = load_model(args.checkpoint)
            print("=== Transformer (grug) rollout ===", flush=True)
            tr = transformer_rollout_grug(model, tok, initial_spins, T, args.n_windows, W,
                                          sample_temp=args.sample_temp, snapshot_every=args.snapshot_every,
                                          rng=np.random.default_rng(args.seed + 2))
        print("\n=== BKL simulation (ground truth) ===", flush=True)
        bkl = ac.bkl_rollout(initial_spins, T, args.n_windows, W,
                             np.random.default_rng(args.seed + 1), snapshot_every=args.snapshot_every)
        print(f"\nFinal |m|: transformer={abs(tr[1][-1]):.3f}  BKL={abs(bkl[1][-1]):.3f}")
        ac.make_animation(*tr, *bkl, T, T, output_path=args.output, fps=args.fps, sample_temp=args.sample_temp)
        return

    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        print(f"=== Transformer (grug) ensemble: {args.n_ensemble} chains ===", flush=True)
        tr_cfg0, tr = transformer_ensemble_grug(model, tok, initial_spins, T, args.n_windows, W,
                                                n_chains=args.n_ensemble, sample_temp=args.sample_temp,
                                                snapshot_every=args.snapshot_every,
                                                rng=np.random.default_rng(args.seed + 2))
    print(f"\n=== BKL ensemble: {args.n_ensemble} chains ===", flush=True)
    bkl_cfg0, bkl = bkl_ensemble(initial_spins, T, args.n_windows, W, n_chains=args.n_ensemble,
                                 snapshot_every=args.snapshot_every, rng=np.random.default_rng(args.seed + 1))
    print(f"\nFinal <|m|>: transformer={tr['mags'][:, -1].mean():.3f}±{tr['mags'][:, -1].std():.3f}  "
          f"BKL={bkl['mags'][:, -1].mean():.3f}±{bkl['mags'][:, -1].std():.3f}")
    print(f"Final <E/N>: transformer={tr['energies'][:, -1].mean():.3f}  BKL={bkl['energies'][:, -1].mean():.3f}")
    print("\nRendering ensemble animation...", flush=True)
    make_ensemble_animation(tr_cfg0, tr, bkl_cfg0, bkl, T, args.n_ensemble,
                            output_path=args.output, fps=args.fps, sample_temp=args.sample_temp)


if __name__ == "__main__":
    main()
