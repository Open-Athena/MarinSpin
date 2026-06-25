# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Break the model's next-flip probability error down by site locality (bulk vs domain-wall).

Each site's locality is its number of disagreeing neighbors k ∈ {0..4} (periodic BCs):
  k=0  bulk          (all neighbors aligned; ΔE=+8; true rate e^(−8/T), almost never flips)
  k=1               (ΔE=+4; rate e^(−4/T))
  k≥2  wall/active   (ΔE≤0; rate 1; the sites BKL actually flips)

For many BKL-evolved configs we compare the model's per-site probability p_model_i to BKL's
p_BKL_i = rate_i/R, grouped by k, to see where the model is "worst" — does it leak mass onto the
suppressed bulk sites (relative over-weighting) or misrank the active wall sites (absolute error)?

    uv run python -m marin_spin.rate_by_locality --checkpoint scratch/ckpt/step-49400 --condition-T 1.7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import ac, checkerboard
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs
from marin_spin.rollout_quench import build_tokenizer

L = 16


def n_disagree(spins: np.ndarray) -> np.ndarray:
    """Per-site count of disagreeing neighbors (0..4), periodic BCs. Row-major flattened."""
    s = spins
    d = ((s != np.roll(s, 1, 0)).astype(np.int8) + (s != np.roll(s, -1, 0))
         + (s != np.roll(s, 1, 1)) + (s != np.roll(s, -1, 1)))
    return d.ravel()


def main() -> None:
    ap = argparse.ArgumentParser(description="Model flip-probability error by site locality (bulk vs wall)")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=1.7)
    ap.add_argument("--n-events", type=int, default=1500, help="BKL events from the checkerboard to sample configs")
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    T = args.condition_T
    outdir = args.outdir or f"scratch/locality_T{T}"
    os.makedirs(outdir, exist_ok=True)

    # Sample configs along one BKL quench (spans bulk-heavy early → wall-heavy late).
    n_windows = max(1, (args.n_events + 49) // 50)
    cfgs, *_ = ac.bkl_rollout(checkerboard(L), T, n_windows, 50,
                              np.random.default_rng(args.seed + 7), snapshot_every=args.snapshot_every)
    print(f"{len(cfgs)} configs sampled along the T={T} quench\n", flush=True)

    from marin_spin.rollout_quench import load_model
    tok = build_tokenizer()
    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        p_model = model_pos_probs(model, tok, cfgs, T)  # (C, N)
    p_bkl = np.stack([bkl_pos_prob(c, T) for c in cfgs])
    ks = np.stack([n_disagree(c) for c in cfgs])  # (C, N)

    pm, pb, kf = p_model.ravel(), p_bkl.ravel(), ks.ravel()
    tv_total = 0.5 * np.abs(pm - pb).sum() / len(cfgs)  # mean per-config TV

    print(f"Per-site flip-probability error by locality k (disagreeing neighbors)  (T={T})\n")
    print(f"{'k':>3} {'meaning':>12} | {'n sites':>8} | {'<p_BKL>':>9} | {'<p_model>':>9} | "
          f"{'model/BKL':>9} | {'<|Δp|>':>9} | {'%of TV':>7}")
    label = {0: "bulk", 1: "ΔE=+4", 2: "wall ΔE=0", 3: "wall ΔE=-4", 4: "wall ΔE=-8"}
    tot_abs = np.abs(pm - pb).sum()
    rows = []
    for k in range(5):
        m = kf == k
        if m.sum() == 0:
            continue
        mb, mm = pb[m].mean(), pm[m].mean()
        abs_err = np.abs(pm[m] - pb[m]).mean()
        tv_frac = 100 * np.abs(pm[m] - pb[m]).sum() / tot_abs
        rows.append((k, m.sum(), mb, mm, mm / mb if mb > 0 else np.inf, abs_err, tv_frac))
        print(f"{k:>3} {label[k]:>12} | {m.sum():>8,} | {mb:>9.2e} | {mm:>9.2e} | "
              f"{(mm / mb if mb > 0 else np.inf):>9.2f} | {abs_err:>9.2e} | {tv_frac:>6.1f}%")

    # Collapse to bulk (k=0) vs boundary (k>=1)
    bulk, bnd = kf == 0, kf >= 1
    print(f"\n  BULK   (k=0):     n={bulk.sum():,}  <p_BKL>={pb[bulk].mean():.2e}  <p_model>={pm[bulk].mean():.2e}  "
          f"over-weight ×{pm[bulk].mean()/pb[bulk].mean():.1f}  ({100*np.abs(pm[bulk]-pb[bulk]).sum()/tot_abs:.0f}% of TV)")
    print(f"  WALL   (k≥1):     n={bnd.sum():,}  <p_BKL>={pb[bnd].mean():.2e}  <p_model>={pm[bnd].mean():.2e}  "
          f"factor ×{pm[bnd].mean()/pb[bnd].mean():.2f}  ({100*np.abs(pm[bnd]-pb[bnd]).sum()/tot_abs:.0f}% of TV)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ksr = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar([k - 0.2 for k in ksr], [r[2] for r in rows], 0.4, label="BKL", color="r")
    ax[0].bar([k + 0.2 for k in ksr], [r[3] for r in rows], 0.4, label="model", color="b")
    ax[0].set_yscale("log"); ax[0].set_xlabel("k (disagreeing neighbors)"); ax[0].set_ylabel("mean prob")
    ax[0].set_title("mean flip-prob per locality"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].bar(ksr, [r[4] for r in rows], color="purple")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("k"); ax[1].set_ylabel("model/BKL"); ax[1].set_yscale("log")
    ax[1].set_title("over/under-weighting (×)"); ax[1].grid(True, alpha=0.3)
    ax[2].bar(ksr, [r[6] for r in rows], color="green")
    ax[2].set_xlabel("k"); ax[2].set_ylabel("% of total |Δp|")
    ax[2].set_title("contribution to total error"); ax[2].grid(True, alpha=0.3)
    fig.suptitle(f"Flip-probability error by site locality (bulk k=0 → wall k≥2)  (T={T})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(outdir, f"rate_by_locality_T{T}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
