# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-site flip-probability error by RASTER position: lattice-edge sites vs interior.

The config is encoded row-major, so with periodic BCs an edge site's wrap-around neighbor is far away
in the token sequence: a top/bottom-row site's vertical-wrap neighbor is ~L² tokens away (opposite end
of the config block), and left/right-column sites sit at the raster's row-wrap discontinuity. If the
1D raster encoding handles those poorly, edge sites should have systematically larger rate error than
interior sites — independent of the spin configuration.

We average |p_model − p_BKL| per *site position* over many configs (several quench seeds × snapshots),
which averages out the config-dependent rate structure and exposes any position (edge-vs-interior)
artifact. Reported as a 16×16 error map plus interior / row-edge / col-edge / corner breakdown.

    uv run python -m marin_spin.raster_edge_error --checkpoint scratch/ckpt/step-49400 --condition-T 1.7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import ac, checkerboard
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs
from marin_spin.rollout_quench import build_tokenizer, load_model

L = 16


def main() -> None:
    ap = argparse.ArgumentParser(description="Model flip-probability error by raster position (edge vs interior)")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=1.7)
    ap.add_argument("--n-seeds", type=int, default=4, help="Independent quenches to average over")
    ap.add_argument("--n-events", type=int, default=1500)
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    T = args.condition_T
    outdir = args.outdir or f"scratch/rasteredge_T{T}"
    os.makedirs(outdir, exist_ok=True)

    n_windows = max(1, (args.n_events + 49) // 50)
    configs = []
    for sd in range(args.n_seeds):
        cfgs, *_ = ac.bkl_rollout(checkerboard(L), T, n_windows, 50,
                                  np.random.default_rng(args.seed + 100 * sd + 7), snapshot_every=args.snapshot_every)
        configs.extend(cfgs)
    print(f"{len(configs)} configs ({args.n_seeds} quenches × {len(configs)//args.n_seeds} snapshots)\n", flush=True)

    tok = build_tokenizer()
    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        pm = np.concatenate([model_pos_probs(model, tok, configs[i:i + 64], T)
                             for i in range(0, len(configs), 64)], axis=0)  # (C, N)
    pb = np.stack([bkl_pos_prob(c, T) for c in configs])

    adp = np.abs(pm - pb).mean(axis=0).reshape(L, L)   # config-averaged |Δp| per site position
    sdp = (pm - pb).mean(axis=0).reshape(L, L)         # config-averaged signed Δp

    rr, cc = np.indices((L, L))
    interior = (rr >= 1) & (rr <= L - 2) & (cc >= 1) & (cc <= L - 2)
    row_edge = (rr == 0) | (rr == L - 1)
    col_edge = (cc == 0) | (cc == L - 1)
    corner = row_edge & col_edge
    perim = row_edge | col_edge

    def stat(mask):
        return adp[mask].mean()

    base = stat(interior)
    print(f"Config-averaged |Δp| by raster position  (T={T}), relative to interior:\n")
    print(f"{'region':>26} | {'n sites':>7} | {'mean|Δp|':>10} | {'× interior':>10}")
    for name, mask in [
        ("interior (rows/cols 1..L-2)", interior),
        ("perimeter (any edge)", perim),
        ("top/bottom row (vert wrap)", row_edge & ~corner),
        ("left/right col (horiz wrap)", col_edge & ~corner),
        ("corners (both wraps)", corner),
        ("row 0 (first raster row)", rr == 0),
        ("row L-1 (last raster row)", rr == L - 1),
    ]:
        print(f"{name:>26} | {int(mask.sum()):>7} | {stat(mask):>10.3e} | {stat(mask)/base:>9.2f}×")

    # ---- corner significance: each corner individually + bootstrap vs random 4-site interior draws ----
    corners = {"(0,0) first-tok": (0, 0), "(0,L-1)": (0, L - 1),
               "(L-1,0)": (L - 1, 0), "(L-1,L-1) last-tok": (L - 1, L - 1)}
    print("\nPer-corner |Δp| (× interior):")
    for nm, (r, c) in corners.items():
        print(f"  {nm:>20}: {adp[r, c]:.3e}  ({adp[r, c]/base:.2f}×)")
    corner_mean = np.mean([adp[r, c] for r, c in corners.values()])
    int_vals = adp[interior]
    rng = np.random.default_rng(0)
    draws = np.array([rng.choice(int_vals, 4, replace=False).mean() for _ in range(20000)])
    pval = float((draws >= corner_mean).mean())
    print(f"\n  corner-mean |Δp| = {corner_mean:.3e}  ({corner_mean/base:.2f}× interior)")
    print(f"  bootstrap: P(4 random interior sites ≥ this) = {pval:.3f}  "
          f"(interior 4-site draw mean {draws.mean():.3e} ± {draws.std():.3e})")
    print(f"  → {'SIGNIFICANT' if pval < 0.05 else 'consistent with noise (n=4)'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = ax[0].imshow(adp, cmap="magma")
    ax[0].set_title("config-averaged |Δp| per site\n(bright edges ⇒ raster-edge artifact)", fontsize=10)
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    v = np.abs(sdp).max()
    im1 = ax[1].imshow(sdp, cmap="seismic", vmin=-v, vmax=v)
    ax[1].set_title("config-averaged signed Δp (model − BKL)", fontsize=10)
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_xlabel("column"); a.set_ylabel("row")
    fig.suptitle(f"Raster-position error: edge vs interior  (T={T}, {len(configs)} configs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(outdir, f"raster_edge_error_T{T}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
