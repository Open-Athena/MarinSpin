# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Is the model anisotropic because of the row-major (raster) config encoding?

The 2D Ising model is isotropic — BKL flip rates depend only on a site's neighbor *sum*, so
``p_BKL(rot90(C)) == rot90(p_BKL(C))`` exactly. The transformer, however, reads the config in
row-major order, so horizontal neighbors are ~2 tokens apart while vertical neighbors are ~2L apart.
If that breaks the symmetry, the model's next-flip distribution will **not** be rotation-equivariant,
and it may mispredict horizontal vs vertical domain walls differently — which would make stripes of one
orientation artificially stable (a candidate explanation for the "stuck in stripes" rollout chains).

Two tests:
  1. Rotation-equivariance (needs no BKL): compare model p(C) to rot90⁻¹(model p(rot90 C)). Identical ⇒
     isotropic. Report per-config correlation and the anisotropy residual ‖p − sym‖/‖p‖, and whether
     rotation-*symmetrizing* the model's prediction lowers KL to BKL.
  2. Controlled walls: a single vertical domain wall vs its 90° rotation (a horizontal wall) — same
     physics rotated, so any KL(BKL‖model) gap between them is pure raster anisotropy.

    uv run python -m marin_spin.raster_anisotropy --checkpoint scratch/ckpt/step-49400 --condition-T 1.7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from scipy.stats import pearsonr

from marin_spin.compare_bkl import ac, checkerboard
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs
from marin_spin.rollout_quench import build_tokenizer, load_model

L = 16


def _kl(p, q, eps=1e-12):
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def vertical_wall(L=L):
    """Left half +1, right half −1 (periodic ⇒ two vertical domain walls; broken bonds are horizontal)."""
    s = np.ones((L, L), dtype=np.int8)
    s[:, L // 2:] = -1
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Raster-induced anisotropy of the model's flip distribution")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=1.7)
    ap.add_argument("--evolve-events", type=int, nargs="*", default=[200, 600, 1200])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    T = args.condition_T
    outdir = args.outdir or f"scratch/aniso_T{T}"
    os.makedirs(outdir, exist_ok=True)

    # Natural configs: checkerboard + BKL-evolved snapshots from the checkerboard quench.
    cb = checkerboard(L)
    cfgs_all, *_ = ac.bkl_rollout(cb, T, max(1, (max(args.evolve_events) + 49) // 50), 50,
                                  np.random.default_rng(args.seed + 7), snapshot_every=1)
    natural = [("checkerboard", cb)] + [(f"{e}ev", cfgs_all[min(e, len(cfgs_all) - 1)]) for e in args.evolve_events]
    vw, hw = vertical_wall(), np.rot90(vertical_wall())

    tok = build_tokenizer()
    # Batch every config we need: for each natural config, its 4 rotations; plus the two walls.
    batch, index = [], {}
    for name, c in natural:
        index[name] = len(batch)
        batch += [np.rot90(c, k) for k in range(4)]
    index["vwall"], index["hwall"] = len(batch), len(batch) + 1
    batch += [vw, hw]

    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        pm = model_pos_probs(model, tok, batch, T)  # (len(batch), N)

    print(f"\nRotation-equivariance of the model's flip distribution  (T={T})")
    print("(isotropic ⇒ corr=1.0, anisotropy≈0; symmetrizing should not help if already isotropic)\n")
    print(f"{'config':>14} | {'corr(p, rot90⁻¹p_rot)':>21} | {'aniso ‖p−sym‖/‖p‖':>18} | "
          f"{'KL(BKL‖p)':>9} | {'KL(BKL‖sym)':>11}")
    rows = []
    for name, c in natural:
        b = index[name]
        # model maps for the 4 rotations, each rotated back into c's frame
        maps = [np.rot90(pm[b + k].reshape(L, L), -k) for k in range(4)]
        m0 = maps[0]
        m90 = maps[1]
        sym = np.mean(maps, axis=0)
        corr = pearsonr(m0.ravel(), m90.ravel())[0]
        aniso = np.linalg.norm(m0 - sym) / np.linalg.norm(m0)
        pb = bkl_pos_prob(c, T)
        kl0 = _kl(pb, m0.ravel())
        kls = _kl(pb, sym.ravel() / sym.sum())
        rows.append((name, corr, aniso, kl0, kls, m0, m90, sym, pb.reshape(L, L)))
        print(f"{name:>14} | {corr:>21.3f} | {aniso:>18.3f} | {kl0:>9.4f} | {kls:>11.4f}")

    # Controlled wall test: hwall = rot90(vwall); BKL identical-up-to-rotation, so KL gap = pure anisotropy.
    pv, ph = pm[index["vwall"]], pm[index["hwall"]]
    bv, bh = bkl_pos_prob(vw, T), bkl_pos_prob(hw, T)
    print("\nControlled domain-wall test (hwall = rot90(vwall); same physics rotated):")
    print(f"  vertical wall  : KL(BKL‖model) = {_kl(bv, pv):.4f}")
    print(f"  horizontal wall: KL(BKL‖model) = {_kl(bh, ph):.4f}")
    # Compare model on vwall vs the rotation of its hwall prediction (should match if isotropic)
    ph_back = np.rot90(ph.reshape(L, L), -1).ravel()
    print(f"  corr(model vwall, rot90⁻¹ model hwall) = {pearsonr(pv, ph_back)[0]:.3f}  "
          f"(1.0 ⇒ isotropic; lower ⇒ wall orientation matters)")

    # Figure: for the most-evolved natural config, p(C) vs rot-back p(rot C) vs their difference, + walls.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name, corr, aniso, kl0, kls, m0, m90, sym, pbk = rows[-1]
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.5))
    vmax = max(m0.max(), m90.max(), pbk.max())
    for a, img, ttl in [
        (ax[0, 0], pbk, f"BKL p ({name})"),
        (ax[0, 1], m0, "model p(C)"),
        (ax[0, 2], m90, "rot90⁻¹ model p(rot90 C)"),
        (ax[0, 3], m0 - m90, "anisotropy: p(C) − rot-back"),
    ]:
        d = ttl.startswith("anisotropy")
        im = a.imshow(img, cmap="seismic" if d else "magma",
                      vmin=-(np.abs(m0 - m90).max()) if d else 0, vmax=(np.abs(m0 - m90).max()) if d else vmax)
        a.set_title(ttl, fontsize=9); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    bvmax = max(bv.max(), bh.max(), pv.max(), ph.max())
    for a, img, ttl in [
        (ax[1, 0], bv.reshape(L, L), "BKL vertical wall"),
        (ax[1, 1], pv.reshape(L, L), f"model vwall (KL={_kl(bv, pv):.3f})"),
        (ax[1, 2], bh.reshape(L, L), "BKL horizontal wall"),
        (ax[1, 3], ph.reshape(L, L), f"model hwall (KL={_kl(bh, ph):.3f})"),
    ]:
        im = a.imshow(img, cmap="magma", vmin=0, vmax=bvmax)
        a.set_title(ttl, fontsize=9); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Raster anisotropy: model rotation-equivariance & H/V domain walls  (T={T})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png = os.path.join(outdir, f"raster_anisotropy_T{T}.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
