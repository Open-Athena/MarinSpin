# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributional distance between the model's next-flip softmax and exact BKL, on equilibrium configs.

The cleanest fidelity metric: for real equilibrium configs at temperature T, compare the model's
softmax over the 256 position tokens, p_model, to BKL's exact selection distribution p_BKL = rate_i/R.
Reports per-config and averaged: KL(BKL‖model) (forward, = what NLL trains), KL(model‖BKL) (reverse),
Bhattacharyya distance D_B = −ln Σ√(p·q), Hellinger √(1−BC), and total variation. No rollout, no |m| —
just the per-step distributions. Compares any number of checkpoints across temps (e.g. the two extremes).

    uv run python -m marin_spin.dist_vs_bkl \
        --checkpoints winner=scratch/ckpt/step-49400 aug=scratch/ckpt-aug/step-49400 \
        --temps 1.50 3.50 --n-configs 128
"""

from __future__ import annotations

import argparse

import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import equilibrium_configs
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs
from marin_spin.rollout_quench import build_tokenizer, load_model


def divergences(p_bkl: np.ndarray, p_model: np.ndarray, eps: float = 1e-12) -> dict:
    """Row-wise divergences between two batches of distributions (N, K). Returns per-config arrays."""
    pb = p_bkl + eps
    pm = p_model + eps
    pb /= pb.sum(1, keepdims=True)
    pm /= pm.sum(1, keepdims=True)
    bc = np.sqrt(pb * pm).sum(1)  # Bhattacharyya coefficient ∈ (0,1]
    return {
        "KL(BKL‖model)": (pb * np.log(pb / pm)).sum(1),
        "KL(model‖BKL)": (pm * np.log(pm / pb)).sum(1),
        "Bhattacharyya": -np.log(np.clip(bc, eps, 1.0)),
        "Hellinger": np.sqrt(np.clip(1.0 - bc, 0.0, 1.0)),
        "TV": 0.5 * np.abs(pb - pm).sum(1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="KL / Bhattacharyya between model softmax and BKL on equilibrium configs")
    ap.add_argument("--checkpoints", nargs="+", default=["winner=scratch/ckpt/step-49400"],
                    help="name=path pairs")
    ap.add_argument("--temps", type=float, nargs="+", default=[1.50, 3.50])
    ap.add_argument("--n-configs", type=int, default=128, help="Equilibrium configs averaged per temperature")
    ap.add_argument("--h5-dir", default="/Users/yaelelmatad/Downloads")
    args = ap.parse_args()

    ckpts = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.checkpoints]
    # Equilibrium configs + exact BKL position distributions per temp (config set is identical across ckpts).
    configs = {T: equilibrium_configs(T, args.n_configs, args.h5_dir) for T in args.temps}
    p_bkl = {T: np.stack([bkl_pos_prob(c, T) for c in configs[T]]) for T in args.temps}

    tok = build_tokenizer()
    keys = ["KL(BKL‖model)", "KL(model‖BKL)", "Bhattacharyya", "Hellinger", "TV"]
    print(f"\nPosition-softmax distance to exact BKL, on {args.n_configs} equilibrium configs/temp\n")
    print(f"{'checkpoint':>10} {'T':>5} | " + " | ".join(f"{k:>14}" for k in keys))
    print("-" * (18 + 17 * len(keys)))
    with set_mesh(compact_grug_mesh()):
        for name, path in ckpts:
            model = load_model(path)
            for T in args.temps:
                pm = model_pos_probs(model, tok, list(configs[T]), T)
                d = divergences(p_bkl[T], pm)
                print(f"{name:>10} {T:>5.2f} | " + " | ".join(f"{np.mean(d[k]):>14.4f}" for k in keys))


if __name__ == "__main__":
    main()
