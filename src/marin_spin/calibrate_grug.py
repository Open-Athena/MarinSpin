# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibrate pos/dt sampling temperatures for the grug marin-spin model on the checkerboard quench.

Reuses the original ``experiments/marin-spin/scripts/calibrate_quench.py`` machinery verbatim — BKL
window generation (``generate_bkl_windows``), the NLL/Brent temperature fit (``calibrate_one``/``nll``),
and the early/mid/late breakdown — and only swaps the teacher-forcing forward pass to the grug/base
JAX model. This finds the temperature T that makes the model's *per-step* distribution best match the
exact BKL-sampled next token (minimum NLL), separately for position and dt tokens.

Calibration set: N independent BKL trajectories started from the **checkerboard** and run at
``condition_T`` (our actual quench scenario), tokenized with T_bin = condition_T. Uses the real fitted
tokenizer (real dt_edges), so the dt calibration is meaningful.

    uv run python -m marin_spin.calibrate_grug \
        --checkpoint scratch/ckpt/step-49400 --condition-T 1.7 --n-traj 30 --n-windows 30
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import torch
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import checkerboard
from marin_spin.rollout_quench import build_tokenizer, load_model

# The original calibration module does `from animate_comparison import ...` at top before fixing its
# path, so scripts/ (and its parent, for dataset/generate_trajectories) must be importable first.
_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "marin-spin" / "scripts"
sys.path.insert(0, str(_SCRIPTS.parent))
sys.path.insert(0, str(_SCRIPTS))
import calibrate_quench as cq  # noqa: E402  (path set up above)


def collect_logits_grug(model, tokens_all, win_all, tok, batch_size, W):
    """Grug teacher-forcing clone of cq.collect_logits: returns restricted pos/dt logits + true tokens.

    Same index convention: feed tokens[:, :-1]; logit at slot ``ctx+3k`` predicts the pos token, slot
    ``ctx+3k+1`` predicts the dt token. Returns torch tensors so cq.calibrate_one/nll work unchanged.
    """
    ctx = 1 + 4 * tok.N
    k = np.arange(W)
    pos_logit_idx, dt_logit_idx = ctx + 3 * k, ctx + 3 * k + 1
    pos_tok_idx, dt_tok_idx = ctx + 3 * k + 1, ctx + 3 * k + 2
    toks = tokens_all.numpy()
    wins = win_all.numpy()
    pl_a, pt_a, pw_a, dl_a, dt_a, dw_a = [], [], [], [], [], []
    for s in range(0, len(toks), batch_size):
        b = toks[s:s + batch_size]
        bw = wins[s:s + batch_size]
        B = b.shape[0]
        logits = np.asarray(model.logits(jnp.asarray(b[:, :-1])))  # (B, S-1, V)
        pl = logits[:, pos_logit_idx, tok.POS_OFFSET:tok.POS_OFFSET + tok.N]
        pt = b[:, pos_tok_idx] - tok.POS_OFFSET
        dl = logits[:, dt_logit_idx, tok.DT_OFFSET:tok.DT_OFFSET + tok.n_dt_tokens]
        dt = b[:, dt_tok_idx] - tok.DT_OFFSET
        bwin = np.broadcast_to(bw[:, None], (B, W))
        pl_a.append(pl.reshape(B * W, tok.N)); pt_a.append(pt.reshape(B * W)); pw_a.append(bwin.reshape(B * W))
        dl_a.append(dl.reshape(B * W, tok.n_dt_tokens)); dt_a.append(dt.reshape(B * W)); dw_a.append(bwin.reshape(B * W))
    t = lambda arrs, f: torch.tensor(np.concatenate(arrs), dtype=f)  # noqa: E731
    return (t(pl_a, torch.float32), t(pt_a, torch.long), t(pw_a, torch.long),
            t(dl_a, torch.float32), t(dt_a, torch.long), t(dw_a, torch.long))


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate pos/dt temperatures for the grug checkerboard quench")
    ap.add_argument("--checkpoint", default="scratch/ckpt/step-49400")
    ap.add_argument("--condition-T", type=float, default=1.7)
    ap.add_argument("--n-traj", type=int, default=30, help="Independent BKL chains from the checkerboard")
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bounds", type=float, nargs=2, default=[0.05, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    T, W = args.condition_T, args.window_size
    bounds = tuple(args.bounds)
    out = args.output or f"scratch/calib_T{T}/calibrated_quench_grug.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    tok = build_tokenizer()
    cb = checkerboard(16)
    print(f"Calibration set: {args.n_traj} BKL chains from checkerboard at T={T}, "
          f"{args.n_windows}×{W} events each\n", flush=True)
    all_tok, all_win = [], []
    for i in range(args.n_traj):
        for tarr, _m, w in cq.generate_bkl_windows(cb, T, args.n_windows, W, tok,
                                                   np.random.default_rng(args.seed + 1000 + i)):
            all_tok.append(tarr); all_win.append(w)
    tokens_all = torch.stack(all_tok)
    win_all = torch.tensor(all_win, dtype=torch.long)
    print(f"Dataset: {len(tokens_all)} windows; teacher-forcing the grug model...\n", flush=True)

    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        pos_lgs, pos_tru, pos_win, dt_lgs, dt_tru, dt_win = collect_logits_grug(
            model, tokens_all, win_all, tok, args.batch_size, W)
    print(f"Collected {len(pos_tru):,} pos + {len(dt_tru):,} dt events\n")

    print("=== Global calibration (all windows pooled) ===")
    T_pos, nll_pos_1, nll_pos_cal = cq.calibrate_one(pos_lgs, pos_tru, "pos (global)", bounds)
    T_dt, nll_dt_1, nll_dt_cal = cq.calibrate_one(dt_lgs, dt_tru, "dt  (global)", bounds)

    print("\n=== Per-window thirds (early / mid / late quench) ===")
    n_win = int(pos_win.max().item()) + 1
    thirds = [
        ("early", pos_win < n_win // 3, dt_win < n_win // 3),
        ("mid", (pos_win >= n_win // 3) & (pos_win < 2 * n_win // 3),
         (dt_win >= n_win // 3) & (dt_win < 2 * n_win // 3)),
        ("late", pos_win >= 2 * n_win // 3, dt_win >= 2 * n_win // 3),
    ]
    per_third = {}
    for lab, pmask, dmask in thirds:
        p_opt, _, _ = cq.calibrate_one(pos_lgs[pmask], pos_tru[pmask], f"pos {lab}", bounds)
        d_opt, _, _ = cq.calibrate_one(dt_lgs[dmask], dt_tru[dmask], f"dt  {lab}", bounds)
        per_third[lab] = {"pos_T_opt": p_opt, "dt_T_opt": d_opt}

    print(f"\n{'':=<62}")
    print(f"  GLOBAL pos_temp = {T_pos:.4f}  (NLL {nll_pos_1:.4f} → {nll_pos_cal:.4f})")
    print(f"  GLOBAL dt_temp  = {T_dt:.4f}  (NLL {nll_dt_1:.4f} → {nll_dt_cal:.4f})")
    print(f"{'':=<62}")

    result = {
        "pos_temp": round(T_pos, 4), "dt_temp": round(T_dt, 4),
        "condition_T": T, "init": "checkerboard", "n_traj": args.n_traj, "n_windows": args.n_windows,
        "nll_pos_T1": round(nll_pos_1, 4), "nll_pos_cal": round(nll_pos_cal, 4),
        "nll_dt_T1": round(nll_dt_1, 4), "nll_dt_cal": round(nll_dt_cal, 4),
        "per_third": per_third,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
