# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkerboard quench rollout of the trained marin-spin v1 (grug/base) Ising model.

Loads the winning fp32-lr1e3-20ep checkpoint, conditions on a chosen temperature, and rolls the
model forward autoregressively from a **checkerboard** initial config (the antiferromagnetic state,
maximally frustrated for the ferromagnetic Ising model — and out-of-distribution vs the training
trajectories). The quench question: does the model, told "you are at T", drive the checkerboard
toward that temperature's equilibrium?

Rollout protocol (ported verbatim from ``experiments/marin-spin/generate.py``):
  context = [T_bin] + [config x2]; then per event autoregressively
    - sample a pos token from the pos range only  [POS_OFFSET, POS_OFFSET+N)
    - sample a dt  token from the dt  range only  [DT_OFFSET, DT_OFFSET+n_dt_tokens)
  the next config is CONSTRUCTED by applying the W predicted single-spin flips (never sampled).

The only change from the original is the model forward: the PyTorch ``model(x)[0,-1]`` is replaced by
the grug/base JAX ``Transformer.logits(ids)[:, -1]`` so we can run many chains as the batch axis.

Observables are tracked vs **event count** (each window = W flips); the physical-time axis (dt) is not
reconstructible because the raw HDF5 (ttl=1d) has aged out, so dt tokens are sampled only to advance
the grammar and their decoded values are not used.

Run locally (CPU is fine — the model is ~10M params); the 58.8 MB checkpoint is pulled once:

    uv run python -m marin_spin.rollout_quench \
        --checkpoint gs://marin-eu-west4/grug/marin-spin-v1-c2e602/checkpoints/step-49400 \
        --condition-T 1.5 --n-windows 30 --n-chains 16 --outdir scratch/quench_T1.5
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.grug.model import GrugModelConfig, Transformer
from marin_spin.ising_tokenizer import IsingTokenizer
from marin_spin.tokenize_ising import LATTICE_L, SEQ_LEN, WINDOW_EVENTS

# The 14 training temperatures (from the per-temperature eval tags). build_tokenizer sorts temps, so
# T_id(1.5) == 0 — the same id the model was trained with. n_dt_bins=64 reproduces vocab_size 338
# (+1 PAD = 339), matching the trained model. dt_edges values are irrelevant here (we never decode dt).
TRAIN_TEMPS = [1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40, 3.50]
N_DT_BINS = 64

# Model config — identical to experiments/grug/marin_spin/launch.py (d256 / 6L / 8H / ffn1024).
MODEL_CONFIG = GrugModelConfig(
    vocab_size=339,
    hidden_dim=256,
    intermediate_dim=1024,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=SEQ_LEN,
    head_dim=None,
)


# The real tokenizer the grug model learned: its dt_edges were verified bit-identical to
# build_tokenizer() run on the original HDF5, so loading this JSON gives genuine physical-time
# decoding (sample_dt → real seconds) and a calibrated dt-token comparison to BKL.
TOKENIZER_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "marin-spin", "tokenizer_L16.json")


def build_tokenizer(tokenizer_json: str | None = None) -> IsingTokenizer:
    """Load the real fitted tokenizer (temps + n_dt_bins + dt_edges) the grug model was trained with.

    Falls back to dummy log-spaced dt_edges only if the JSON is missing — in that case absolute Δt
    decoding is uncalibrated (token indices are still correct, so rollouts evolving the config are
    unaffected; only physical-time decoding would be wrong).
    """
    path = tokenizer_json or TOKENIZER_JSON
    if os.path.exists(path):
        tok = IsingTokenizer.load(path)
    else:
        dt_edges = np.exp(np.linspace(np.log(1e-3), np.log(1e3), N_DT_BINS + 1))
        tok = IsingTokenizer(temps=TRAIN_TEMPS, L=LATTICE_L, n_dt_bins=N_DT_BINS, dt_edges=dt_edges)
    assert tok.vocab_size + 1 == MODEL_CONFIG.vocab_size, (tok.vocab_size, MODEL_CONFIG.vocab_size)
    return tok


def load_model(checkpoint: str) -> Transformer:
    """Init a fresh Transformer and deserialize the trained params (state.params subtree)."""
    model = Transformer.init(MODEL_CONFIG, key=jax.random.PRNGKey(0))
    model = load_checkpoint(model, checkpoint, subpath="params", mesh=None, axis_mapping=None)
    return model


def checkerboard(L: int) -> np.ndarray:
    """Antiferromagnetic checkerboard: s[i,j] = +1 if (i+j) even else -1. |m| = 0, every bond unsatisfied."""
    i, j = np.indices((L, L))
    return np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)


def energy_per_spin(spins: np.ndarray) -> np.ndarray:
    """E/N = -(1/N) sum_<ij> s_i s_j with periodic BCs. Accepts (...,L,L); reduces the last two axes."""
    nb = (
        np.roll(spins, 1, -2)
        + np.roll(spins, -1, -2)
        + np.roll(spins, 1, -1)
        + np.roll(spins, -1, -1)
    )
    n = spins.shape[-1] * spins.shape[-2]
    return -(spins * nb).sum(axis=(-1, -2)) / (2 * n)


def _context_tokens(tok: IsingTokenizer, T: float, spins_b: np.ndarray) -> np.ndarray:
    """Build the [T_bin][config x2] context for a batch of configs. Returns int32 [B, 1+4N]."""
    B = spins_b.shape[0]
    ctx = np.empty((B, 1 + 4 * tok.N), dtype=np.int32)
    for b in range(B):
        ctx[b] = tok.encode(T, spins_b[b], np.zeros(0, np.int32), np.zeros(0, np.float64))
    return ctx


def _sample_masked(logits_bv: np.ndarray, lo: int, hi: int, temp: float, rng: np.random.Generator) -> np.ndarray:
    """Per-row constrained categorical sample over token ids [lo, hi). Returns int32 [B] of token ids."""
    sub = logits_bv[:, lo:hi].astype(np.float64) / max(temp, 1e-6)
    sub -= sub.max(axis=1, keepdims=True)
    p = np.exp(sub)
    p /= p.sum(axis=1, keepdims=True)
    out = np.empty(logits_bv.shape[0], dtype=np.int32)
    for b in range(p.shape[0]):
        out[b] = lo + rng.choice(hi - lo, p=p[b])
    return out


def rollout(
    model: Transformer,
    tok: IsingTokenizer,
    *,
    T: float,
    n_windows: int,
    n_chains: int,
    sample_temp: float,
    gif_chain_every: int,
    rng: np.random.Generator,
) -> dict:
    """Batched checkerboard quench. Returns observables vs event count plus chain-0 config snapshots."""
    L, N, W = tok.L, tok.N, WINDOW_EVENTS
    spins = np.broadcast_to(checkerboard(L), (n_chains, L, L)).copy()  # [B,L,L]
    t_tok = tok.T_id(T)

    abs_mag = [float(np.abs(spins.reshape(n_chains, -1).mean(1)).mean())]
    energy = [float(energy_per_spin(spins).mean())]
    energy_std = [float(energy_per_spin(spins).std())]
    events_at = [0]
    snaps = [spins[0].copy()]  # chain-0 lattice for the gif
    snap_mag = [float(np.abs(spins[0].mean()))]
    snap_energy = [float(energy_per_spin(spins[0]))]

    event_count = 0
    for w in range(n_windows):
        ctx = _context_tokens(tok, T, spins)  # [B, 1+4N]
        seq = jnp.asarray(ctx)  # grows along axis 1 as we generate
        for _ in range(W):
            seq = jnp.concatenate([seq, jnp.full((n_chains, 1), t_tok, dtype=jnp.int32)], axis=1)
            logits = np.asarray(model.logits(seq)[:, -1, :])  # [B,V]
            pos_tok = _sample_masked(logits, tok.POS_OFFSET, tok.POS_OFFSET + N, sample_temp, rng)
            seq = jnp.concatenate([seq, jnp.asarray(pos_tok)[:, None]], axis=1)

            logits = np.asarray(model.logits(seq)[:, -1, :])
            dt_tok = _sample_masked(logits, tok.DT_OFFSET, tok.DT_OFFSET + tok.n_dt_tokens, sample_temp, rng)
            seq = jnp.concatenate([seq, jnp.asarray(dt_tok)[:, None]], axis=1)

            flat = pos_tok - tok.POS_OFFSET  # [B] flipped site per chain
            flat_view = spins.reshape(n_chains, -1)
            flat_view[np.arange(n_chains), flat] *= -1  # apply the single-spin flips

            event_count += 1
            if event_count % gif_chain_every == 0:
                snaps.append(spins[0].copy())
                snap_mag.append(float(spins[0].mean()))
                snap_energy.append(float(energy_per_spin(spins[0])))

        abs_mag.append(float(np.abs(spins.reshape(n_chains, -1).mean(1)).mean()))
        e = energy_per_spin(spins)
        energy.append(float(e.mean()))
        energy_std.append(float(e.std()))
        events_at.append(event_count)
        print(f"  window {w + 1:>3}/{n_windows}  events={event_count:>5}  "
              f"<|m|>={abs_mag[-1]:.3f}  <E/N>={energy[-1]:+.3f}", flush=True)

    return {
        "events_at": np.array(events_at),
        "abs_mag": np.array(abs_mag),
        "energy": np.array(energy),
        "energy_std": np.array(energy_std),
        "snaps": np.array(snaps),
        "snap_mag": np.array(snap_mag),
        "snap_energy": np.array(snap_energy),
        "final_spins": spins,
        "T": T,
        "n_chains": n_chains,
    }


def write_outputs(result: dict, outdir: str) -> None:
    """Write the summary plot (PNG), the chain-0 quench gif, and the raw arrays (npz)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    T = result["T"]
    ev, m, e, es = result["events_at"], result["abs_mag"], result["energy"], result["energy_std"]

    # --- summary plot: <|m|> and <E/N> vs event count (mean over chains) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ev, m, marker="o", ms=4, color="C3")
    ax1.set_xlabel("events (spin flips)")
    ax1.set_ylabel(r"$\langle |m| \rangle$ over chains")
    ax1.set_title(f"Checkerboard quench → T={T}: magnetization")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(ev, e - es, e + es, color="C0", alpha=0.2)
    ax2.plot(ev, e, marker="o", ms=4, color="C0")
    ax2.set_xlabel("events (spin flips)")
    ax2.set_ylabel(r"$\langle E/N \rangle$ over chains")
    ax2.set_title(f"Checkerboard quench → T={T}: energy/spin")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(outdir, f"quench_T{T}_summary.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"  wrote {png}")

    # --- gif: chain-0 lattice evolving from the checkerboard ---
    snaps, sm, se = result["snaps"], result["snap_mag"], result["snap_energy"]
    figg, axg = plt.subplots(figsize=(4.5, 5))
    im = axg.imshow(snaps[0], cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    axg.axis("off")
    title = axg.set_title("")

    def update(k):
        im.set_data(snaps[k])
        title.set_text(f"T={T}  frame {k}/{len(snaps) - 1}\n m={sm[k]:+.3f}   E/N={se[k]:+.3f}")
        return im, title

    anim = animation.FuncAnimation(figg, update, frames=len(snaps), interval=120, blit=False)
    gif = os.path.join(outdir, f"quench_T{T}_chain0.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=8))
    plt.close(figg)
    print(f"  wrote {gif}")

    npz = os.path.join(outdir, f"quench_T{T}_arrays.npz")
    np.savez(npz, events_at=ev, abs_mag=m, energy=e, energy_std=es, snaps=snaps)
    print(f"  wrote {npz}")


def main() -> None:
    p = argparse.ArgumentParser(description="Checkerboard quench rollout of the marin-spin v1 model")
    p.add_argument("--checkpoint", default="gs://marin-eu-west4/grug/marin-spin-v1-c2e602/checkpoints/step-49400")
    p.add_argument("--condition-T", type=float, default=1.5, help="T_bin to condition the rollout on")
    p.add_argument("--n-windows", type=int, default=30, help=f"Number of {WINDOW_EVENTS}-event windows")
    p.add_argument("--n-chains", type=int, default=16, help="Parallel rollout chains (batch axis)")
    p.add_argument("--sample-temp", type=float, default=1.0, help="Sampling temperature over masked logits")
    p.add_argument("--gif-chain-every", type=int, default=10, help="Snapshot chain-0 every N events for the gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    outdir = args.outdir or f"scratch/quench_T{args.condition_T}"
    print(f"Loading model from {args.checkpoint} ...", flush=True)
    tok = build_tokenizer()
    # grug params/forward use raw PartitionSpecs (P("data","model") etc.), so model init, checkpoint
    # load, and every forward must run under grug's mesh. On one CPU device every axis is size 1.
    with set_mesh(compact_grug_mesh()):
        model = load_model(args.checkpoint)
        print(f"Model loaded. vocab={MODEL_CONFIG.vocab_size}, L={tok.L}, "
              f"T_id({args.condition_T})={tok.T_id(args.condition_T)}")
        print(f"Quench: checkerboard → T={args.condition_T}, {args.n_windows} windows x {WINDOW_EVENTS} events, "
              f"{args.n_chains} chains, sample_temp={args.sample_temp}\n", flush=True)

        result = rollout(
            model, tok,
            T=args.condition_T,
            n_windows=args.n_windows,
            n_chains=args.n_chains,
            sample_temp=args.sample_temp,
            gif_chain_every=args.gif_chain_every,
            rng=np.random.default_rng(args.seed),
        )
    print(f"\nFinal <|m|> = {result['abs_mag'][-1]:.3f}, <E/N> = {result['energy'][-1]:+.3f} "
          f"after {result['events_at'][-1]} events\n")
    write_outputs(result, outdir)


if __name__ == "__main__":
    main()
