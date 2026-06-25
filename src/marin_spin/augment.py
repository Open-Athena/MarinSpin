# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lattice-symmetry data augmentation in token space for marin-spin v1 windows.

The periodic square lattice's symmetry group — translations (L² toroidal shifts) × D4 (4 rotations ×
2 reflections) — leaves the Ising energy and the BKL transition rates invariant, so applying a random
group element ``g`` to a tokenized window is a label-preserving augmentation. We apply it in TOKEN
space (no dt re-binning): ``g`` induces a permutation π of the N flat site indices, and we

  1. reorder the two config copies' spin tokens into the new raster order
     (new raster slot j holds the old spin at π⁻¹(j); the pos tokens stay POS_OFFSET+0..N-1), and
  2. remap each event's position token: old site i → POS_OFFSET + π(i),

leaving T_bin, dt tokens, padding, and the loss-weight mask untouched (structure is identical).

Training with this makes the model (approximately) translation- and rotation-equivariant, removing the
raster anisotropy (horizontal neighbors are ~2 tokens apart, vertical ~2L — rotations average it out).
"""

from __future__ import annotations

import numpy as np


def grid_transform(grid: np.ndarray, rot: int, flip: bool, shift: tuple[int, int]) -> np.ndarray:
    """Apply a periodic-square-lattice symmetry to an L×L array: rot90×``rot``, optional reflection, roll.

    ``rot`` ∈ {0,1,2,3} and ``flip`` (a transpose-style reflection) together generate the 8-element D4;
    ``shift`` = (dr, dc) is a toroidal translation. Composition order is fixed (rotate, reflect, shift)
    so the same (rot, flip, shift) yields one well-defined group element.
    """
    g = np.rot90(grid, rot)
    if flip:
        g = g.T
    return np.roll(g, shift, axis=(0, 1))


def site_permutation(L: int, rot: int, flip: bool, shift: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (perm, inv_perm) on flat indices for the transform: perm[i_old]=new slot, inv_perm[j]=old index."""
    idx = np.arange(L * L).reshape(L, L)
    inv_perm = grid_transform(idx, rot, flip, shift).ravel()  # new slot j ← old index inv_perm[j]
    perm = np.argsort(inv_perm)  # old index i → new slot perm[i] (inverse of the permutation inv_perm)
    return perm, inv_perm


def sample_transform(rng: np.random.Generator, L: int) -> tuple[int, bool, tuple[int, int]]:
    """Uniformly sample a group element (rot, flip, shift) from translations × D4."""
    return int(rng.integers(4)), bool(rng.integers(2)), (int(rng.integers(L)), int(rng.integers(L)))


def augment_window_tokens(
    input_ids: np.ndarray,
    perm: np.ndarray,
    inv_perm: np.ndarray,
    *,
    n_sites: int,
    pos_offset: int,
    context_len: int,
    window_events: int,
) -> np.ndarray:
    """Apply the site permutation to one tokenized window's config copies + event position tokens.

    ``input_ids`` is the full padded window (``[T_bin][pos,spin]×2N [T_bin,pos,dt]×W`` then PAD).
    Returns a new array; T_bin, dt tokens, and padding are copied unchanged. ``perm``/``inv_perm`` come
    from :func:`site_permutation`.
    """
    out = input_ids.copy()
    n = n_sites
    for copy in range(2):
        base = 1 + copy * 2 * n
        spins = input_ids[base + 1 : base + 2 * n : 2]  # spin token per raster slot j
        out[base + 1 : base + 2 * n : 2] = spins[inv_perm]  # new slot j ← old slot inv_perm[j]
        # pos tokens (base : base+2n : 2) stay POS_OFFSET+0..N-1 in raster order — unchanged
    for k in range(window_events):
        slot = context_len + 3 * k + 1  # event pos token slot
        out[slot] = pos_offset + perm[input_ids[slot] - pos_offset]
    return out


# ---------------------------------------------------------------------------
# Self-test: token-space augmentation must equal re-encoding the transformed trajectory.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from marin_spin.ising_tokenizer import IsingTokenizer
    from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS, encode_window

    L, N, W = LATTICE_L, LATTICE_L * LATTICE_L, WINDOW_EVENTS
    dt_edges = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 65))
    tok = IsingTokenizer(temps=[1.5, 1.7, 2.0, 3.5], L=L, n_dt_bins=64, dt_edges=dt_edges)
    ctx = 1 + 4 * N
    rng = np.random.default_rng(0)

    n_ok = 0
    for trial in range(200):
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
        positions = rng.integers(0, N, size=W).astype(np.int32)
        dts = np.exp(rng.uniform(np.log(1e-2), np.log(1e1), size=W))
        rec = encode_window(tok, 1.7, spins, positions, dts)
        ids = rec["input_ids"]

        rot, flip, shift = sample_transform(rng, L)
        perm, inv_perm = site_permutation(L, rot, flip, shift)
        aug = augment_window_tokens(ids, perm, inv_perm, n_sites=N, pos_offset=tok.POS_OFFSET,
                                    context_len=ctx, window_events=W)

        # Ground truth: re-encode the trajectory after applying g to the config grid and the positions.
        spins_g = grid_transform(spins, rot, flip, shift)
        pos_g = perm[positions]
        ref = encode_window(tok, 1.7, spins_g, pos_g, dts)["input_ids"]
        assert np.array_equal(aug, ref), f"trial {trial}: token-space augment != re-encode"

        # Identity element must be a no-op.
        if (rot, flip, shift) == (0, False, (0, 0)):
            assert np.array_equal(aug, ids)

        # Decoded config must equal the grid transform; decoded positions the permuted positions.
        # decode() expects the native (unpadded) window, so slice off the right-padding.
        dec = tok.decode(aug[: 1 + 4 * N + 3 * W])
        assert np.array_equal(dec["initial_spins"], spins_g)
        assert np.array_equal(dec["positions"], pos_g)
        # Loss-bearing structure (T_bin + dt tokens) untouched.
        assert aug[0] == ids[0]
        assert np.array_equal(aug[ctx + 3 * np.arange(W) + 2], ids[ctx + 3 * np.arange(W) + 2])
        n_ok += 1

    # Identity over the full group: composing is a permutation (every site hit once).
    perm0, inv0 = site_permutation(L, 0, False, (0, 0))
    assert np.array_equal(perm0, np.arange(N)) and np.array_equal(inv0, np.arange(N))
    print(f"OK: {n_ok} trials — token-space augmentation == re-encode, decode round-trips, dt/T_bin preserved.")
