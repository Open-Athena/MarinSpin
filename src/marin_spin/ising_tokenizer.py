# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer for Ising BKL trajectories.

Vocabulary layout (contiguous integer IDs):
  [T_bin_0 .. T_bin_{n_temps-1}]      one token per training temperature
  [spin_up, spin_down]                 +1 / -1
  [pos_0 .. pos_{L²-1}]               flat spin index
  [dt_underflow, dt_bin_0..dt_bin_{B-2}, dt_overflow]   log-binned Δt + overflow bins

Sequence format — one WINDOW of W events starting from spin config at step k:
  [T_bin]
  [pos_0][spin_0] ... [pos_{N-1}][spin_{N-1}]        ← config at step k (copy 1, context)
  [pos_0][spin_0] ... [pos_{N-1}][spin_{N-1}]        ← config at step k (copy 2, given)
  [T_bin][pos_k][dt_k] ... [T_bin][pos_{k+W-1}][dt_{k+W-1}]  ← W events to predict

T_bin is repeated before every (pos, dt) event pair so the temperature conditioning
signal is always at most 2 positions away from any event token.  Since BKL is Markov
and each window is statistically independent, this is equivalent to conditioning every
short trajectory on T — the model never needs to attend far back to find T.

The config is the CURRENT spin state at the window start, not always the initial config.
It is duplicated so the model sees the full configuration before the event sequence begins,
avoiding any rasterization-order bias in the causal mask.

Loss is computed only on pos and dt tokens; T_bin and all config tokens are pure context.

Window length: 1 + 4*L² + 3*W  tokens
  L=16, W=50: 1 + 1024 + 150  = 1175
  L=32, W=50: 1 + 4096 + 150  = 4247
  L=64, W=50: 1 + 16384 + 150 = 16535
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator

import fsspec
import h5py
import numpy as np


@contextlib.contextmanager
def open_h5(path: str) -> Iterator[h5py.File]:
    """Open an HDF5 file for reading, downloading remote (e.g. ``gs://``) paths to a local temp.

    h5py cannot read object-store URLs directly and HDF5 access is seek-heavy, so remote files are
    staged to local disk first (fast and same-region on a co-located worker).
    """
    if "://" not in path:
        with h5py.File(path, "r") as f:
            yield f
        return
    with fsspec.open(path, "rb") as src, tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(src.read())
        tmp_path = tmp.name
    try:
        with h5py.File(tmp_path, "r") as f:
            yield f
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class IsingTokenizer:
    """
    Maps Ising trajectory elements to integer token IDs and back.

    Parameters
    ----------
    temps      : sorted list of training temperatures
    L          : lattice side length
    n_dt_bins  : number of interior Δt bins (excluding overflow bins)
    dt_edges   : (n_dt_bins + 1,) array of bin edges in linear space
                 dt_edges[0] is the underflow boundary,
                 dt_edges[-1] is the overflow boundary.
    """

    SPIN_UP = +1
    SPIN_DOWN = -1

    def __init__(
        self,
        temps: list[float],
        L: int,
        n_dt_bins: int,
        dt_edges: np.ndarray,
        n_config_copies: int = 2,
    ):
        self.temps = sorted(temps)
        self.L = L
        self.N = L * L
        self.n_dt_bins = n_dt_bins
        self.dt_edges = np.asarray(dt_edges, dtype=np.float64)
        # Number of times the initial config raster is written in the context. The original grammar
        # duplicated it (2); the de-duplicated grammar writes it once (1). Vocab is identical either way.
        self.n_config_copies = n_config_copies

        # Total Δt tokens = n_dt_bins interior + 1 underflow + 1 overflow
        self.n_dt_tokens = n_dt_bins + 2

        # ---- ID ranges ----
        self.T_OFFSET = 0
        self.SPIN_OFFSET = len(self.temps)
        self.POS_OFFSET = self.SPIN_OFFSET + 2
        self.DT_OFFSET = self.POS_OFFSET + self.N

        self.vocab_size = self.DT_OFFSET + self.n_dt_tokens

        # Context length: 1 (T_bin) + n_config_copies copies of the 2N-token [pos,spin] raster.
        self.ctxlen = 1 + 2 * self.N * self.n_config_copies

        # Convenience maps
        self._temp_to_id = {round(t, 10): i for i, t in enumerate(self.temps)}

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    def T_id(self, T: float) -> int:
        """Token ID for temperature T."""
        key = min(self._temp_to_id, key=lambda t: abs(t - T))
        return self.T_OFFSET + self._temp_to_id[key]

    def spin_id(self, s: int) -> int:
        """Token ID for spin value +1 or -1."""
        return self.SPIN_OFFSET + (0 if s == self.SPIN_UP else 1)

    def pos_id(self, idx: int) -> int:
        """Token ID for flat position index."""
        return self.POS_OFFSET + idx

    def _dt_ids_vec(self, dts: np.ndarray) -> np.ndarray:
        """Vectorised version of dt_id over an array of delta-times."""
        dts = np.asarray(dts, dtype=np.float64)
        ids = np.searchsorted(self.dt_edges, dts, side="right")  # 0..n_dt_bins
        ids = np.clip(ids, 0, self.n_dt_bins)  # interior bins → 1..n_dt_bins
        # underflow: dts < dt_edges[0]  → ids==0 after searchsorted → DT_OFFSET + 0
        # overflow:  dts >= dt_edges[-1] → ids==n_dt_bins → DT_OFFSET + n_dt_tokens - 1
        result = np.where(
            dts < self.dt_edges[0],
            self.DT_OFFSET,
            np.where(dts >= self.dt_edges[-1], self.DT_OFFSET + self.n_dt_tokens - 1, self.DT_OFFSET + ids),
        )
        return result.astype(np.int32)

    def dt_id(self, dt: float) -> int:
        """
        Token ID for a time increment.
        dt_edges defines the interior bin boundaries:
          dt < dt_edges[0]             → underflow  (DT_OFFSET + 0)
          dt_edges[k] ≤ dt < dt_edges[k+1] → bin k+1   (DT_OFFSET + 1 + k)
          dt ≥ dt_edges[-1]            → overflow   (DT_OFFSET + n_dt_tokens - 1)
        """
        if dt < self.dt_edges[0]:
            return self.DT_OFFSET
        if dt >= self.dt_edges[-1]:
            return self.DT_OFFSET + self.n_dt_tokens - 1
        k = int(np.searchsorted(self.dt_edges, dt, side="right")) - 1
        k = max(0, min(k, self.n_dt_bins - 1))
        return self.DT_OFFSET + 1 + k

    # ------------------------------------------------------------------
    # Inverse maps (for decoding)
    # ------------------------------------------------------------------

    def id_to_T(self, tok: int) -> float:
        return self.temps[tok - self.T_OFFSET]

    def id_to_spin(self, tok: int) -> int:
        return self.SPIN_UP if (tok - self.SPIN_OFFSET) == 0 else self.SPIN_DOWN

    def id_to_pos(self, tok: int) -> int:
        return tok - self.POS_OFFSET

    def id_to_dt_center(self, tok: int) -> float:
        """Return the log-midpoint of the bin (or edge value for over/underflow)."""
        k = tok - self.DT_OFFSET
        if k == 0:
            return float(self.dt_edges[0]) * 0.5  # underflow: half the lower edge
        if k == self.n_dt_tokens - 1:
            return float(self.dt_edges[-1]) * 2.0  # overflow: double the upper edge
        lo, hi = self.dt_edges[k - 1], self.dt_edges[k]
        return float(np.exp(0.5 * (np.log(lo) + np.log(hi))))

    @property
    def dt_centers(self) -> np.ndarray:
        """Log-midpoints for all dt tokens, shape (n_dt_tokens,)."""
        centers = np.empty(self.n_dt_tokens, dtype=np.float64)
        for k in range(self.n_dt_tokens):
            centers[k] = self.id_to_dt_center(self.DT_OFFSET + k)
        return centers

    def sample_dt(self, tok: int, rng: np.random.Generator) -> float:
        """
        Sample a continuous dt log-uniformly within the chosen bin.

        For interior bins: dt ~ exp(Uniform(log(lo), log(hi)))
        For underflow/overflow: falls back to the geometric midpoint (unbounded bins).
        """
        k = tok - self.DT_OFFSET
        if k == 0:
            return float(self.dt_edges[0]) * 0.5  # underflow fallback
        if k == self.n_dt_tokens - 1:
            return float(self.dt_edges[-1]) * 2.0  # overflow fallback
        lo, hi = self.dt_edges[k - 1], self.dt_edges[k]
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def token_type(self, tok: int) -> str:
        if self.T_OFFSET <= tok < self.SPIN_OFFSET:
            return "T_bin"
        if self.SPIN_OFFSET <= tok < self.POS_OFFSET:
            return "spin"
        if self.POS_OFFSET <= tok < self.DT_OFFSET:
            return "pos"
        if self.DT_OFFSET <= tok < self.DT_OFFSET + self.n_dt_tokens:
            return "dt_bin"
        return "unknown"

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        T: float,
        initial_spins: np.ndarray,
        positions: np.ndarray,
        delta_times: np.ndarray,
    ) -> np.ndarray:
        """
        Encode one trajectory window to a 1-D integer token array.

        Format: [T_bin] [config x n_config_copies] [T_bin][pos][dt] x n_events

        T_bin is repeated before every event pair so the temperature signal is
        always <=2 positions away from any event token.

        Returns int32 array of length self.ctxlen + 3*n_events.
        """
        spins_flat = initial_spins.flatten()
        n_events = len(positions)
        total = self.ctxlen + 3 * n_events
        tokens = np.empty(total, dtype=np.int32)

        tokens[0] = self.T_id(T)

        # Initial config: vectorised interleaved [pos][spin] pairs, written n_config_copies times.
        pos_ids = np.arange(self.N, dtype=np.int32) + self.POS_OFFSET
        spin_ids = np.where(spins_flat == self.SPIN_UP, self.SPIN_OFFSET, self.SPIN_OFFSET + 1).astype(np.int32)
        for copy in range(self.n_config_copies):
            base = 1 + copy * 2 * self.N
            tokens[base : base + 2 * self.N : 2] = pos_ids
            tokens[base + 1 : base + 2 * self.N : 2] = spin_ids

        # Event sequence: [T_bin][pos][dt] triplets
        offset = self.ctxlen
        t_id = self.T_id(T)
        event_pos = positions.astype(np.int32) + self.POS_OFFSET
        event_dt = self._dt_ids_vec(delta_times)
        tokens[offset : offset + 3 * n_events : 3] = t_id  # T_bin repeated
        tokens[offset + 1 : offset + 3 * n_events : 3] = event_pos  # pos
        tokens[offset + 2 : offset + 3 * n_events : 3] = event_dt  # dt

        return tokens

    def decode(self, tokens: np.ndarray) -> dict:
        """
        Decode a token array back to trajectory components.
        Returns dict with keys: T, initial_spins, positions, delta_times.
        """
        T = self.id_to_T(int(tokens[0]))

        # Read initial config from the first copy (any further copies are identical — skip them)
        spins_flat = np.empty(self.N, dtype=np.int8)
        for i in range(self.N):
            spins_flat[i] = self.id_to_spin(int(tokens[1 + 2 * i + 1]))
        initial_spins = spins_flat.reshape(self.L, self.L)

        # Event section: [T_bin][pos][dt] triplets
        offset = self.ctxlen
        n_events = (len(tokens) - offset) // 3
        positions = np.array([self.id_to_pos(int(tokens[offset + 3 * k + 1])) for k in range(n_events)], dtype=np.int32)
        delta_times = np.array(
            [self.id_to_dt_center(int(tokens[offset + 3 * k + 2])) for k in range(n_events)], dtype=np.float64
        )

        return {"T": T, "initial_spins": initial_spins, "positions": positions, "delta_times": delta_times}

    # ------------------------------------------------------------------
    # Serialise / deserialise
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        config = {
            "temps": self.temps,
            "L": self.L,
            "n_dt_bins": self.n_dt_bins,
            "dt_edges": self.dt_edges.tolist(),
            "n_config_copies": self.n_config_copies,
        }
        with fsspec.open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "IsingTokenizer":
        with fsspec.open(path, "r") as f:
            cfg = json.load(f)
        return cls(
            temps=cfg["temps"],
            L=cfg["L"],
            n_dt_bins=cfg["n_dt_bins"],
            dt_edges=np.array(cfg["dt_edges"]),
            n_config_copies=cfg.get("n_config_copies", 2),
        )

    def __repr__(self) -> str:
        return (
            f"IsingTokenizer(L={self.L}, temps={self.temps}, "
            f"vocab_size={self.vocab_size}, "
            f"n_dt_tokens={self.n_dt_tokens} "
            f"[1 underflow + {self.n_dt_bins} interior + 1 overflow])"
        )


# ---------------------------------------------------------------------------
# Vocab builder  (scans HDF5 files to compute Δt bin edges)
# ---------------------------------------------------------------------------


def build_tokenizer(
    h5_files: list[str],
    L: int,
    n_dt_bins: int = 64,
    dt_clip_lo_pct: float = 0.1,
    dt_clip_hi_pct: float = 99.9,
    sample_trajs: int = 200,
) -> IsingTokenizer:
    """
    Scan `h5_files` to collect Δt samples, compute log-uniform bin edges,
    and return a fitted IsingTokenizer.
    """
    temps_seen = set()
    dt_samples = []

    for path in h5_files:
        with open_h5(path) as f:
            T = float(f.attrs["T"])
            temps_seen.add(round(T, 6))
            n = min(sample_trajs, int(f.attrs["n_traj"]))
            for i in range(n):
                dt_samples.append(f[f"trajectories/{i}/delta_times"][:])

    all_dts = np.concatenate(dt_samples)
    lo = np.percentile(all_dts, dt_clip_lo_pct)
    hi = np.percentile(all_dts, dt_clip_hi_pct)
    dt_edges = np.exp(np.linspace(np.log(lo), np.log(hi), n_dt_bins + 1))

    return IsingTokenizer(
        temps=sorted(temps_seen),
        L=L,
        n_dt_bins=n_dt_bins,
        dt_edges=dt_edges,
    )
