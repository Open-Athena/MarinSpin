# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Ising BKL trajectory HDF5 files into a grug-ingestible, loss-masked token cache.

This is the v1 (tokenizing) data path for marin-spin. It reuses the verbatim-ported
``IsingTokenizer`` as the encoder of record and emits one prebuilt-cache record per window:

    {"id": ..., "input_ids": int32[SEQ_LEN], "loss_weights": float32[SEQ_LEN]}

The records are written as a single pool of ``.jsonl.gz`` shards. ``ising_tokenize_step`` wraps
the Marin tokenize pipeline (``PrebuiltLmDatasetFormat`` + ``passthrough`` tokenizer) to build the
Levanter ``TreeCache`` from those shards; ``marin_spin.data`` then turns the step
into an ``LmDataConfig`` that grug/base consumes. Train/val splitting is delegated to Levanter via
``num_validation_sequences`` (see ``data.py``).

Grammar / loss mask (v1, L=16, W=50): each window is
``[T_bin] [pos,spin]x2N [T_bin,pos,dt]xW`` of native length ``1 + 4N + 3W = 1175``, right-padded to
``SEQ_LEN = 1280`` with a dedicated PAD token. Loss is scored only on the ``2W`` pos/dt event tokens.
Because every window is identical in structure and length, the grug per-position ``loss_weight`` is a
single constant vector (``LOSS_WEIGHT_TEMPLATE``), not a per-record computation.
"""

from __future__ import annotations

import json
import os
import zlib
from collections import defaultdict
from functools import lru_cache
from collections.abc import Iterator, Sequence

import fsspec
import numpy as np
from levanter.data.text import PrebuiltLmDatasetFormat
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import TokenizeConfig, tokenize

from marin_spin.augment import grid_transform, site_permutation
from marin_spin.ising_tokenizer import IsingTokenizer, open_h5

# The 8 elements of D4 as (rot90 count, reflect) — 4 rotations + 4 reflections.
_D4 = [(rot, flip) for rot in range(4) for flip in (False, True)]

# v1 fixes the lattice size and window size; L=32/64 are separate caches (different seq lengths).
LATTICE_L = 16
WINDOW_EVENTS = 50  # W
STRIDE = WINDOW_EVENTS  # non-overlapping windows

# Native window length from the grammar: 1 (T_bin) + 4*N (two [pos,spin] config copies) + 3*W events.
_N = LATTICE_L * LATTICE_L
NATIVE_WINDOW_LEN = 1 + 4 * _N + 3 * WINDOW_EVENTS  # L=16, W=50 -> 1175

# Padded sequence length the cache (and the v2 trainer's max_seq_len) must use. Round, TPU-friendly.
SEQ_LEN = 1280
assert SEQ_LEN >= NATIVE_WINDOW_LEN, f"SEQ_LEN={SEQ_LEN} < native window length {NATIVE_WINDOW_LEN}"

# Slot where the event section begins: after the T_bin + two config copies.
_CONTEXT_LEN = 1 + 4 * _N


@lru_cache(maxsize=None)
def _loss_weight_template(seq_len: int, context_len: int, window_events: int) -> np.ndarray:
    """The grug loss-weight vector shared by every window with this (seq_len, context_len) grammar.

    grug applies ``loss_weight[i]`` to the prediction at position ``i``, which targets token ``i+1``
    (``experiments/grug/base/model.py`` ``next_token_loss``). The event tokens (the ``2W`` pos/dt
    tokens) sit at fixed slots ``context_len + 3k + {1, 2}``; shifting left by one to grug's
    predict-``i``-targets-``i+1`` convention puts the weight on the *preceding* slots:

        template[context_len + 3k + 0] = 1.0   # T_bin slot predicts the pos token
        template[context_len + 3k + 1] = 1.0   # pos slot predicts the dt token

    Everything else (config copies, the dt->next-T_bin slot, PAD, the final position) stays 0.0.
    ``sum == 2 * window_events``. Memoized: one array per distinct grammar (context_len = tok.ctxlen).
    """
    template = np.zeros(seq_len, dtype=np.float32)
    k = np.arange(window_events)
    template[context_len + 3 * k + 0] = 1.0
    template[context_len + 3 * k + 1] = 1.0
    return template


def pad_id_for(tokenizer: IsingTokenizer) -> int:
    """PAD token id: appended just past the base vocab so PAD < vocab_size_for(tokenizer)."""
    return tokenizer.vocab_size


def vocab_size_for(tokenizer: IsingTokenizer) -> int:
    """Trainer-facing vocab size: base tokenizer vocab + 1 for PAD."""
    return tokenizer.vocab_size + 1


def encode_window(
    tokenizer: IsingTokenizer,
    T: float,
    spins: np.ndarray,
    positions: np.ndarray,
    delta_times: np.ndarray,
    *,
    seq_len: int = SEQ_LEN,
    pad_id: int | None = None,
) -> dict[str, np.ndarray]:
    """Encode one window into a prebuilt-cache record (pure compute, no I/O).

    Calls ``tokenizer.encode`` for the native window, right-pads ``input_ids`` to ``seq_len`` with
    ``pad_id``. ``loss_weights`` comes from the memoized template for this grammar (keyed on the
    tokenizer's ``ctxlen``, so it follows ``n_config_copies`` automatically).
    Raises ValueError if the native window exceeds ``seq_len``.
    """
    if pad_id is None:
        pad_id = pad_id_for(tokenizer)
    native = tokenizer.encode(T, spins, positions, delta_times)
    if native.shape[0] > seq_len:
        raise ValueError(f"native window length {native.shape[0]} exceeds seq_len {seq_len}")
    input_ids = np.full(seq_len, pad_id, dtype=np.int32)
    input_ids[: native.shape[0]] = native
    loss_weights = _loss_weight_template(seq_len, tokenizer.ctxlen, WINDOW_EVENTS)
    return {"input_ids": input_ids, "loss_weights": loss_weights}


def iter_window_records(
    h5_files: Sequence[str],
    tokenizer: IsingTokenizer,
    *,
    window_size: int = WINDOW_EVENTS,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
    only_converged: bool = True,
) -> Iterator[dict]:
    """Yield one prebuilt-cache record per window across all trajectories.

    Ports the incremental-flip window replay from ``experiments/marin-spin/dataset.py``: for each
    trajectory the current spin config at window start is reconstructed by applying the window's
    leading flips, so each window carries its own correct snapshot. No train/val split here — all
    windows go to one pool; Levanter carves validation later via ``num_validation_sequences``.
    """
    pad_id = pad_id_for(tokenizer)
    for path in sorted(h5_files):
        basename = os.path.basename(path)
        with open_h5(path) as f:
            T = float(f.attrs["T"])
            n_traj = int(f.attrs["n_traj"])
            for i in range(n_traj):
                traj = f[f"trajectories/{i}"]
                if only_converged and not bool(traj.attrs["converged"]):
                    continue
                positions = traj["positions"][:]
                delta_times = traj["delta_times"][:]
                n_events = positions.shape[0]

                spins = traj["initial_spins"][:].copy()
                spins_flat = spins.reshape(-1)  # shares memory with spins
                prev_k = 0
                for k in range(0, n_events - window_size + 1, stride):
                    for j in range(prev_k, k):  # apply flips prev_k..k-1 incrementally
                        spins_flat[positions[j]] *= -1
                    prev_k = k
                    record = encode_window(
                        tokenizer,
                        T,
                        spins,
                        positions[k : k + window_size],
                        delta_times[k : k + window_size],
                        seq_len=seq_len,
                        pad_id=pad_id,
                    )
                    yield {
                        "id": f"{basename}:{i}:{k}",
                        "input_ids": record["input_ids"].tolist(),
                        "loss_weights": record["loss_weights"].tolist(),
                    }


def _augmented_records(
    tokenizer: IsingTokenizer,
    T: float,
    spins: np.ndarray,
    positions: np.ndarray,
    delta_times: np.ndarray,
    rec_id: str,
    *,
    seq_len: int,
) -> Iterator[dict]:
    """Yield the 8 D4-orientation records (each with an independent random toroidal shift) for one window.

    Each variant applies a lattice symmetry g (rotation/reflection + shift) to the config and the event
    positions — a label-preserving transform (Ising rates are isotropic) that teaches the model the
    square-lattice symmetry the row-major raster otherwise breaks. ``delta_times`` are unchanged (timing
    is symmetry-invariant), so re-encoding reproduces the identical dt tokens.
    """
    L = tokenizer.L
    pad_id = pad_id_for(tokenizer)
    rng = np.random.default_rng(zlib.crc32(rec_id.encode()))  # deterministic per window (stable across runs)
    for v, (rot, flip) in enumerate(_D4):
        shift = (int(rng.integers(L)), int(rng.integers(L)))
        perm, _ = site_permutation(L, rot, flip, shift)
        rec = encode_window(
            tokenizer, T, grid_transform(spins, rot, flip, shift), perm[positions], delta_times,
            seq_len=seq_len, pad_id=pad_id,
        )
        yield {"id": f"{rec_id}:a{v}", "input_ids": rec["input_ids"].tolist(),
               "loss_weights": rec["loss_weights"].tolist()}


def temp_tag(T: float) -> str:
    """Stable per-temperature tag used for W&B eval panels, e.g. ``T1.50`` -> ``eval/T1.50/loss``."""
    return f"T{T:.2f}"


def iter_split_window_records(
    h5_files: Sequence[str],
    tokenizer: IsingTokenizer,
    *,
    window_size: int = WINDOW_EVENTS,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
    val_frac: float = 0.1,
    only_converged: bool = True,
    seed: int = 0,
    augment_train: bool = False,
) -> Iterator[tuple[str, str, dict]]:
    """Yield ``(split, temp_tag, record)`` with a group-level train/val split per temperature.

    Within each temperature file, trajectories are grouped by their initial spin config and whole
    groups are assigned to train or val (deterministic shuffle), so windows from one trajectory never
    straddle the split. ``split`` is ``"train"`` or ``"val"``; ``temp_tag`` is :func:`temp_tag`.

    With ``augment_train`` each train window is emitted as its 8 D4 orientations (each with a random
    toroidal shift); validation is always canonical so eval loss stays comparable to the oracle floor.
    """
    pad_id = pad_id_for(tokenizer)
    rng = np.random.default_rng(seed)
    for path in sorted(h5_files):
        basename = os.path.basename(path)
        with open_h5(path) as f:
            T = float(f.attrs["T"])
            tag = temp_tag(T)
            n_traj = int(f.attrs["n_traj"])

            groups: dict[bytes, list[int]] = defaultdict(list)
            for i in range(n_traj):
                traj = f[f"trajectories/{i}"]
                if only_converged and not bool(traj.attrs["converged"]):
                    continue
                groups[traj["initial_spins"][:].tobytes()].append(i)

            group_keys = list(groups)
            rng.shuffle(group_keys)
            n_val = max(1, round(len(group_keys) * val_frac)) if group_keys else 0
            val_keys = set(group_keys[:n_val])

            for key in group_keys:
                split = "val" if key in val_keys else "train"
                for i in groups[key]:
                    traj = f[f"trajectories/{i}"]
                    positions = traj["positions"][:]
                    delta_times = traj["delta_times"][:]
                    n_events = positions.shape[0]
                    spins = traj["initial_spins"][:].copy()
                    spins_flat = spins.reshape(-1)
                    prev_k = 0
                    for k in range(0, n_events - window_size + 1, stride):
                        for j in range(prev_k, k):
                            spins_flat[positions[j]] *= -1
                        prev_k = k
                        rec_id = f"{basename}:{i}:{k}"
                        win_pos = positions[k : k + window_size]
                        win_dt = delta_times[k : k + window_size]
                        if augment_train and split == "train":
                            for aug in _augmented_records(tokenizer, T, spins, win_pos, win_dt, rec_id,
                                                          seq_len=seq_len):
                                yield split, tag, aug
                        else:
                            record = encode_window(tokenizer, T, spins, win_pos, win_dt,
                                                   seq_len=seq_len, pad_id=pad_id)
                            yield split, tag, {
                                "id": rec_id,
                                "input_ids": record["input_ids"].tolist(),
                                "loss_weights": record["loss_weights"].tolist(),
                            }


class _ShardWriter:
    """Streaming writer of JSON records to ``<output_dir>/part-NNNNN.jsonl.gz`` shards.

    Buffers at most ``shard_size`` records in memory before flushing — so total memory stays bounded
    regardless of dataset size (the full window set is ~25 GB if accumulated).
    """

    def __init__(self, output_dir: str, *, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.urls: list[str] = []
        self._buffer: list[dict] = []
        fs, _ = fsspec.core.url_to_fs(output_dir)
        fs.makedirs(output_dir, exist_ok=True)

    def add(self, record: dict) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        url = os.path.join(self.output_dir, f"part-{len(self.urls):05d}.jsonl.gz")
        with fsspec.open(url, "wt", compression="gzip") as out:
            for rec in self._buffer:
                out.write(json.dumps(rec) + "\n")
        self.urls.append(url)
        self._buffer.clear()


def write_ising_splits(
    h5_files: Sequence[str],
    tokenizer: IsingTokenizer,
    output_dir: str,
    *,
    window_size: int = WINDOW_EVENTS,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
    val_frac: float = 0.1,
    only_converged: bool = True,
    shard_size: int = 10_000,
    seed: int = 0,
    augment_train: bool = False,
) -> tuple[list[str], dict[str, list[str]]]:
    """Write a single train shard pool plus one validation pool per temperature.

    Layout: ``<output_dir>/train/part-*.jsonl.gz`` and ``<output_dir>/val/<temp_tag>/part-*.jsonl.gz``.
    Returns ``(train_urls, {temp_tag: val_urls})``. Records are streamed to disk (bounded memory):
    the train pool flushes continuously, and each temperature's val writer is flushed when the next
    temperature begins (files are processed one temperature at a time).

    ``augment_train`` emits each train window in its 8 D4 orientations (random shift each) — ~8× the
    train pool — to make the model lattice-symmetry equivariant; validation stays canonical.
    """
    train_writer = _ShardWriter(os.path.join(output_dir, "train"), shard_size=shard_size)
    val_writers: dict[str, _ShardWriter] = {}
    prev_val_tag: str | None = None

    for split, tag, record in iter_split_window_records(
        h5_files,
        tokenizer,
        window_size=window_size,
        stride=stride,
        seq_len=seq_len,
        val_frac=val_frac,
        only_converged=only_converged,
        seed=seed,
        augment_train=augment_train,
    ):
        if split == "train":
            train_writer.add(record)
            continue
        if prev_val_tag is not None and tag != prev_val_tag:
            val_writers[prev_val_tag].flush()  # done with the previous temperature's val pool
        prev_val_tag = tag
        if tag not in val_writers:
            val_writers[tag] = _ShardWriter(os.path.join(output_dir, "val", tag), shard_size=shard_size)
        val_writers[tag].add(record)

    train_writer.flush()
    for writer in val_writers.values():
        writer.flush()
    return train_writer.urls, {tag: val_writers[tag].urls for tag in sorted(val_writers)}


def write_ising_shards(
    h5_files: Sequence[str],
    tokenizer: IsingTokenizer,
    output_dir: str,
    *,
    window_size: int = WINDOW_EVENTS,
    stride: int = STRIDE,
    seq_len: int = SEQ_LEN,
    only_converged: bool = True,
    shard_size: int = 10_000,
) -> list[str]:
    """Replay every trajectory into windows and write a single pool of ``.jsonl.gz`` shards.

    Returns the list of written shard URLs (``<output_dir>/part-NNNNN.jsonl.gz``). I/O is here;
    per-window compute lives in ``encode_window`` / ``iter_window_records``.
    """
    fs, _ = fsspec.core.url_to_fs(output_dir)
    fs.makedirs(output_dir, exist_ok=True)

    shard_urls: list[str] = []
    shard_idx = 0
    buffer: list[dict] = []

    def flush() -> None:
        nonlocal shard_idx
        if not buffer:
            return
        url = os.path.join(output_dir, f"part-{shard_idx:05d}.jsonl.gz")
        with fsspec.open(url, "wt", compression="gzip") as out:
            for rec in buffer:
                out.write(json.dumps(rec) + "\n")
        shard_urls.append(url)
        shard_idx += 1
        buffer.clear()

    for record in iter_window_records(
        h5_files,
        tokenizer,
        window_size=window_size,
        stride=stride,
        seq_len=seq_len,
        only_converged=only_converged,
    ):
        buffer.append(record)
        if len(buffer) >= shard_size:
            flush()
    flush()
    return shard_urls


def list_ising_shards(output_dir: str) -> list[str]:
    """List the ``part-*.jsonl.gz`` shard pool previously written by ``write_ising_shards``.

    Lets the training launch reconstruct the exact tokenize step (same shard URLs -> same executor
    version hash -> same cache path) without rewriting the shards.
    """
    fs, _ = fsspec.core.url_to_fs(output_dir)
    proto = f"{output_dir.split('://', 1)[0]}://" if "://" in output_dir else ""
    matches = fs.glob(f"{output_dir.rstrip('/')}/part-*.jsonl.gz")
    urls = [p if (not proto or str(p).startswith(proto)) else f"{proto}{p}" for p in matches]
    if not urls:
        raise FileNotFoundError(f"No part-*.jsonl.gz shards under {output_dir}; run write_ising_shards first")
    return sorted(urls)


def list_ising_split_pools(output_dir: str) -> tuple[list[str], dict[str, list[str]]]:
    """List the train pool and per-temperature val pools written by :func:`write_ising_splits`.

    Returns ``(train_urls, {temp_tag: val_urls})`` from ``<output_dir>/train`` and
    ``<output_dir>/val/<temp_tag>``. Lets the training launch rebuild the exact tokenize steps
    (same shard URLs -> same cache paths) without rewriting shards.
    """
    fs, _ = fsspec.core.url_to_fs(output_dir)
    proto = f"{output_dir.split('://', 1)[0]}://" if "://" in output_dir else ""
    train_urls = list_ising_shards(os.path.join(output_dir, "train"))
    val_by_temp: dict[str, list[str]] = defaultdict(list)
    for p in fs.glob(f"{output_dir.rstrip('/')}/val/*/part-*.jsonl.gz"):
        full = p if (not proto or str(p).startswith(proto)) else f"{proto}{p}"
        tag = os.path.basename(os.path.dirname(full.rstrip("/")))
        val_by_temp[tag].append(full)
    if not val_by_temp:
        raise FileNotFoundError(f"No per-temperature val pools under {output_dir}/val; run write_ising_splits first")
    return train_urls, {tag: sorted(urls) for tag, urls in sorted(val_by_temp.items())}


def ising_tokenize_step(
    shard_urls: Sequence[str],
    *,
    name: str = "marin_spin",
    tags: Sequence[str] | None = None,
    is_validation: bool = False,
) -> ExecutorStep:
    """Wrap the Marin tokenize pipeline to build a Levanter TreeCache from prebuilt Ising shards.

    Reads the ``{input_ids, loss_weights}`` ``.jsonl.gz`` pool via ``PrebuiltLmDatasetFormat`` (no real
    tokenizer; ``passthrough`` is resolved by the ``load_tokenizer`` guard). ``tags`` are attached to
    the resulting dataset component and become the W&B eval tag for per-temperature validation panels.
    ``is_validation`` routes the shards to ``validation_paths`` (so the cache's *validation* split is
    populated, which is what ``tagged_eval_sets`` reads) instead of ``train_paths``.
    """
    urls = list(shard_urls)
    return ExecutorStep(
        name=os.path.join("tokenized", name),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[] if is_validation else urls,
            validation_paths=urls if is_validation else [],
            cache_path=this_output_path(),
            tokenizer="passthrough",
            format=PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights"),
            tags=list(tags) if tags else [],
            # Shard paths may legitimately contain "test"/"validation" substrings (e.g. the val pool
            # directory), so don't reject them.
            allow_test_in_train=True,
        ),
    )


def ising_val_steps_by_temp(val_urls_by_temp: dict[str, Sequence[str]]) -> dict[str, ExecutorStep]:
    """Build one tagged validation tokenize step per temperature, keyed by temp tag (e.g. ``T1.50``).

    Each step is tagged with its temperature and routes its shards to the validation split, so grug's
    tagged evaluator emits ``eval/<T>/loss`` panels.
    """
    return {
        tag: ising_tokenize_step(urls, name=f"marin_spin_val_{tag}", tags=[tag], is_validation=True)
        for tag, urls in sorted(val_urls_by_temp.items())
    }
