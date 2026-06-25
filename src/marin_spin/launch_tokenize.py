# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the marin-spin v1 token cache from Ising BKL trajectory HDF5 files.

Two stages:
  1. Eager (CPU): fit the IsingTokenizer on the HDF5 data and write the prebuilt-cache shard pool
     (``{input_ids, loss_weights}`` ``.jsonl.gz``) to a staging path.
  2. Executor: run the Marin tokenize step (``PrebuiltLmDatasetFormat`` + ``passthrough``) to build the
     versioned Levanter ``TreeCache`` the v2 trainer will consume.

Usage (local CPU run):

    uv run python -m marin_spin.launch_tokenize \
        --data_dir gs://<bucket>/ising/L16 \
        --staging_dir gs://<bucket>/ising/L16-shards \
        --prefix gs://<bucket>/marin

``--data_dir`` must contain ``ising_L16_T*.h5``. Cache build is CPU-only (no TPU). Training on the
resulting cache is the v2 model port (out of scope here). Hand the step to
``make_marin_spin_data_config`` to get the ``LmDataConfig`` for grug/base.
"""

import argparse

import fsspec
from marin.execution.executor import ExecutorMainConfig, executor_main

from marin_spin.ising_tokenizer import IsingTokenizer, build_tokenizer
from marin_spin.tokenize_ising import (
    LATTICE_L,
    ising_tokenize_step,
    ising_val_steps_by_temp,
    list_ising_split_pools,
    write_ising_splits,
)


def _list_h5(data_dir: str) -> list[str]:
    fs, _ = fsspec.core.url_to_fs(data_dir)
    proto = data_dir.split("://", 1)[0] + "://" if "://" in data_dir else ""
    pattern = f"{data_dir.rstrip('/')}/*ising_L{LATTICE_L}_T*.h5"
    paths = [f"{proto}{p}" if proto and not str(p).startswith(proto) else str(p) for p in fs.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"No ising_L{LATTICE_L}_T*.h5 files under {data_dir}")
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the marin-spin v1 token cache")
    parser.add_argument("--data_dir", required=True, help="Dir containing ising_L16_T*.h5")
    parser.add_argument("--staging_dir", required=True, help="Where to write the .jsonl.gz shard pool")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Executor output prefix. Omit on Iris to use the worker's regional marin_prefix() "
        "(gs://marin-{region}) — the same prefix the training job resolves, so the cache lines up.",
    )
    parser.add_argument("--name", default="marin_spin_train", help="Train tokenize step name suffix")
    parser.add_argument(
        "--tokenizer_json",
        default=None,
        help="Pin the tokenizer (temps + dt_edges) to a saved JSON instead of refitting from --data_dir. "
        "Use to keep dt binning identical to a prior run when the data mix changes (e.g. adding hot trajectories).",
    )
    parser.add_argument("--shard_size", type=int, default=10_000)
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of per-T config groups held for val")
    parser.add_argument(
        "--skip_write",
        action="store_true",
        help="Reuse existing shards in --staging_dir (skip rewriting); only (re)build the caches.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Emit each train window in its 8 D4 orientations (random shift each) for lattice-symmetry "
        "equivariance (~8x train pool). Validation stays canonical.",
    )
    args = parser.parse_args()

    if args.skip_write:
        train_urls, val_urls_by_temp = list_ising_split_pools(args.staging_dir)
        print(f"Reusing existing shards: train pool ({len(train_urls)} shards) + {len(val_urls_by_temp)} val pools")
    else:
        h5_files = _list_h5(args.data_dir)
        if args.tokenizer_json:
            tokenizer = IsingTokenizer.load(args.tokenizer_json)
            print(f"Pinned tokenizer from {args.tokenizer_json}: vocab_size={tokenizer.vocab_size}")
        else:
            tokenizer = build_tokenizer(h5_files, L=LATTICE_L)
            print(f"Fitted tokenizer: vocab_size={tokenizer.vocab_size} over {len(h5_files)} files")
        train_urls, val_urls_by_temp = write_ising_splits(
            h5_files, tokenizer, args.staging_dir, val_frac=args.val_frac, shard_size=args.shard_size,
            augment_train=args.augment,
        )
        print(f"Wrote train pool ({len(train_urls)} shards) + {len(val_urls_by_temp)} per-temperature val pools")

    train_step = ising_tokenize_step(train_urls, name=args.name)
    val_steps = ising_val_steps_by_temp(val_urls_by_temp)
    executor_main(
        ExecutorMainConfig(prefix=args.prefix),
        steps=[train_step, *val_steps.values()],
        description="marin-spin v1 Ising token cache (train + per-temperature val)",
    )


if __name__ == "__main__":
    main()
