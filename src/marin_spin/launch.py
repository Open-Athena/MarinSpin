# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the v1-grammar Ising model on TPU, reusing the grug/base transformer.

The Ising next-token model is a plain decoder-only transformer, so we reuse ``grug/base`` unchanged
and only supply (a) a small model config and (b) the marin-spin data config built from the v1
tokenize cache. No edge list, no neighbor list (those are the v2/v3 grammars) — just the v1 config
snapshot grammar produced by ``tokenize_ising``.

Two-step run (the cache build is CPU; training is TPU):

    # 1. Build the versioned token cache (CPU). Use the SAME --prefix/bucket as training.
    uv run python -m marin_spin.launch_tokenize \
        --data_dir gs://<bucket>/ising/L16 --staging_dir gs://<bucket>/ising/L16-shards \
        --prefix gs://<bucket>/marin

    # 2. Train on TPU (references the same shard pool -> same cache).
    uv run python -m marin_spin.launch \
        --staging_dir gs://<bucket>/ising/L16-shards --data_dir gs://<bucket>/ising/L16

``launch.py`` does not rewrite shards; it lists the staging pool to reconstruct the identical
tokenize step (same shard URLs -> same executor hash -> same cache path). The training prefix
(``marin_prefix`` on the worker) must match the ``--prefix`` used for the cache build.
"""

import argparse

import fsspec
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.types import this_output_path, versioned

from marin_spin.grug.launch import GrugBaseLaunchConfig, _resolve_run_id, train_grug
from marin_spin.grug.model import GrugModelConfig
from marin_spin.grug.train import GrugEvalConfig, GrugTrainerConfig
from marin_spin.data import make_marin_spin_tagged_data_config
from marin_spin.ising_tokenizer import build_tokenizer
from marin_spin.tokenize_ising import (
    LATTICE_L,
    SEQ_LEN,
    ising_tokenize_step,
    ising_val_steps_by_temp,
    list_ising_split_pools,
    vocab_size_for,
)


def _list_h5(data_dir: str) -> list[str]:
    fs, _ = fsspec.core.url_to_fs(data_dir)
    proto = f"{data_dir.split('://', 1)[0]}://" if "://" in data_dir else ""
    matches = fs.glob(f"{data_dir.rstrip('/')}/*ising_L{LATTICE_L}_T*.h5")
    paths = [p if (not proto or str(p).startswith(proto)) else f"{proto}{p}" for p in matches]
    if not paths:
        raise FileNotFoundError(f"No ising_L{LATTICE_L}_T*.h5 under {data_dir}")
    return sorted(paths)


def build_marin_spin_launch(
    *,
    data_dir: str,
    staging_dir: str,
    run_id: str,
    tpu_type: str,
    steps: int,
    batch_size: int,
    seed: int,
    wandb_entity: str | None,
    wandb_project: str,
    steps_per_eval: int = 1000,
    mp: str = "params=float32,compute=bfloat16,output=bfloat16",
    learning_rate: float = 3e-4,
    min_lr_ratio: float = 0.0,
    vocab_size: int | None = None,
    train_step_name: str = "marin_spin_train",
    hidden_dim: int = 256,
    intermediate_dim: int = 1024,
    num_layers: int = 6,
    num_heads: int = 8,
    num_kv_heads: int = 8,
) -> GrugBaseLaunchConfig:
    """Construct the grug launch config for the v1-grammar Ising run (model + data + knobs).

    ``vocab_size`` (= base tokenizer vocab + 1 for PAD; 339 for the standard L=16, 14-temperature
    dataset) can be passed to skip refitting the tokenizer from ``data_dir`` — useful when the raw
    HDF5 has aged out (the prebuilt cache is permanent, but the tokenizer fit re-reads the data).
    """
    if vocab_size is None:
        vocab_size = vocab_size_for(build_tokenizer(_list_h5(data_dir), L=LATTICE_L))

    train_urls, val_urls_by_temp = list_ising_split_pools(staging_dir)
    # Distinct from the legacy single-pool "marin_spin" cache: the executor keys caches on step name
    # (not train_paths content), so a fresh name forces the held-out split pool to be cached separately.
    train_step = ising_tokenize_step(train_urls, name=train_step_name)
    val_steps = ising_val_steps_by_temp(val_urls_by_temp)
    data = make_marin_spin_tagged_data_config(train_step, val_steps, vocab_size=vocab_size)

    # Decoder-only model; defaults match the original Ising experiment (config.toml: d256/6L/8H/ffn1024).
    # Dims are parameterized for capacity-scaling experiments (e.g. d512/8L/ffn2048).
    model = GrugModelConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=SEQ_LEN,
        head_dim=None,
    )

    return GrugBaseLaunchConfig(
        model=versioned(model),
        data=data,
        output_path=this_output_path(),
        run_id=_resolve_run_id(run_id),
        resources=versioned(ResourceConfig.with_tpu(tpu_type)),
        steps=versioned(steps),
        batch_size=versioned(batch_size),
        seed=versioned(seed),
        mp=versioned(mp),
        tracker=WandbConfig(
            entity=wandb_entity,
            project=wandb_project,
            tags=["grug", "marin-spin", "ising", "v1"],
            group="marin-spin-v1",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=learning_rate,
                weight_decay=0.0,
                lr_schedule="cosine",
                warmup=200,
                min_lr_ratio=min_lr_ratio,
            ),
        ),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            # compute_bpb=False: bits-per-byte needs a byte/text tokenizer; the passthrough integer
            # vocab (physics-event tokens) has no byte mapping (would call encode(".") -> int error).
            GrugEvalConfig(
                eval_batch_size=batch_size,
                steps_per_eval=steps_per_eval,
                max_eval_batches=8,
                eval_ema=False,
                compute_bpb=False,
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v1-grammar Ising model on TPU")
    parser.add_argument("--data_dir", required=True, help="Dir with ising_L16_T*.h5 (for tokenizer/vocab)")
    parser.add_argument("--staging_dir", required=True, help="Shard pool written by launch_tokenize")
    parser.add_argument("--run_id", default="marin-spin-v1")
    parser.add_argument("--tpu_type", default="v5p-8")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps_per_eval", type=int, default=1000, help="Run the per-temperature eval every N steps")
    parser.add_argument(
        "--mp",
        default="params=float32,compute=bfloat16,output=bfloat16",
        help="jmp mixed-precision policy. Use 'params=float32,compute=float32,output=float32' for fp32.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak LR (OG used 3e-4; grug default 3e-3)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0, help="Cosine floor as a fraction of peak LR")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity (org/user) with write access")
    parser.add_argument("--wandb_project", default="marin-spin", help="W&B project")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Trainer vocab size (base tokenizer + 1 PAD; 339 for L=16/14-temps). "
        "Pass to skip refitting the tokenizer from --data_dir (e.g. if the raw HDF5 has aged out).",
    )
    parser.add_argument(
        "--train_name",
        default="marin_spin_train",
        help="Train tokenize step name; must match the one used at cache build (e.g. marin_spin_train_aug).",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model d_model (default 256; baseline)")
    parser.add_argument("--intermediate_dim", type=int, default=1024, help="FFN hidden dim (default 1024)")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer layers (default 6)")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads (default 8)")
    parser.add_argument("--num_kv_heads", type=int, default=8, help="KV heads (default 8)")
    args = parser.parse_args()

    launch = build_marin_spin_launch(
        data_dir=args.data_dir,
        staging_dir=args.staging_dir,
        run_id=args.run_id,
        tpu_type=args.tpu_type,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        steps_per_eval=args.steps_per_eval,
        mp=args.mp,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        vocab_size=args.vocab_size,
        train_step_name=args.train_name,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
    )
    train_grug(name="grug/marin-spin-v1", launch=launch)


if __name__ == "__main__":
    main()
