# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the ``LmDataConfig`` that grug/base consumes for marin-spin v1."""

import dataclasses
import os

from levanter.data.text import LmDataConfig
from marin.execution.types import ExecutorStep
from marin.processing.tokenize import add_validation_sets_to_mixture, lm_data_config


def make_marin_spin_data_config(
    tokenize_step: ExecutorStep,
    *,
    num_validation_sequences: int,
    vocab_size: int,
) -> LmDataConfig:
    """Turn an Ising tokenize step into an ``LmDataConfig`` for grug/base.

    The train/val split is delegated to Levanter: ``num_validation_sequences`` windows are carved
    from the single shard pool as validation. ``vocab_size`` is the trainer-facing vocab
    (``vocab_size_for(tokenizer)`` = base tokenizer vocab + 1 for PAD); it must be set explicitly
    because the mixture helpers do not, and the ``passthrough`` tokenizer reads it at train time.

    The consumer MUST set ``model.max_seq_len == SEQ_LEN`` and ``model.vocab_size == vocab_size`` so
    each flat-cache slice is exactly one padded window.
    """
    component_name = os.path.basename(tokenize_step.name)
    config = lm_data_config(
        tokenize_step,
        num_validation_sequences={component_name: num_validation_sequences},
    )
    return dataclasses.replace(config, vocab_size=vocab_size)


def make_marin_spin_tagged_data_config(
    train_step: ExecutorStep,
    val_steps_by_temp: dict[str, ExecutorStep],
    *,
    vocab_size: int,
) -> LmDataConfig:
    """Data config with one tagged validation set per temperature.

    Training is the single all-temperature pool (weight 1.0); each per-temperature validation step is
    added as a weight-0 component tagged with its temperature, so grug's tagged evaluator emits
    ``eval/<T>/loss`` (a W&B panel per temperature) plus the ``eval/macro_loss`` average. ``vocab_size``
    is set explicitly for the passthrough tokenizer (the mixture helpers omit it).
    """
    config = lm_data_config(train_step)
    config = add_validation_sets_to_mixture(config, val_steps_by_temp)
    # Reshuffle the training pool every epoch (matches the original PyTorch DataLoader(shuffle=True));
    # important here because we train many epochs over a small fixed window pool.
    return dataclasses.replace(config, vocab_size=vocab_size, epoch_reshuffle=True)
