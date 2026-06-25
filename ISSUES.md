# MarinSpin → "Marin as a library": migration issues & handoff

This repo was split out of the `marin-community/marin` monorepo (commit
`73bee3a9`, from `experiments/grug/marin_spin/` + related `scratch/` scripts). The goal is for
MarinSpin to depend on **Marin as a library** rather than living inside the monorepo.

This document enumerates everything that blocks or complicates that, for the *marin-as-library*
team. Each item lists **what**, **why it blocks library use**, and a **recommendation**.

---

## Dependency surface (what MarinSpin actually imports)

Library (want these to be pip-installable from Marin):
- `levanter.*` — `levanter.grug.{sharding,loss,attention}`, `levanter.data.*`, `levanter.optim`,
  `levanter.trainer`, `levanter.checkpoint`, `levanter.tracker.wandb`, `levanter.eval`,
  `levanter.callbacks.*`, `levanter.analysis.backward_flow`, `levanter.models.lm_model`, …
- `haliax.partitioning`, `haliax.jax_utils`
- `marin.execution.{executor,types}`, `marin.processing.tokenize`, `marin.training.training`
- `fray.cluster`, `fray.current_client`, `fray.types`

Not library — experiment/monorepo code MarinSpin is built on (the hard part):
- `experiments.grug.base` (model / train / launch)  → **vendored** into `src/marin_spin/grug/`
- `experiments.grug.checkpointing`, `experiments.grug.dispatch` → **vendored**
- `experiments.defaults`, `experiments.pretraining_datasets` → **still unresolved** (see #4)

---

## Blockers

### 1. `levanter` `epoch_reshuffle` feature is not upstreamed  ⛔ hard blocker
- **What:** `src/marin_spin/data.py` does `dataclasses.replace(config, …, epoch_reshuffle=True)`.
  That flag, plus `AsyncDataset.reshuffle_each_epoch()` and the `EpochReshufflingDataset` class,
  exist only as a **local (staged) diff** to `lib/levanter` in the monorepo working tree — they are
  not in any released/`main` levanter.
- **Why it blocks:** installing `marin-levanter` from `main` will not have the flag; the config
  construction raises immediately.
- **Recommendation:** upstream the feature to `lib/levanter` (it is self-contained:
  `EpochReshufflingDataset` + `LmDataConfig.epoch_reshuffle` + a unit test). The full diff is
  preserved at `docs/levanter-epoch-reshuffle.patch` in this repo. Until merged, pin
  `marin-levanter` to a branch that contains it (see `pyproject.toml`).

### 2. The "grug" model/trainer scaffolding lives in `experiments/`, not in a library  ⛔ hard blocker
- **What:** MarinSpin trains a *grug* transformer via `experiments.grug.base.{model,train,launch}`
  (`GrugModelConfig`, `Transformer`, `GrugTrainerConfig`, `GrugEvalConfig`, `train_grug`,
  `GrugBaseLaunchConfig`). These are experiment files, not packaged code. They are **vendored** here
  under `src/marin_spin/grug/` as a stopgap.
- **Why it blocks:** a library consumer cannot `import` from `experiments/`. Vendoring forks shared
  code (grug is also used by `experiments/grug/{moe,modular_opt}`).
- **Recommendation:** promote the grug base framework into the **`levanter.grug`** subpackage
  (which already exists and already holds `sharding`/`loss`/`attention`). Then MarinSpin imports
  `levanter.grug.model` etc. and the vendored copy here is deleted.

### 3. `experiments.grug.{checkpointing,dispatch}` are experiment code  ⚠️ vendored
- **What:** `checkpointing.py` (only needs `levanter.checkpoint`) and `dispatch.py` (only needs
  `fray.*`). Small and self-contained; **vendored** under `src/marin_spin/grug/`.
- **Recommendation:** fold into the same `levanter.grug` (or `fray`) library home as #2.

### 4. `experiments.defaults` + `experiments.pretraining_datasets` pulled in by grug launch  ⛔
- **What:** the vendored `src/marin_spin/grug/launch.py` still imports
  `from experiments.defaults import _submit_train_job, default_validation_sets` and
  `from experiments.pretraining_datasets import nemotron_mix`. These are general Marin pretraining
  infrastructure (default eval sets, the Nemotron data mixture) — not relevant to Ising training.
- **Why it blocks:** these imports fail outside the monorepo; left in place deliberately so the
  team can see them. MarinSpin almost certainly does not need `nemotron_mix`.
- **Recommendation:** either (a) give `_submit_train_job` / `default_validation_sets` a library
  home (`marin.training`?), and make the dataset mixture a parameter rather than a module-level
  import; or (b) prune them from the grug launch path so it doesn't hard-import pretraining infra.

### 5. Marin packages are published, but only as a nightly dev channel  ✅/ℹ️
- **Resolved:** `marin-core`, `marin-levanter`, `marin-haliax`, `marin-fray` **are on PyPI**
  (and GitHub Packages, github.com/orgs/marin-community/packages). Latest at migration time:
  `0.2.27.dev202606250842`. `pyproject.toml` now pins these from PyPI (no git pins).
- **Remaining nuance:** they are **dev/nightly** builds (no stable semver), so downstream pins are
  to dated dev versions; `[tool.uv] prerelease = "allow"` is required. The `tpu`/`gpu` JAX extras
  story (`marin-levanter[tpu]`) should be documented for library consumers.
- **Recommendation:** publish periodic **stable** releases and document the accelerator-extras
  install path.

### 6. `levanter.grug` is already a partial library — good, finish it  ✅/⚠️
- `levanter.grug.{sharding,loss,attention,_moe,grug_moe}` already ship in the levanter package.
  The remaining grug pieces (#2/#3) should join them so "grug" is wholly a library.

### 7. Skills reference monorepo paths  ⚠️
- The copied skills (`.claude/skills/`, e.g. `babysit-job`) resolve cluster configs from
  `lib/iris/config/*.yaml` and call `./infra/pre-commit.py` — paths that don't exist here.
- **Recommendation:** adapt the skills' cluster-config resolution and lint entrypoint for a
  library consumer, or have Marin expose those as a tool/CLI.

### 8. Runtime data & checkpoints are not in the repo  ℹ️
- Scripts hardcode paths like `scratch/ckpt/step-49400`, `scratch/data/*.npz`,
  `/Users/.../Downloads/*.h5` (KMC trajectory HDF5). These are large artifacts, intentionally not
  migrated. Re-point them (config/env) when wiring up runs.
- **TODO:** host the KMC trajectory dataset (and possibly the trained checkpoint) on
  **Hugging Face** (a dataset/model repo) and link to it from this repo's README, so runs are
  reproducible without local files.

---

## Vendored code inventory (delete once upstreamed)

| Path in this repo | Came from (monorepo) | Resolve via |
|---|---|---|
| `src/marin_spin/grug/model.py`        | `experiments/grug/base/model.py`     | #2 → `levanter.grug` |
| `src/marin_spin/grug/train.py`        | `experiments/grug/base/train.py`     | #2 → `levanter.grug` |
| `src/marin_spin/grug/launch.py`       | `experiments/grug/base/launch.py`    | #2, #4 |
| `src/marin_spin/grug/checkpointing.py`| `experiments/grug/checkpointing.py`  | #3 |
| `src/marin_spin/grug/dispatch.py`     | `experiments/grug/dispatch.py`       | #3 |

## Suggested end state
1. Upstream `epoch_reshuffle` to levanter (#1).
2. Move grug base/checkpointing/dispatch into `levanter.grug` (#2, #3); delete `src/marin_spin/grug/`.
3. Parameterize the data mixture / eval sets so grug launch doesn't import pretraining infra (#4).
4. Provide an installable Marin distribution with accelerator extras (#5).
5. MarinSpin then depends purely on `marin-levanter`, `marin-haliax`, `marin-core`, `marin-fray`.
