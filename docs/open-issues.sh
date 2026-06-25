#!/usr/bin/env bash
# Open the marin-as-library handoff issues on Open-Athena/MarinSpin.
# Prereq:  brew install gh && gh auth login
# Run:     bash docs/open-issues.sh
set -euo pipefail
REPO="Open-Athena/MarinSpin"

gh label create agent-generated --repo "$REPO" --color BFD4F2 \
  --description "Created by an AI agent" 2>/dev/null || true
gh label create blocker --repo "$REPO" --color B60205 \
  --description "Blocks consuming Marin as a library" 2>/dev/null || true

mk() { # title  labels  body
  gh issue create --repo "$REPO" --title "$1" --label "$2" --body "$3"
}

mk "Upstream levanter \`epoch_reshuffle\` feature (BLOCKER)" "agent-generated,blocker" \
'🤖 `src/marin_spin/data.py` sets `epoch_reshuffle=True`, a flag (+ `AsyncDataset.reshuffle_each_epoch` and `EpochReshufflingDataset`) that only exists as a local staged diff to `lib/levanter` — not in any released levanter. Installing `marin-levanter` from a release will raise on config construction.

Fix: upstream the (self-contained) feature to `lib/levanter`; until then pin `marin-levanter` to a branch that has it. Patch preserved at `docs/levanter-epoch-reshuffle.patch`. See ISSUES.md #1.'

mk "Promote grug model/trainer scaffolding into \`levanter.grug\` (BLOCKER)" "agent-generated,blocker" \
'🤖 MarinSpin trains a grug transformer via `experiments.grug.base.{model,train,launch}` (`GrugModelConfig`, `Transformer`, `GrugTrainerConfig`, `train_grug`, ...) — experiment code, not a library. Vendored as a stopgap under `src/marin_spin/grug/`.

`levanter.grug` already ships `sharding`/`loss`/`attention`; the base model/train/launch should join it so consumers `import levanter.grug...` and the vendored copy is deleted. See ISSUES.md #2.'

mk "Fold grug checkpointing/dispatch into a library" "agent-generated" \
'🤖 `experiments/grug/checkpointing.py` (needs only `levanter.checkpoint`) and `experiments/grug/dispatch.py` (needs only `fray.*`) are vendored under `src/marin_spin/grug/`. Small and self-contained — give them a library home alongside the grug base (levanter.grug / fray). See ISSUES.md #3.'

mk "grug launch hard-imports pretraining infra (\`experiments.defaults\`, \`pretraining_datasets\`) (BLOCKER)" "agent-generated,blocker" \
'🤖 The vendored `src/marin_spin/grug/launch.py` still imports `from experiments.defaults import _submit_train_job, default_validation_sets` and `from experiments.pretraining_datasets import nemotron_mix`. These are general Marin pretraining infra (not relevant to Ising) and fail outside the monorepo.

Fix: give `_submit_train_job`/`default_validation_sets` a library home and make the data mixture a parameter rather than a module-level import. See ISSUES.md #4.'

mk "Publish stable Marin releases (only nightly dev channel today)" "agent-generated" \
'🤖 `marin-core`, `marin-levanter`, `marin-haliax`, `marin-fray` ARE published (PyPI + GitHub Packages) but only as nightly dev builds (e.g. `0.2.27.dev202606250842`) with no stable semver, so downstream must pin dated dev versions and set `prerelease = "allow"`. Provide periodic stable releases and document the `tpu`/`gpu` accelerator-extras install path. See ISSUES.md #5.'

mk "Adapt copied .claude skills to non-monorepo paths" "agent-generated" \
'🤖 Copied skills (`.claude/skills/babysit-job`, ...) resolve cluster configs from `lib/iris/config/*.yaml` and call `./infra/pre-commit.py` — paths that do not exist here. Adapt the cluster-config resolution and lint entrypoint for a library consumer, or expose them via a Marin CLI/tool. See ISSUES.md #7.'

mk "Externalize runtime data & checkpoint paths" "agent-generated" \
'🤖 Scripts hardcode `scratch/ckpt/step-49400`, `scratch/data/*.npz`, and `~/Downloads/*.h5` (KMC trajectory HDF5). These artifacts were intentionally not migrated. Re-point them via config/env when wiring up runs. See ISSUES.md #8.'

echo "Done — opened the handoff issues on $REPO."
