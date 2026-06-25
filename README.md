# MarinSpin

**Can a transformer learn 2D-Ising kinetic Monte-Carlo coarsening dynamics — flip rates and
waiting times — from trajectory data alone?** Built on [Marin](https://marin.community).

Split out of the `marin-community/marin` monorepo (from `experiments/grug/marin_spin/`) to consume
**Marin as a library**. That migration is **in progress** — see [`ISSUES.md`](ISSUES.md) for the
remaining blockers to hand to the marin-as-library team (notably: the `epoch_reshuffle` levanter
feature and the `grug` model scaffolding both still need a library home).

## Layout

```
src/marin_spin/        importable package (the Ising-specific work)
  tokenize_ising.py    trajectory -> token grammar pipeline
  ising_tokenizer.py   IsingTokenizer (vocab, dt bins, encode/decode)
  data.py              data config (sets epoch_reshuffle=True  <- ISSUES #1)
  launch.py            training entrypoint
  launch_tokenize.py   tokenization entrypoint
  rollout_quench.py    KV-cached autoregressive rollout + quench eval
  kv_decode.py         KV-cache prefill/decode_step
  compare_bkl.py       per-step model vs exact KMC (BKL) comparison
  probs_vs_bkl.py      flip-probability / KL diagnostics
  corr_anisotropy.py   directional correlation length
  augment.py, calibrate_grug.py, dist_vs_bkl.py,
  raster_anisotropy.py, raster_edge_error.py, rate_by_locality.py
  grug/                VENDORED grug model/trainer scaffolding (delete once it's a
                       levanter.grug library — see ISSUES #2/#3)
analysis/              plotting / eval scripts (poster figures); run from this dir
poster/                conference poster (poster.tex + figures/, 36x48")
slides/                talk decks pulled from Open-Athena/ising-ntp (motivation_slides + talk)
docs/                  levanter-epoch-reshuffle.patch (the un-upstreamed feature)
.claude/skills/        babysit-job, debug, scan-logs, commit, ... (paths need adapting, ISSUES #7)
```

## Status

This is a faithful **code** migration with imports rewritten
(`experiments.grug.marin_spin` → `marin_spin`). It will **not run end-to-end yet** until the
`ISSUES.md` blockers are resolved (the un-upstreamed levanter feature and the grug/pretraining
monorepo dependencies). Runtime data and checkpoints were intentionally not migrated.

## Install (once the library gaps in ISSUES.md are closed)

```bash
uv sync          # pulls marin-{levanter,haliax,core,fray} per pyproject.toml
```
