# Context: 2026-06-26 FM experiments

Glossary for this experiment directory. Definitions only — no procedures, no
numbers, no file inventory.

## Naming a model

- **Architecture tag** (`nc-sfno`, `nc-swin-v2`, `nc-swin-v2.1`) — names a
  module builder family together with the architectural knobs held fixed for
  it. A new tag is minted when the architecture changes in a way that makes
  checkpoints non-interchangeable.
- **Config version** (the `-v1` / `-v2` / `-v3` suffix) — a revision of the
  training recipe or dataset for a given architecture. Orthogonal to the
  architecture tag: the `v2` in `nc-swin-v2` is part of the tag, not a config
  version, and the version counter restarts at `v1` for each new tag.

## Naming an experiment

- **Regime** (`c96`, `era5`, `fm`) — which training data mix a run uses.
- **Arm** (A1, A2, A3) — how normalization statistics are grouped across data
  sources. A1 is the shared/pooled control.
- **Conditional** — whether the dataset labels, besides selecting statistics,
  also drive the module's adaLN/CLN conditioning.
- **Masking** (e.g. `mask10`) — a training-only variant that applies synthetic
  dropout to input channels.
- **Cell** — one point of architecture x regime x arm x conditional x masking.
  Each cell is one training run.

## Naming a file or a run

- **Base config** — hand-written, lives in `base_configs/`. Either the source a
  generator composes from, or a run in its own right.
- **Run config** — generated, lives in `run_configs/`. Never hand-edited;
  regenerate instead.
- **Job name** — the wandb/beaker run name, derived from a config filename by
  dropping the config prefix and the dataset tag and prepending `ace2-fm-`.

## Comparing runs

- **Baseline** — the A1 cell of the previous architecture tag in the same
  regime. A new architecture tag is judged against it.
