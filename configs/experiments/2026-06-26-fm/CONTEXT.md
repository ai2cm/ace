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

## Paper-replication experiments

- **Climate** — one of the 1x/2x/3x/4xCO2 slab-ocean equilibrium states of the
  SHiELD-SOM ensemble, each with its own ensemble members.
- **Equilibrium run** — a slab-ocean free run forced by, and initialized from,
  a member of one climate.
- **Abrupt run** — a slab-ocean run initialized from a 1xCO2 state with the CO2
  forcing set to another climate's value.
- **Data-only evaluation** — a reference member evaluated against itself,
  giving the reference climate's diagnostics with no model skill involved.
- **Kind** — one config family of `generate_paper_configs.py`, named
  `{data}-{experiment}-{co2}-{shape}-{ocean}-{mode}`, six slots always
  present, no token containing `-` (e.g. `som-abrupt-4xCO2-ens-sstslab-eval`);
  see the plan's "Kind naming".
- **Ocean mode** — where SST comes from at inference: `sstslab` (mixed-layer
  ocean driven by the model's fluxes, SOM data only), `sstprescribed` (read
  from the data store as in training), `sstdata` (no model ran; data-only).
  "SST held at control while CO2 steps" is spelled by the data slot (`som`,
  a control store) plus the co2 slot (`4xCO2`), not by the ocean slot.
- **Forcing grid** — `shield` (SOM, AMIP, ramped stores) or `era5`; a kind runs
  on the regimes that trained on its grid (c96 → shield, era5 → era5, fm →
  both).
- **Prescribed-SST evaluation** — an evaluator run with the training-time
  ocean (SST and sea ice read from the reference at every step) scored against
  that reference: the SOM member (`som-eq-dataCO2-10yr-sstprescribed-eval`), an AMIP or AMIP +2 K / +4 K
  run, or a ramped-SST random-CO2 run. The paper's AMIP inference plus
  data-only evaluator, in one job.
- **Held-out member** — an ensemble member the norm-ablation cells did not
  train on: AMIP `ic_0002`, ramped `ic_0003`, SOM `ic_0005` (3xCO2 `ic_0002`).
  The hand-written `fm-random-v1/v3` and `fm-0.x-v1` runs trained on the first
  two as well.
