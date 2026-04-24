# CMIP6 Daily Pilot Dataset

Pilot project to download and stage a multi-model CMIP6 daily-mean dataset
for an ACE-style emulator with a per-model label embedding. Scope is
deliberately narrow: daily cadence only, CMIP6 only, Pangeo catalog only
(unless revisited), no training runs, drastically time-subset at first with
config knobs to scale up to the full roster later.

## Goals

- **Maximize data availability.** Default: pull every available member for
  every included model for both `historical` and `ssp585`. Holdout and
  per-model member caps are future concerns, not ingest-time decisions.
- Support ~30–50 CMIP6 models to train a robust source-model label
  embedding that can be fine-tuned on unseen models.
- Heterogeneous variable sets across models are expected; ingest records
  what each model has. Downstream training support for heterogeneous
  variables is a separate future code change.
- Configurable via YAML loaded into a dataclass with `dacite`; defaults
  cover most datasets, per-dataset overrides handle exceptions.

## Dataset Specification

### Experiments & members

- `historical` (1850–2014) and `ssp585` (2015–2100).
- **All available members** by default. Whether a per-model cap is applied
  is a future decision.
- Experiment is recorded as metadata and may be used in experiments, but
  **not** encoded in the label.

### Variables (CMIP6 `day` table, CF names kept as-is)

**Core state (required — any model missing any of these is dropped):**

`ta`, `ua`, `va`, `hus`, `zg` (3D on `plev8`); `tas`, `huss`, `ps`, `psl`,
`pr` (2D).

**Optional (include per-model when published):**

- TOA radiation: `rsdt`, `rsut`, `rlut`
- Surface radiation: `rsds`, `rsus`, `rlds`, `rlus`
- Surface turbulent fluxes: `hfss`, `hfls`
- Surface wind: `sfcWind`, `uas`, `vas`

### Vertical & horizontal grid

- **Vertical**: `plev8` = {1000, 850, 700, 500, 250, 100, 50, 10} hPa.
  Pressure-level vertical departs from the project's hybrid-sigma baseline
  convention; pressure levels are used directly.
- **Horizontal**: 4°×4° regular lat-lon (pilot resolution chosen to keep
  dataset size small). Prefer `grid_label=gr` if already on lat-lon; else
  regrid from `gn` with `xesmf` — **conservative** for fluxes and precip,
  **bilinear** for state fields (pressure-level state doesn't close a
  mass-weighted budget, so bilinear's smoothness is preferable).

### Temporal

- Daily means (`cell_methods: time: mean`). Validated per file at ingest.
- Each model keeps its **native calendar** (`noleap`, `360_day`,
  `gregorian`, `proleptic_gregorian`); recorded as metadata per dataset.
  Uniform Δt = 86400 s within each dataset, so no calendar harmonization
  needed for timestep uniformity.

### Labels

- One label per dataset corresponding to a **physics configuration**.
  Working definition: `(source_id, physics_index p)` from the CMIP6 variant
  label, since different `p` indices within one `source_id` denote distinct
  physics configurations. Realization `r` and initialization `i` are
  ensemble variation and excluded. Forcing `f` is external boundary
  conditions, not physics, and is excluded. See **Issue 2**.

### Storage layout

```
gs://<bucket>/cmip6-daily-pilot/
  v0/
    <source_id>/<experiment>/<variant_label>/
      data.zarr
    stats/
      <source_id>/{mean,std}.nc      # per-model for now; see Issue 7
    index.parquet                    # one row per dataset
```

One zarr per `(source_id, experiment, variant_label)`. Chunked with
short-window training in mind; see **Issue 6**.

## Extraction Approach (high level)

Split into two scripts:

### `inventory.py` — dataset discovery & metadata

Queries the Pangeo GCS CMIP6 intake-esm catalog for our variable list ×
experiments and emits a tidy table (parquet/csv) with one row per
`(source_id, experiment, variant_label, variable)` and enough columns to
answer cross-model comparison questions: variables present, grid_label,
native calendar, time range, horizontal/vertical grid info, member counts
per model. Used both to measure Pangeo coverage (Issue 5) and as input to
the processing script's planning step. No data movement — metadata only.

### `process.py` — per-dataset processing

Driven by the YAML config + the inventory. For each selected
`(source_id, experiment, variant_label)`:

1. Drop if any core variable missing.
2. Open each variable's zarr.
3. Validate `cell_methods` / vertical / grid.
4. Regrid horizontally to the 4°×4° target.
5. Apply below-surface persistence fill + emit mask channel.
6. Time-subset per config.
7. Write zarr via fsspec (local or gs://).
8. Record metadata to `index.parquet`.

Per-dataset jobs are embarrassingly parallel. Pilot runs locally
(single-process dask, no argo) for debuggability; a batch/argo wrapper
comes later if needed.

Normalization stats (per-model, NaN-aware) are a separate later step,
computed from the produced zarrs.

## Open Issues (to work through sequentially)

### Issue 1 — YAML/dacite config design

Shape of the config file. Proposed: a single YAML loaded into a dataclass
via `dacite`, with a top-level `defaults:` section plus a `datasets:` list
of per-entry overrides. Per-entry entries can be sparse — most datasets
inherit everything from defaults, exceptional ones set fields explicitly.
Open: exact dataclass schema, override semantics, how time subsetting knobs
are expressed (per-dataset vs global).

### Issue 2 — Label schema

Working proposal: label = `(source_id, p)`. Need to confirm this matches
the user's intent given "physics configuration (model and hyperparameters),
not ensemble member". Alternatives: `source_id` alone (simpler, conflates
p-variants within a model), or `(source_id, p, f)` (includes forcing
variants as separate labels).

### Issue 3 — Member inclusion & caps

Base assumption: ingest all members. Some models publish dozens of
historical members; others publish one. Decision: do we cap per model at
ingest time (simpler pipeline, loses data) or ingest everything and cap at
training time (more flexible, bigger dataset)? Related: whether member
filtering is expressed in the YAML config.

### Issue 4 — Time subsetting for pilot

"Drastically subset" requires a concrete rule. Options: N years per
experiment per model (e.g. 2 years of historical + 2 years of ssp585),
single year, or a specific decade. Must be a config knob that can be set to
"full" to produce the full dataset without code changes.

### Issue 5 — Pangeo-only vs ESGF

How much model coverage do we lose by excluding ESGF? Pangeo's daily-table
coverage is selective — plausibly 40–70% of CMIP6 daily publishers.
Recommendation for pilot: **exclude ESGF** to keep the pipeline simple
(zarr in, zarr out; no netCDF download-and-convert; no DRS bookkeeping);
revisit if the Pangeo roster is too thin. Need to confirm, and decide how
we measure "too thin" — an inventory pass against Pangeo is cheap and
would give us the number.

### Issue 6 — Chunking strategy

Training reads short windows (2–4 timesteps at a time). Chunking must be
small enough to avoid reading huge unused blocks per window, large enough
to avoid GCS per-object overhead. At 4°×4° with `plev8`, one timestep of a
3D field is ~130 KB; chunking `time=32` gives ~4 MB chunks, likely a
reasonable balance. Need to confirm the target chunk size and whether to
differ between 2D and 3D variables.

### Issue 7 — Below-surface masking (confirm)

Proposed: **persistence fill** from the lowest valid level downward, plus
a per-variable **mask channel** so the model sees where fills were
applied. Applies mainly to the lowest `plev8` level (1000 hPa) over high
topography and sometimes 850 hPa. Confirm and decide whether the mask
channel is per-variable or a single shared surface-topography mask.

### Issue 8 — Regridding methods (confirm)

Proposed: conservative for fluxes and `pr`; bilinear for state. Cache
regrid weights per source grid. Confirm, and decide whether any variable
needs nearest-neighbor (e.g., categorical fields — unlikely in our list).

### Issue 9 — `index.parquet` schema

What to record per dataset: `source_id`, `experiment`, `variant_label`,
native calendar, native horizontal grid, `grid_label` used, regrid method,
time range, variables present, flags (e.g., `cell_methods` validation
result, data-quality issues, ESGF vs Pangeo source). Need a concrete
schema.

## Deferred / Future Issues

- **Heterogeneous variables at train time.** Core code change in
  `fme/core/dataset/` and the stepper to allow per-dataset variable
  subsets. Tracked separately from this pilot.
- **Calendar heterogeneity across datasets.** Each dataset records its
  native calendar now; cross-dataset handling (does `ConcatDatasetConfig`
  tolerate mixed calendars?) is a later concern.
- **Train / val / holdout splits.** By time, by model, by experiment, by
  member — all reasonable. Decided post-ingest.
- **Normalization pooling strategy.** Per-model stats now. Whether training
  uses pooled, per-model, or hierarchical normalization is a later
  decision and may depend on what works for the embedding.
- **Per-model member caps.** See Issue 3 — if we defer capping to training
  time, this is future work.
- **Sub-daily or monthly cadence.** Separate datasets, separate pilots.
- **Non-uniform timestep support in the data loader.** Not needed at
  daily cadence but relevant for monthly later.
