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

3D on `plev8`: `ua`, `va`, `hus`, `zg`
2D: `tas`, `huss`, `psl`, `pr`

Notably **absent from core**:

- `ta` (air temperature on `plev`) — Pangeo daily coverage is essentially
  nil (3 models). Instead we derive 7 layer-mean temperatures from `zg` +
  `hus` via the hypsometric equation at ingest time and store them as
  `ta_derived_layer_{0..6}`. This is a proxy, not a true temperature; it's
  treated as derived throughout and labelled as such in the zarr.
- `ps` (surface pressure) — not published at daily cadence by any CMIP6
  model. `psl` (mean sea-level pressure) + a topography/`zg`-derived
  surface mask stand in for surface pressure when needed.

See **Derived variables** below for details.

**Optional (include per-model when published):**

- TOA radiation: `rsdt`, `rsut`, `rlut`
- Surface radiation: `rsds`, `rsus`, `rlds`, `rlus`
- Surface turbulent fluxes: `hfss`, `hfls`
- Surface wind: `sfcWind`, `uas`, `vas`

### Forcings (atmosphere-only lower boundary)

Pulled from monthly tables and **interpolated to daily** at ingest
(ocean thermal inertia and sea-ice variability are slow enough that
monthly → daily is acceptable; true daily availability is essentially
nil for these fields in Pangeo).

- `ts` from `Amon` — surface temperature (SST over ocean, ice-top temp
  over sea ice, skin temp over land). The correct atmosphere-only
  lower-boundary quantity. 64 models publish it.
- `siconc` from `SImon` — sea-ice fraction on the ocean grid;
  regridded to the F22.5 target. 57 models.

### Static per-model fields

From the CMIP6 `fx` table; broadcast along time by the data loader.

- `sftlf` — land fraction (land-sea mask). 51 models.
- `orog` — surface altitude (orography). 47 models.

### Expected model coverage

From the inventory run: **~21 source_ids** satisfy all of (full `day`
core coverage) + `ts` + `siconc` + `sftlf` + `orog` across at least one
member; **~25** if `orog` is treated as optional. The multi-member cap
(Issue 3) retains up to 3 realizations per `(source_id, experiment, p,
f)` label, yielding the effective pilot dataset.

### Derived variables

**Layer-mean temperature `ta_derived_layer_{0..6}`.** Computed at ingest
from `zg` and `hus` using the hypsometric equation under hydrostatic
balance — an excellent approximation for daily means at 4°×4°:

```
T_v^{layer_i}  = g · (z_{i+1} - z_i) / (R_d · ln(p_i / p_{i+1}))
q^{layer_i}    = (q_i + q_{i+1}) / 2
T^{layer_i}    = T_v^{layer_i} / (1 + 0.608 · q^{layer_i})
```

with `R_d = 287.05 J/kg/K`, `g = 9.80665 m/s²`. All inputs are co-located
on `plev8` levels, so the computation is just differences and averages
along the `plev` dimension — no interpolation. The 7 layer values live at
the log-pressure midpoints `√(p_i · p_{i+1})`. Expected error from
applying an instantaneous relation to daily means is <<1 K at these
scales; far below inter-model spread.

Variables are named `ta_derived_layer_*` throughout (never `ta`) to keep
it unambiguous that these are proxies, not native CMIP6 temperatures.

### Vertical & horizontal grid

- **Vertical**: `plev8` = {1000, 850, 700, 500, 250, 100, 50, 10} hPa.
  Pressure-level vertical departs from the project's hybrid-sigma baseline
  convention; pressure levels are used directly.
- **Horizontal**: Gauss-Legendre `F22.5` — a full rectangular
  `(nlat=45, nlon=90)` lat-lon grid where latitudes are the
  Gauss-Legendre quadrature nodes (not equispaced) and longitudes are
  equispaced at 4°. Chosen so spherical-harmonic transforms on the
  processed data are exact; matches the project's existing
  `gaussian_grid_45_by_90` convention. Grid built by
  `grid.make_target_grid("F22.5")` (see `grid.py`).
- **Regridding**: `xesmf` targeting `F22.5`. **Conservative** for fluxes
  and precip (preserves spatial integrals across coarsening);
  **bilinear** for state fields (pressure-level state doesn't close a
  mass-weighted budget, so bilinear's smoothness is preferable).
  Weights cached per source grid.

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
2. Open each variable's zarr (state, forcings from `Amon`/`SImon`,
   static from `fx`).
3. Validate `cell_methods` / vertical / grid.
4. Regrid horizontally to F22.5 (Gauss-Legendre 45 x 90) via `xesmf`
   (bilinear for state, conservative for fluxes).
5. Apply below-surface nearest-above fill + emit time-varying
   `below_surface_mask(time, plev, lat, lon)` (uint8, single shared
   field).
6. Compute derived `ta_derived_layer_{0..6}` from `zg` + `hus`.
7. Linear-interpolate monthly forcings (`ts`, `siconc`) onto the daily
   time axis; broadcast static fields along time (or leave as
   time-invariant coords, TBD with Issue 9).
8. Time-subset per config.
9. Write zarr via fsspec (local or gs://).
10. Record metadata to `index.parquet`.

Per-dataset jobs are embarrassingly parallel. Pilot runs locally
(single-process dask, no argo) for debuggability; a batch/argo wrapper
comes later if needed.

Normalization stats (per-model, NaN-aware) are a separate later step,
computed from the produced zarrs.

## Resolved Issues

- **Issue 1 — Config design.** Single YAML per script, loaded via
  `dacite` into dataclasses in `config.py`. Top-level `defaults:` +
  `selection:` + sparse `overrides:` list. See `ProcessConfig` and
  `InventoryConfig`.
- **Issue 3 — Member caps.** `require_i = 1` and `max_members_per_f = 3`
  at ingest, applied per `(source_id, experiment, p, f)` label.
  Deterministic selection by `(variant_f, variant_r)`.
- **Issue 4 — Time subset.** Pilot: `historical` 2010, `ssp585` 2015
  (one full year each). Set `defaults.time_subset: null` for full range.
- **Issue 5 — Pangeo-only vs ESGF.** Pangeo-only. `ta` replaced by
  derived layer-T; `ps` replaced by `psl` + topography mask. Confirmed
  via full inventory pass.
- **Issue 8 — Regridding.** `xesmf` targeting Gauss-Legendre F22.5
  (45 x 90). Conservative for fluxes/precip, bilinear for state.
  Weights cached per source grid.
- **Forcings wiring.** Monthly `ts` (`Amon`) + `siconc` (`SImon`)
  interpolated to daily; static `sftlf` + `orog` (`fx`) broadcast.
  ~21 source_ids have full core + forcing + static coverage in Pangeo.
- **Issue 7 — Below-surface masking.** Single shared time-varying
  `below_surface_mask(time, plev, lat, lon)` (uint8) per dataset.
  Primary derivation: NaN union across 3D plev variables (captures
  day-to-day surface-pressure variation via the model's own masking
  decisions). Fallback: `zg < orog` (still time-varying via `zg`).
  Drop dataset if both unavailable. Masked cells in 3D variables get
  nearest-above-in-the-vertical fill — each column's below-surface
  levels inherit the lowest above-surface level's value, handling any
  number of consecutive masked bottom levels. `mask_source` recorded
  in `index.parquet`.

## Open Issues (to work through sequentially)

### Issue 6 — Chunking strategy (next)

Training reads short windows (2–4 timesteps at a time). Chunking must be
small enough to avoid reading huge unused blocks per window, large enough
to avoid GCS per-object overhead. At F22.5 (45×90) with `plev8`, one
timestep of a 3D field is ~130 KB; chunking `time=32` gives ~4 MB chunks,
likely a reasonable balance. Need to confirm the target chunk size and
whether to differ between 2D and 3D variables.

### Issue 2 — Label schema

Working proposal: label = `(source_id, p)` (strict reading). Realization
`r` and initialization `i` are ensemble variation; forcing `f` is an
external input, not physics. Confirm that the stored label column is
just these two pieces, and whether we also record `f` for downstream
analysis even if it's not part of the label.

### Issue 9 — `index.parquet` schema

What to record per dataset: `source_id`, `experiment`, `variant_label`,
parsed `r`/`i`/`p`/`f`, label columns from Issue 2, native calendar,
native horizontal grid, `grid_label` used, regrid method, time range,
variables present, data-quality flags. Decided together with the
processing script.

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
