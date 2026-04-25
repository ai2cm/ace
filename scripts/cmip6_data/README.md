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

**Vertical-grid note (important for downstream masking).** The derived
temperatures live on **layers between adjacent plev levels**, so there
are only 7 of them when plev has 8 levels. This is a different
vertical grid from `ua`/`va`/`hus`/`zg`, which are on the 8 plev
levels themselves. The shape of each `ta_derived_layer_i` is
`(time, lat, lon)`; the layer pressure is the log-midpoint
`sqrt(plev[i] * plev[i+1])`.

**Masking derived T.** The stored `below_surface_mask(time, plev, lat,
lon)` is on plev levels, so users need to combine the two bounding
levels of each layer to get a layer mask. Layer `i` is invalid
wherever either `plev[i]` or `plev[i+1]` is below surface:

```python
import xarray as xr

ds = xr.open_zarr(<path-to-dataset.zarr>, consolidated=True)
i = 0  # e.g. the 1000-850 hPa layer
layer_mask_i = (
    ds.below_surface_mask.isel(plev=i)
    | ds.below_surface_mask.isel(plev=i + 1)
).astype(bool)
ta_layer_i_valid = ds[f"ta_derived_layer_{i}"].where(~layer_mask_i)
```

During ingest we apply a **cascading nearest-above fill** to the
derived T layers (top-down: each masked layer inherits the layer
above), so the stored values are NaN-free and physically plausible
even in below-surface columns. Downstream code that wants to ignore
filled cells should apply the layer mask recipe above.

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
  Conservative regridding preserves the area-weighted integral but
  can leave sub-epsilon floating-point residuals (e.g., tiny negatives
  in a physically non-negative flux). We deliberately do **not** clip
  those — clipping would break the integral conservation. The sanity
  checks tolerate a small ``_EPS`` margin below / above the nominal
  physical range for this reason.

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
<output_directory>/                     # e.g. ./data/cmip6-daily-pilot/v0
  <source_id>/<experiment>/<variant_label>/
    data.zarr/                          # one per (src, exp, member)
      metadata.json                     # sidecar; also serves as the
                                        # "done marker" for resumable
                                        # re-runs
  index.csv                             # one row per dataset attempted
  index.parquet                         # ditto, when pyarrow available
  stats/<source_id>/{mean,std}.nc       # later, per-model
```

One zarr per `(source_id, experiment, variant_label)`. Chunk/shard per
Issue 6 (inner `time=1`, outer `time=365`).

## Dependencies

See `requirements.txt`. Most are already present in the base `fme`
conda env (numpy, pandas, xarray, zarr, fsspec, dacite, pyyaml); the
file also lists what `process.py` additionally needs (`xesmf`,
`gcsfs`, `cftime`, `pyarrow`).

**Install**: `pip install -r requirements.txt` works for most of it,
but `xesmf`/`esmpy` wrap the ESMF C library and install more reliably
via conda: `conda install -c conda-forge xesmf esmpy`.

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
3. Validate `cell_methods`.
4. Regrid to F22.5 (Gauss-Legendre 45 x 90) via `xesmf` (bilinear for
   state, conservative for fluxes).
5. Below-surface nearest-above fill + emit time-varying
   `below_surface_mask(time, plev, lat, lon)` (uint8).
6. Derived `ta_derived_layer_{0..6}` from `zg` + `hus`.
7. Linear-interpolate monthly forcings onto the daily axis; attach
   static fields.
8. Time-subset per config.
9. Write zarr with zarr v3 chunks+shards; drop sidecar metadata.json.
10. Append a row to the central `index.{csv,parquet}`.

**Resumability.** Re-running is cheap: the `metadata.json` sidecar
serves as the "completed" marker. A complete sidecar → skip. A zarr
directory without a sidecar → treated as partial, deleted, and
re-processed. `--force` rebuilds everything; `--dry-run` lists the
selection without processing; `--source-ids`/`--max-datasets` are
debug aids.

Per-dataset jobs are embarrassingly parallel. Pilot runs locally
(single-process; dask + argo wrapping comes later if needed).

Normalization stats (per-model, NaN-aware) are a separate later step
over the produced zarrs.

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
- **Issue 6 — Chunking.** Matches the `scripts/data_process`
  convention: inner zarr v3 chunks of `time=1` (per-timestep), outer
  shards of `time=365` (~one shard per year per variable). Per-shard
  size: 3D ~47 MB, 2D ~6 MB, mask ~12 MB — all within healthy
  GCS-object bounds. `shard_time: None` available as a debug escape
  hatch for unsharded writes.
- **Issue 2 — Label schema.** Strict reading: label =
  `(source_id, physics_index p)`. Realization `r`, initialization `i`,
  and forcing `f` are within-label variation. Stored as a scalar
  coord on each zarr and as a column in the index, in composite
  `{source_id}.p{p}` form (e.g. `"CanESM5.p1"`, `"GISS-E2-1-G.p3"`).
  Always includes `.pN` even for single-`p` source_ids, mapping 1:1
  to CMIP6 metadata. Helper: `config.make_label`. Inventory shows ~57
  labels across 53 source_ids (3 models publish alternate `p`:
  CMCC-CM2-SR5, CanESM5, GISS-E2-1-G). `variant_label` and parsed
  `r`/`i`/`f` are also retained in the zarr and index for downstream
  flexibility, though they are not part of the label itself.
- **Issue 9 — Index schema.** `DatasetIndexRow` dataclass in
  `index.py` captures identity, provenance (input zstores), processing
  (regrid methods, mask source, target grid), output (zarr path, time
  range, variables present), and audit (status, skip_reason, warnings).
  Outputs: `index.csv` always; `index.parquet` when a parquet engine
  is available; per-successful-zarr `metadata.json` sidecar alongside
  the data.

## Open Issues

### Known Pangeo data-quality issues (to raise with Pangeo)

- **CESM2-WACCM `historical r2i1p1f1` `psl`** has its 1850–2014
  series **stored twice end-to-end** in the Pangeo mirror at
  `gs://cmip6/.../day/psl/gn/v20190227/`. Both halves are
  bit-identical (we verify with `np.allclose`); the other day-table
  variables for the same member are clean (60226 timestamps, no
  duplicates). Symptoms point at a mirror-side ingest error where
  the per-variable concatenation appended the same source-file list
  twice without truncating. Until Pangeo re-publishes, the pilot
  enables `allow_dedupe: true` for this dataset via the per-dataset
  override in `pilot.yaml`.

- **ACCESS-CM2 `ssp585 r1i1p1f1`** has variables backed by
  inconsistent underlying zarr stores:

  | variable | time range |
  |----------|-----------|
  | `ua`, `va`, `zg`, `psl`, `uas` | 2015-2100 (standard ssp585) |
  | `sfcWind`, `huss`, `hfss` | 2015-2300 (ssp585 + extension) |
  | `tas`, `hfls` | 2201-2300 (just the tail — ssp585-over?) |
  | `pr` | 2251-2300 (only the last 50 years) |

  Taking the intersection across the CMIP6-required ssp585 window
  gives zero timesteps. Our pipeline detects this and skips with a
  descriptive reason. Other members (`r2`, `r3`, etc.) and historical
  runs are fine — this is a per-(member) catalog anomaly, not a
  systematic ACCESS-CM2 problem.

### Known model-side data quirks (sanity warnings observed, datasets still write)

- **INM-CM4-8 `zg` at 10 hPa** is ~22 km in a subset of grid cells
  vs the expected ~32 km elsewhere. Drops layer-6 (50-10 hPa)
  thickness to ~2.9 km in those cells, which makes
  `ta_derived_layer_6` come out ~61 K (clearly unphysical — the
  real polar stratosphere is ~190-220 K). Likely a fill-value /
  QC issue in INM-CM4-8's published `zg` at the very top of the
  atmosphere.
- **CESM2-FV2 `sftlf`** comes back from the conservative regrid at
  up to ~114% in some cells (way beyond the typical few-percent
  edge overshoot). Source publication likely has cells with
  values outside [0, 100] or with an unusual fill-value scheme
  that conservative weighting amplifies. Doesn't break the data,
  but the land-mask numerics are off.

### Known regridding / data-pipeline limitations

- **CESM2 SImon `siconc`** trips ESMF's regridder with `rc = 506`
  even with correctly-converted `(N+1, M+1)` vertex bounds (via
  `cf_xarray.bounds_to_vertices`). Likely a pole/tripolar-pivot
  cell issue in CESM2's specific ocean grid. The per-forcing
  try/except catches it and writes the rest of the dataset
  without siconc; the sidecar records the failure in `warnings`.
  Fixing it properly probably means clamping out-of-range lat
  values before handing the grid to xesmf, or switching to a
  different regridding backend for ocean grids.
- **`use_cftime=True` deprecation warning** in `_open_zstore`. The
  kwarg form is deprecated in newer xarray; should migrate to
  ``decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)``.
  Cosmetic for now.
- **Zarr v3 `NullTerminatedBytes` + consolidated-metadata warnings**
  on every write. Both flag v3-portability concerns but don't affect
  xarray+zarr-python reads. Eventually: drop string scalar coords
  (``type``, ``height``) and decide whether to keep
  ``consolidated=True``.

### Deferred engineering work

- **Per-dataset presence table.** The central `index.csv` already
  carries `variables_present` per dataset as a JSON list; a readable
  cross-model pivot (rows = dataset, columns = variable, values = 1/0)
  would make it easy to eyeball coverage at a glance. Easy follow-up.
- **Normalization stats**. Per-model mean/std over the produced
  zarrs. Separate later step; not yet written.
- **Proper `siconc` regridding for CESM2 (and any similarly affected
  models)**. Investigate pole/tripolar cell trimming or bypass.
- **Sanity-check upper bounds for `hfss`/`hfls`** were tuned to the
  models we've seen so far. May need further widening for more
  extreme publications.

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
