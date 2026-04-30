# CMIP6 Daily Pilot Dataset

Pilot project to download and stage a multi-model CMIP6 daily-mean dataset
for an ACE-style emulator with a per-model label embedding. Scope is
deliberately narrow: daily cadence only, CMIP6 only, drastically time-subset
at first with config knobs to scale up to the full roster later.

## Goals

- **Maximize data availability.** Default: pull every available member for
  every included model for both `historical` and `ssp585`. Holdout and
  per-model member caps are future concerns, not ingest-time decisions.
- Support ~30–50 CMIP6 models to train a robust source-model label
  embedding that can be fine-tuned on unseen models. Current count:
  37 models (32 from Pangeo, 5 ESGF-only) with 146 eligible
  model/experiment/member combinations.
- Heterogeneous variable sets across models are expected; ingest records
  what each model has. Downstream training support for heterogeneous
  variables is a separate future code change.
- Configurable via YAML loaded into a dataclass with `dacite`; defaults
  cover most datasets, per-dataset overrides handle exceptions.

## Dataset Specification

### Experiments & members

- `historical` (1850–2014), `ssp245` (2015–2100), and `ssp585` (2015–2100).
- **All available members** by default. Whether a per-model cap is applied
  is a future decision.
- Experiment is recorded as metadata and may be used in experiments, but
  **not** encoded in the label.

### Variables (CMIP6 `day` table, CF names kept as-is)

**Core state (required — any model missing any of these is dropped):**

On `plev8` (flattened to pressure-named 2D variables in the zarr
output — see **Plev flattening** below): `ua`, `va`, `hus`, `zg`
2D: `tas`, `huss`, `psl`, `pr`

Notably **absent from core**:

- `ta` (air temperature on `plev`) — Pangeo daily coverage is essentially
  nil (3 models). Instead we derive 7 layer-mean temperatures from `zg` +
  `hus` via the hypsometric equation at ingest time and store them as
  `ta_derived_layer_{lo}_{hi}` (e.g. `ta_derived_layer_1000_850`). This
  is a proxy, not a true temperature; it's treated as derived throughout
  and labelled as such in the zarr.
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

Pulled from monthly tables. For each day, the model receives the
**previous month's mean** — strictly causal, no future leakage. (The
current interpolation approach of placing monthly means at month
midpoints and linearly interpolating is non-causal: values on day 20
of month N already contain information from month N+1. The
previous-month scheme eliminates this at the cost of ~15–45 day
staleness, which is acceptable for slowly-varying boundary conditions.)

Daily data from `Oday`/`Eday` tables will replace the monthly forcing
for models that have it (see **Ocean variables** below); the
previous-month scheme is the fallback for models without daily data.

- `ts` from `Amon` — surface temperature (SST over ocean, ice-top temp
  over sea ice, skin temp over land). 64 models publish it.
- `siconc` from `SImon` — sea-ice fraction on the ocean grid;
  regridded to the F22.5 target. 57 models.

### External forcings (planned — not yet in pipeline)

Prescribed input4MIPs forcing fields, shared across all models within
a scenario. These are the boundary conditions that distinguish
historical from ssp245 from ssp585. Stored per experiment (one copy
per scenario's time window).

- **CO2 concentration** (global scalar, annual) — dominant greenhouse
  gas forcing. Captures ~80% of the radiative forcing difference
  between scenarios.
- **SO2 emissions** (gridded monthly) — anthropogenic sulfate aerosol
  precursor. Drives spatially heterogeneous aerosol cooling that
  differs dramatically between SSPs. Weather-scale effects on cloud
  microphysics and precipitation.
- **BC emissions** (gridded monthly) — black carbon. Absorbing aerosol
  with regional warming effects and impacts on atmospheric stability.
- **Total forest fraction** (gridded annual, from LUH2) — land use
  change proxy. Affects surface albedo, roughness, and
  evapotranspiration with immediate weather-scale signatures.

These four fields were chosen as a compact set that captures the main
independent axes of forcing variation between scenarios: greenhouse
warming (CO2), aerosol cooling (SO2, BC), and land surface change
(forest fraction). With only 3 scenarios (historical, ssp245, ssp585)
the forcings are collinear in the climate mean, but their spatial
patterns provide weather-scale inductive bias — the model can learn
local forcing→response relationships (e.g. high SO2 → brighter clouds)
that generalize across grid cells.

Other input4MIPs forcings (CH4, N2O, CFC equivalents, ozone, volcanic
aerosol, solar irradiance, biomass burning) are deferred. Most are
either strongly correlated with CO2 across scenarios (CH4, N2O),
identical across all scenarios (solar, volcanic), or of secondary
importance. They can be added later as more scenarios are included.

### Ocean variables (planned — not yet in pipeline)

Daily ocean surface fields from `Oday` (include when available, do not
drop models that lack them):

- `tos` — sea surface temperature (~42 models on ESGF, 29 on Pangeo).
  Better than the monthly `ts` forcing for SST since it's daily and
  ocean-specific. Ocean-grid only (NaN over land).
- `sos` — sea surface salinity (~34 models).
- `tossq`, `sossq` — squared SST/SSS for sub-daily variance (~25–27 models).
- `omldamax` — daily max mixed layer depth (~28 models).

Monthly 2D ocean diagnostics from `Omon` (interpolated to daily, as
with `ts`/`siconc`) to provide deep-ocean memory without 3D data
volume:

- `zos` — sea surface height. Integrates full-column density, so it
  encodes deep heat/salt content in a single 2D field (~56 models).
- `hfds` — net downward heat flux at ocean surface (~51 models).
- `mlotst` — mixed layer depth (~34 models).
- `tob` — ocean bottom temperature (~29 models).

All ocean variables live on curvilinear/tripolar ocean grids and
require regridding to F22.5 (bilinear; conservative generally fails
for ocean grids lacking usable vertex bounds — same issue as `siconc`).

**Not included:** 3D ocean fields (`thetao`, `so`, `uo`, `vo`) due to
data volume at 50–75 depth levels. Surface chlorophyll (`chlos`) may be
added as an optional diagnostic if cheap, but is not a priority for
physical climate.

### Static per-model fields

From the CMIP6 `fx` table; broadcast along time by the data loader.

- `sftlf` — land fraction (land-sea mask). 51 models.
- `orog` — surface altitude (orography). 47 models.

### Expected model coverage

**37 models** satisfy all core variable requirements across both the
Pangeo catalog and ESGF, yielding **146 eligible tasks**
(model/experiment/member combinations): 60 historical, 48 ssp245,
38 ssp585. 32 models are sourced from Pangeo GCS mirrors; 5 are
ESGF-only (ACCESS-ESM1-5, AWI-ESM-1-REcoM, CESM2-WACCM-FV2,
FGOALS-f3-L, IPSL-CM6A-LR). The multi-member cap (Issue 3) retains
up to 3 realizations per `(source_id, experiment, p, f)` label.

### Derived variables

**Layer-mean temperature `ta_derived_layer_{lo}_{hi}`.** Computed at
ingest from `zg` and `hus` using the hypsometric equation under
hydrostatic balance — an excellent approximation for daily means at
4°×4°:

```
T_v^{layer}  = g · (z_hi - z_lo) / (R_d · ln(p_lo / p_hi))
q^{layer}    = (q_lo + q_hi) / 2
T^{layer}    = T_v^{layer} / (1 + 0.608 · q^{layer})
```

with `R_d = 287.05 J/kg/K`, `g = 9.80665 m/s²`. All inputs are
co-located on `plev8` levels, so the computation is just differences
and averages along the `plev` dimension — no interpolation. The 7
layers are named by their bounding pressures:
`ta_derived_layer_1000_850`, `ta_derived_layer_850_700`, ...,
`ta_derived_layer_50_10`. Expected error from applying an
instantaneous relation to daily means is <<1 K at these scales; far
below inter-model spread.

Variables are named `ta_derived_layer_*` throughout (never `ta`) to
keep it unambiguous that these are proxies, not native CMIP6
temperatures.

**Vertical-grid note (important for downstream masking).** The derived
temperatures live on **layers between adjacent plev levels**, so there
are only 7 of them when plev has 8 levels. This is a different
vertical grid from `ua`/`va`/`hus`/`zg`, which are on the 8 plev
levels themselves. The shape of each layer variable is
`(time, lat, lon)`.

**Masking derived T.** The stored per-level
`below_surface_mask{hPa}(time, lat, lon)` variables (see **Plev
flattening**) give the mask at each pressure level. To mask a derived
layer, combine its two bounding levels — a layer is invalid wherever
either bounding level is below surface:

```python
import xarray as xr

ds = xr.open_zarr(<path-to-dataset.zarr>, consolidated=True)
layer_mask = (ds["below_surface_mask1000"] | ds["below_surface_mask850"]).astype(bool)
ta_valid = ds["ta_derived_layer_1000_850"].where(~layer_mask)
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

  **Regrid method by variable.** The *requested* method is determined
  by `RegridConfig.method_for()` in `config.py`: variables in
  `FLUX_LIKE_VARIABLES` get conservative, everything else bilinear.
  The *actual* method may differ if the source file lacks the grid
  bounds (`lon_bnds`/`lat_bnds` or equivalent) that conservative
  regridding requires — in that case `make_regridder` falls back to
  bilinear and logs a warning. The actual method used for each
  variable is recorded in the `regrid_methods` field of the dataset
  index row and the per-dataset `metadata.json` sidecar.

  | Variable | Category | Requested | Actual (typical) | Notes |
  |----------|----------|-----------|------------------|-------|
  | `ua`, `va`, `hus`, `zg` | core (plev) | bilinear | bilinear | |
  | `tas`, `huss`, `psl` | core (2D) | bilinear | bilinear | |
  | `pr` | core (2D flux) | conservative | conservative | atmos grid has bounds |
  | `rsdt`, `rsut`, `rlut` | optional (TOA) | conservative | conservative | atmos grid has bounds |
  | `rsds`, `rsus`, `rlds`, `rlus` | optional (sfc rad) | conservative | conservative | atmos grid has bounds |
  | `hfss`, `hfls` | optional (sfc turb) | conservative | conservative | atmos grid has bounds |
  | `sfcWind`, `uas`, `vas` | optional (wind) | bilinear | bilinear | |
  | `ts` | forcing (Amon) | bilinear | bilinear | |
  | `siconc` | forcing (SImon) | conservative | **bilinear** | ocean grid; see below |
  | `sftlf` | static (fx) | conservative | conservative | atmos grid has bounds |
  | `orog` | static (fx) | bilinear | bilinear | |

  **`siconc` fallback.** Sea-ice concentration is published on the
  ocean grid (`SImon` table), which for most CMIP6 models is a
  curvilinear or tripolar grid whose vertex bounds are either absent
  or in a format that `xesmf`'s conservative regridder cannot ingest
  (e.g. CESM2's tripolar pivot cells cause `rc = 506`). The pipeline
  falls back to bilinear for `siconc` in practice for nearly all
  models. Since `siconc` is a fraction field (0–100%), bilinear
  interpolation can produce slight undershoot/overshoot near sharp
  ice edges; the sanity checks tolerate this. A model whose ocean
  grid *does* carry usable rectilinear bounds would get conservative
  as requested — the actual method is always recorded per-variable.

### Plev flattening

The zarr output **does not** store a `plev` dimension. Instead,
`process.py` flattens every variable with a vertical dimension into
pressure-named 2D variables before writing:

**On-level variables** (`plev` dim) use `{var}{hPa}`:
- `ua` → `ua1000`, `ua850`, `ua700`, `ua500`, `ua250`, `ua100`, `ua50`, `ua10`
- Same pattern for `va`, `hus`, `zg`, and `below_surface_mask`

**Derived layer variables** (`plev_layer` dim) use
`{var}_{lo_hPa}_{hi_hPa}`:
- `ta_derived_layer` → `ta_derived_layer_1000_850`,
  `ta_derived_layer_850_700`, ..., `ta_derived_layer_50_10`

This makes all stored variables uniformly `(time, lat, lon)` or
`(lat, lon)`, which is compatible with the fme data loader. The
pressure values are readable directly from the variable name.

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

Two pipelines feed the same output directory and index:

### `inventory.py` — Pangeo dataset discovery & metadata

Queries the Pangeo GCS CMIP6 intake-esm catalog for our variable list ×
experiments and emits a tidy table (parquet/csv) with one row per
`(source_id, experiment, variant_label, variable)` and enough columns to
answer cross-model comparison questions: variables present, grid_label,
native calendar, time range, horizontal/vertical grid info, member counts
per model. Used both to measure Pangeo coverage and as input to
`process.py`. No data movement — metadata only.

### `process.py` — Pangeo per-dataset processing

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
9. Flatten `plev` dimension into pressure-named 2D variables (see
   **Plev flattening**).
10. Write zarr with zarr v3 chunks+shards; drop sidecar metadata.json.
11. Append a row to the central `index.{csv,parquet}`.

### `process_esgf.py` — ESGF per-dataset processing

Supplements `process.py` for models not on Pangeo's GCS mirror and
for additional experiments/members not available there. Driven by
`configs/process_esgf.yaml` + `inventory_esgf.csv`. Same output
format, same zarr layout and sidecar convention.

Key difference: downloads NetCDF files from ESGF HTTP servers one
variable at a time (to cap disk usage at ~50 GB), processes each
variable through regridding, then merges. Downloaded files are cached
in a scratch directory and cleaned up per-variable after regridding.

Currently adds 5 ESGF-only models (ACCESS-ESM1-5, AWI-ESM-1-REcoM,
CESM2-WACCM-FV2, FGOALS-f3-L, IPSL-CM6A-LR) and backfills ~30
additional experiment/member combinations for models already processed
via Pangeo.

**Resumability.** Same as `process.py`: `metadata.json` sidecar marks
completion. `--force` rebuilds; `--source-ids`/`--max-datasets` are
debug aids.

Per-dataset jobs are embarrassingly parallel. Pilot runs locally
(single-process; dask + argo wrapping comes later if needed).

Normalization stats (per-model, NaN-aware) are a separate later step
over the produced zarrs.

### `make_presence.py` — variable-presence views

Joins the central `index.csv` with the inventory to produce a
per-dataset view of "what's in this dataset, what was available in
the source, and what wasn't." Outputs four files into
`<output_directory>/`:

- `presence.csv` and `presence.parquet`: wide pivot, one row per
  attempted dataset, one column per variable. Cell encoding is
  `2` (written), `1` (available in source but not written), or
  `0` (not in source). Plus identity, status, calendar, mask
  source, warning count, and zarr path.
- `presence.png`: heatmap of the same matrix, columns grouped by
  category (core → derived → forcing → static → optional), rows
  sorted by `(source_id, experiment, variant_label)`. A side
  stripe shows dataset status. Skipped rows show up as walls of
  amber (source had it, we didn't ingest).
- `presence.md`: per-model rollup with one-line dataset summaries
  and a category-coverage table.

Run via:
```
python make_presence.py --config configs/pilot.yaml
```

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
- **Issue 5 — Pangeo-only vs ESGF.** Both Pangeo and ESGF are used.
  Pangeo provides the bulk of the data (32 models); ESGF adds 5 new
  models and ~30 backfill tasks via `process_esgf.py`. `ta` replaced
  by derived layer-T; `ps` replaced by `psl` + topography mask.
- **Issue 8 — Regridding.** `xesmf` targeting Gauss-Legendre F22.5
  (45 x 90). Conservative for fluxes/precip, bilinear for state.
  Weights cached per source grid.
- **Forcings wiring.** Monthly `ts` (`Amon`) + `siconc` (`SImon`)
  interpolated to daily; static `sftlf` + `orog` (`fx`) broadcast.
  ~21 source_ids have full core + forcing + static coverage in Pangeo.
- **Issue 7 — Below-surface masking.** Per-level time-varying
  `below_surface_mask{hPa}(time, lat, lon)` (uint8) per dataset
  (flattened from a single `(time, plev, lat, lon)` array; see
  **Plev flattening**). Primary derivation: NaN union across 3D plev
  variables (captures
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

- **INM-CM4-8 `zg` at 10 hPa** drops to as low as ~22 km in a small
  set of grid cells vs the expected ~32 km. Layer-6 (50-10 hPa)
  thickness collapses to ~2.9 km at those cells and the hypsometric
  `ta_derived_layer_6` reads ~61 K (real polar stratosphere is
  190-220 K).

  Diagnosis (verified against the raw Pangeo zarrs, not our
  pipeline output):
  - 33 affected (cell, day) pairs in 2010 across 7.9 million
    (cell, day) points at native 120 × 180 resolution — about 4 in
    a million.
  - **Each affected cell appears on exactly one day** and the bad
    cells are scattered across both poles, mid-latitudes, and the
    tropics with no spatial coherence.
  - **The values are present in the source zarr** (min = 20,530 m
    at native res before our regrid).
  - **INM-CM5-0** — same publishing centre, same cmorization
    pipeline — is clean (zero cells below 25 km in 2010), so this
    is not a pipeline-side artefact.

  Most likely cause: numerical edge cases in INM-CM4-8's
  stratospheric output. INM-CM4-8 is the lower-top precursor to
  INM-CM5-0; 10 hPa sits right at or above its model top, so each
  cell at that level is sensitive to interpolation from limited
  native levels and to occasional one-step instabilities that
  hyperdiffusion damps out the next day.

  **INM-CM4-8 is excluded from training** via
  `selection.exclude_source_ids` in `pilot.yaml`; the data on disk
  is preserved for inspection but new ingest runs skip it.
  See `figures/inm_cm4_8_zg_top.png` — the bad cells appear as
  scattered dark pixels in the spatial map and a low-end histogram
  tail (down to ~22,125 m) that the reference CanESM5 lacks.

- **CESM2-FV2 `sftlf`** comes back from the conservative regrid at up
  to ~114% along the **southernmost row of the target grid** (lat
  ≈ -87 to -90°). The rest of the field is in [0, 100] as expected.
  The defect is purely the row of polar cells, so it's a regridder
  pole-handling effect against CESM2-FV2's specific source grid
  layout — same family of issue as the CESM2 SImon siconc rc=506
  failure. Doesn't break the data; the land-mask numerics are off
  in that one row of cells. Worth either masking out the
  southernmost row at training time or fixing the pole-cell
  weighting in xesmf as a follow-up. See
  `figures/cesm2_fv2_sftlf_overshoot.png`.

### Known ESGF data-quality issues

- **CESM2-WACCM-FV2** publishes files on ESGF with overlapping time
  ranges that produce data-identical duplicate timestamps after
  concatenation. Handled via `allow_dedupe: true` override in
  `configs/process_esgf.yaml`. The duplicates are verified to be
  data-identical before deduplication; materially different duplicates
  (indicating a simulation boundary) raise `SimulationBoundaryError`.

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
- **`use_cftime=True` deprecation warning** — migrated to
  ``decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)`` in
  both pipelines.
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

- **Ocean variable ingestion.** Daily `Oday` variables (`tos`, `sos`,
  `tossq`, `sossq`, `omldamax`) and monthly `Omon` variables (`zos`,
  `hfds`, `mlotst`, `tob`) are planned but not yet in the pipeline.
  Requires adding ocean table queries to inventory, handling
  curvilinear ocean grids (bilinear fallback), and deciding whether
  `tos` replaces or supplements `ts` as the SST forcing. See
  **Ocean variables** section above for availability counts.
- **Radiative forcing for coupled runs.** Model-diagnosed CO2/CH4
  (`AERmon`) is available for only ~3 models. For coupled
  ocean-atmosphere runs, options are: (1) use input4MIPs prescribed
  forcing (scenario, year) → shared scalar time series, (2) encode
  scenario+time and let the model learn the forcing trajectory, or
  (3) use TOA incoming solar (`rsdt`) as a proxy for radiative
  forcing and time encoding for the rest.
- **Heterogeneous variables at train time.** Core code change in
  `fme/core/dataset/` and the stepper to allow per-dataset variable
  subsets. Tracked separately from this pilot. Once this is supported,
  revisit the "core variable" gate in `process.py` — currently any
  model missing any core variable is dropped entirely, but with
  heterogeneous-variable training we should allow a certain number of
  core variables to be missing. This would recover a significant
  number of models: e.g. 8 models have `hus` on pressure levels but
  not `huss` (near-surface specific humidity), and 17 models lack
  `zg` (geopotential height). Some of these could also be addressed
  by deriving missing variables from available ones (e.g. approximating
  `huss` from the 1000 hPa level of `hus`).
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
