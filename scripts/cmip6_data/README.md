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

### Surface and ocean variables

Variables whose CMIP6 source table varies across datasets (e.g. some
models publish daily `Eday.ts`, others only monthly `Amon.ts`) get
**source-prefixed output names** so multiple cadences/tables can
coexist in a single dataset. The training-side variable-masking
machinery handles which ones are actually populated per model.

The naming convention is `{table}_{var}` lowercased — e.g.
`amon_ts`, `eday_ts`, `simon_siconc`, `siday_siconc`, `oday_tos`,
`omon_zos`. The bare atmospheric daily variables (`ua1000`, `tas`,
`pr`, …) come from one canonical table (`day`) for every model and
keep their unprefixed names.

**Monthly tables (causal previous-month transform applied).**
Each day receives the *previous calendar month's* mean — strictly
causal, no future leakage. The transform is applied at ingest, so
the stored data is already daily-aligned. ~15–45 day staleness is
inherent at this cadence.

- `amon_ts` from `Amon.ts` — surface temperature (SST over ocean,
  ice-top over sea ice, skin over land). Universal fallback — ~64
  models publish it.
- `simon_siconc` from `SImon.siconc` — sea-ice fraction.
- `simon_sitemptop` from `SImon.sitemptop` — sea-ice top T.
- `omon_zos` from `Omon.zos` — sea surface height (integrates
  full-column ocean density; broad coverage).
- `omon_hfds` from `Omon.hfds` — net downward heat flux at ocean surface.
- `omon_mlotst` from `Omon.mlotst` — mixed layer depth.
- `omon_tob` from `Omon.tob` — ocean bottom temperature.

**Daily tables (drop-in when published).** Source axis already
matches the daily target; we reindex nearest-neighbor.

- `eday_ts` from `Eday.ts` — same definition as `amon_ts` but
  daily (~21/37 eligible ESGF models, ~1/30 on Pangeo).
- `oday_tos` from `Oday.tos` — daily SST (~29/37 ESGF).
- `oday_tossq`, `oday_omldamax`, `oday_sos` — daily SST², daily-max
  MLD, daily SSS (sparser).
- `siday_siconc`, `siday_sitemptop`, `siday_sithick` — daily
  sea-ice fraction / top T / thickness (~22–25 / 37 ESGF).

**Per-variable masks.** Ocean and sea-ice variables have NaN over
land (and, for sea-ice variables, over ice-free cells). For each
such variable the pipeline writes a companion `{output_name}_mask`
channel (uint8: 1 = valid, 0 = invalid), and **fills the NaN
regions via iterative horizontal diffusion** so the stored value
field is NaN-free. The mask is 2D `(lat, lon)` when the valid
pattern is time-invariant (e.g. ocean-only variables on the static
land mask) and 3D `(time, lat, lon)` when it varies in time (sea-
ice variables). See **Ocean fill** below for the fill scheme.

**Atmospheric surface temperature** variables (`amon_ts`, `eday_ts`)
are full-surface global fields (model's own area-weighted composite)
and do **not** get a per-cell mask. Heterogeneity across datasets is
handled at the training level (`allow_variable_masking`).

Coverage on ESGF tiers cleanly: 21/37 eligible models publish
`eday_ts` (Tier A drop-in for amon_ts); 10 more publish `oday_tos`
+ `siday_siconc` for a daily SST + ice-mask composite (Tier B);
remaining models fall back to the causal monthly path. See
`process_esgf.py` and the surface-T discussion in `training.md` for
details.

### External forcings

Prescribed input4MIPs / LUH2 forcing fields, shared across all models
within a scenario. These are the boundary conditions that distinguish
historical from ssp245 from ssp585.

**Implemented:**

- **`input4mips_co2`** — global-mean annual CO2 concentration (ppm),
  broadcast to `(time, lat, lon)`. Captures the dominant greenhouse
  forcing — ~80% of the radiative-forcing difference between scenarios.

  - Historical (≥1959): NOAA Mauna Loa annual mean record. Differs
    from the CMIP6-prescribed Meinshausen et al. values by <1 ppm at
    any year, functionally equivalent for emulator training.
  - Pre-1959: constant-extrapolation fallback to the 1959 NOAA value
    (315.97 ppm). For pre-1959 scientific exactness, point the time
    subset to ≥1959 — the published Meinshausen historical file
    isn't currently indexed on ESGF.
  - SSP245 / SSP585: UoM input4MIPs annual files (Meinshausen et al.
    2017, 2015–2500). For 2015 onwards the SSP file values supersede
    NOAA.
  - Mapped onto each model's daily axis via **causal previous-year**:
    every day in calendar year `Y` reads year `Y-1`'s value.

**Deferred to a follow-up commit:**

- **`input4mips_so2`** (gridded monthly) — anthropogenic sulfate
  aerosol precursor.
- **`input4mips_bc`** (gridded monthly) — black carbon.
- **`luh2_forest`** (gridded annual) — total forest fraction from LUH2.

These three add the aerosol-cooling and land-surface-change axes the
embedding will need beyond pure GHG forcing. CO2 alone captures most
of the inter-scenario signal, so it's a useful first step.

Other input4MIPs forcings (CH4, N2O, CFC equivalents, ozone, volcanic
aerosol, solar irradiance, biomass burning) are deferred. Most are
either strongly correlated with CO2 across scenarios (CH4, N2O),
identical across all scenarios (solar, volcanic), or of secondary
importance.

### External forcings staging

`external_forcings.py` writes a small per-scenario zarr at
`<output_directory>/external_forcings/<experiment>.zarr` containing an
annual `co2(time)` time series. `process.py` and `process_esgf.py`
opportunistically attach those forcings to each per-model dataset at
processing time — if the per-scenario zarr is absent they record a
warning and the per-model output simply lacks the `input4mips_*`
variables (training handles the missingness via the existing
`allow_variable_masking` machinery).

```
python external_forcings.py --output-directory ./data/cmip6-daily-pilot/v0
python external_forcings.py --output-directory ... --experiments historical ssp585
python external_forcings.py --output-directory ... --force  # rebuild
```

### Ocean fill

Ocean and sea-ice variables come back from regridding with **large
2D NaN regions** (continents for ocean variables; continents + ice-
free ocean for sea-ice variables). Unlike the column-wise
`nearest_above_fill` used for below-surface plev cells, ocean fill
runs in the horizontal: iterative 4-connected neighbor-mean
diffusion that propagates valid-cell values into the NaN region.
Periodic in longitude, clamped in latitude. After ~50 iterations
the entire grid is filled smoothly; any disconnected island of NaN
falls back to the global mean of originally-finite cells.

The mask channel preserves the valid extent so the network can
distinguish "actual ocean cell" from "extrapolated value." See
`fill_horizontal_diffuse` and `emit_mask_and_fill` in
`processing.py`.

**Not included:** 3D ocean fields (`thetao`, `so`, `uo`, `vo`) due
to data volume at 50–75 depth levels. Surface chlorophyll
(`chlos`) may be added as an optional diagnostic if cheap, but is
not a priority for physical climate.

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
  failure. Handled by `clamp_static_fractions` in `processing.py`,
  which clips `sftlf` to [0, 100] after regridding and records the
  pre-clip overshoot in `index.warnings` so the regridder defect
  remains visible. Since `sftlf` is a fraction with no compensating
  variable in the pipeline, clipping doesn't break any budget. See
  `figures/cesm2_fv2_sftlf_overshoot.png`.

- **KACE-1-0-G sparse NaN at 10 hPa**: All KACE-1-0-G datasets have
  sparse NaN values (typically 1 grid cell per timestep) in the 10 hPa
  variables (ua10, va10, hus10, zg10). These propagate through the
  network as NaN input and produce all-NaN model output.
  **KACE-1-0-G is excluded from training** via
  `selection.exclude_source_ids` in `pilot.yaml`.

- **EC-Earth3/historical/r4i1p1f1 missing plev timesteps**: The last
  13 timesteps (indices 347-359, late December) have all-NaN values
  for all 3D pressure-level variables (ua, va, hus, zg at all levels),
  while surface variables (tas, huss, psl, pr) and derived layer-mean
  temperatures are present. Likely incomplete data upload.
  Other EC-Earth3 members are unaffected.

- **HadGEM3-GC31-MM/historical/r2i1p1f3 missing plev timesteps**: 22
  timesteps (indices 40-61, mid-February) have all-NaN values for all
  3D pressure-level variables, while surface variables are present.
  Other HadGEM3-GC31-MM members are unaffected.

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

- **Radiative forcing for coupled runs.** Model-diagnosed CO2/CH4
  (`AERmon`) is available for only ~3 models. For coupled
  ocean-atmosphere runs, options are: (1) use input4MIPs prescribed
  forcing (scenario, year) → shared scalar time series, (2) encode
  scenario+time and let the model learn the forcing trajectory, or
  (3) use TOA incoming solar (`rsdt`) as a proxy for radiative
  forcing and time encoding for the rest.
- **Core variable gate relaxation.** Now that heterogeneous-variable
  training is supported, `defaults.max_core_missing` controls how
  many core variables may be absent without skipping the dataset.
  The most common single missing core var is `zg` (20/62 models),
  followed by `va`/`hus`/`ua` (~10 each, mostly E3SM family) and
  `huss` (9 models). Default is **3** — generous because dataset
  generation is expensive and we'd rather have the data on disk and
  filter at training time. Coverage by threshold (37 at 0): +10 at
  1, +13 at 2, +15 at 3, +17 at 4. Note: when `zg` or `hus` is
  missing, the derived layer-T variables are not emitted for that
  dataset.
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
