# CMIP6 Daily Pilot Dataset

Pilot project to download and stage a multi-model CMIP6 daily-mean dataset
for an ACE-style emulator with a per-model label embedding. Scope is
daily cadence only, CMIP6 only, with both Pangeo and ESGF as data sources.

## Data plan (current target)

The production ingest targets **~415 (model, experiment, member)
datasets** totalling **~11.5 TB** on disk:

| Window | Coverage | Datasets | Storage |
|---|---|---|---|
| `historical` 1940-01-01 to 2014-12-31 | 75 years | ~120 | ~3.0 TB |
| `ssp126` 2015-01-01 to 2100-12-31 | 86 years | ~70 | ~2.0 TB |
| `ssp245` 2015-01-01 to 2100-12-31 | 86 years | ~80 | ~2.3 TB |
| `ssp370` 2015-01-01 to 2100-12-31 | 86 years | ~70 | ~2.0 TB |
| `ssp585` 2015-01-01 to 2100-12-31 | 86 years | ~75 | ~2.2 TB |

Each dataset is one zarr per `(source_id, experiment, variant_label)`
at the F22.5 Gauss-Legendre 45×90 grid, containing ~80 channels:
8 plev × 4 core 3D vars + 7 derived layer-mean T + 4 surface +
8 below-surface masks + 2 statics + surface-and-ocean variables with
masks + 4 external forcings (`input4mips_co2`, `input4mips_so2`,
`input4mips_bc`, `luh2_forest`). Configured in
`configs/pilot.yaml` (`defaults.time_subset`) and
`configs/process_esgf.yaml`.

Source split:
- **Pangeo** provides the bulk of historical, ssp245, ssp585 (233
  datasets after the multi-member cap).
- **ESGF** is the sole source for ssp126 and ssp370 (Pangeo lacks
  daily 3D state for those scenarios) and backfills ~50-60 additional
  datasets across the others.
- 1940 was chosen for the historical start to match our existing ERA5
  and SHiELD AMIP datasets (both 1940 onwards) and the modern
  reanalysis era.

See **Expected model coverage** and **External forcings** below for
the detailed breakdown.

## Goals

- **Maximize data availability.** Default: pull every available member
  for every included model across historical + all 4 SSPs (126, 245,
  370, 585). Holdout and per-model member caps are future concerns,
  not ingest-time decisions.
- Support ~30–50 CMIP6 models to train a robust source-model label
  embedding that can be fine-tuned on unseen models. With
  `max_core_missing=3` (the new default after heterogeneous-variable
  training landed) and Pangeo + ESGF combined, ~415 eligible
  (model, experiment, member) datasets across the five experiments.
- Heterogeneous variable sets across models are expected and supported
  end-to-end: ingest emits per-dataset variable lists and training
  reads them via the `allow_variable_masking` flag (see PR #1160).
- Configurable via YAML loaded into a dataclass with `dacite`; defaults
  cover most datasets, per-dataset overrides handle exceptions.

## Dataset Specification

### Experiments & members

- `historical` (1850–2014), `ssp126` / `ssp245` / `ssp370` / `ssp585`
  (each 2015–2100).
- Main training window is `historical` from 1940 (matches our ERA5 /
  SHiELD AMIP datasets and the modern reanalysis era) plus the full
  CMIP6 SSP window 2015–2100 for all four SSPs. Pre-1940 historical
  is on disk via Pangeo/ESGF but not in the current pilot.yaml
  time-subset.
- **All available members** by default. Whether a per-model cap is applied
  is a future decision.
- Experiment is recorded as metadata and may be used in experiments, but
  **not** encoded in the label.

### Variables (CMIP6 `day` + `CFday` tables, renamed to the SHIELD/ERA5 baseline convention on write)

**Output naming.** CMIP6 variables that have a baseline-dataset equivalent
(ERA5, SHIELD AMIP) are renamed at write time so the zarr columns line
up across data sources. The map is in ``CMIP_TO_OUTPUT_RENAMES`` in
``config.py``; each renamed variable carries an ``original_name``
attribute pointing back to the bare CMIP6 source name. Variables with no
baseline equivalent (e.g. ``psl``, ``wap500``, ``clwvi``, ``clivi``,
``sfcWind``) keep their CMIP6 names.

| CMIP6 source | Output name | CMIP6 source | Output name |
|---|---|---|---|
| `rlds` | `DLWRFsfc` | `rsdt` | `DSWRFtoa` |
| `rlus` | `ULWRFsfc` | `rsut` | `USWRFtoa` |
| `rsds` | `DSWRFsfc` | `rlut` | `ULWRFtoa` |
| `rsus` | `USWRFsfc` | `rsutcs` | `UCSWRFtoa` |
| `rldscs` | `DCLWRFsfc` | `rlutcs` | `UCLWRFtoa` |
| `rsdscs` | `DCSWRFsfc` | `rsuscs` | `UCSWRFsfc` |
| `hfls` | `LHTFLsfc` | `hfss` | `SHTFLsfc` |
| `pr` | `PRATEsfc` | `ps` | `PRESsfc` |
| `tas` | `TMP2m` | `huss` | `Q2m` |
| `uas` | `UGRD10m` | `vas` | `VGRD10m` |
| `zg500` | `h500` | `ta700` | `TMP700` |

Note that `psl` (mean sea-level pressure, ``day`` table) is **not**
renamed — it's kept distinct from `PRESsfc` (real surface pressure
from ``CFday.ps``).

**Core state (required — any model missing more than `max_core_missing`
of these is dropped):**

On `plev8` (flattened to pressure-named 2D variables in the zarr
output — see **Plev flattening** below): `ua`, `va`, `hus`, `zg`
2D: `tas` (→`TMP2m`), `huss` (→`Q2m`), `psl`, `pr` (→`PRATEsfc`)

Notably **absent from core**:

- `ta` (air temperature on `plev`) — Pangeo daily coverage is essentially
  nil (3 models). Instead we derive 7 layer-mean temperatures from `zg` +
  `hus` via the hypsometric equation at ingest time and store them as
  `ta_derived_layer_{lo}_{hi}` (e.g. `ta_derived_layer_1000_850`). This
  is a proxy, not a true temperature; it's treated as derived throughout
  and labelled as such in the zarr.
- `ps` (surface pressure) — not published on the standard `day` table by
  any CMIP6 model. It *is* available on `CFday` for ~5,600 models and is
  ingested when present (output `PRESsfc`). For models that publish
  neither, `psl` (mean sea-level pressure) + a topography/`zg`-derived
  surface mask stand in.

See **Derived variables** below for details.

**Optional (include per-model when published):**

Standard `day` table:

- Surface radiation: `rsds`, `rsus`, `rlds`, `rlus` (→ `DSWRFsfc`,
  `USWRFsfc`, `DLWRFsfc`, `ULWRFsfc`)
- TOA outgoing longwave: `rlut` (→ `ULWRFtoa`)
- Surface turbulent fluxes: `hfss`, `hfls` (→ `SHTFLsfc`, `LHTFLsfc`)
- Surface wind: `sfcWind`, `uas`, `vas` (last two → `UGRD10m`, `VGRD10m`)

`CFday` table (TOA shortwave, clear-sky pairs, and assorted single-
level diagnostics — see ``CFDAY_VARIABLES`` in ``config.py``):

- TOA shortwave: `rsdt`, `rsut` (→ `DSWRFtoa`, `USWRFtoa`). Neither is
  published on the standard `day` table by any CMIP6 model; both live
  on `CFday`.
- Clear-sky radiation (for cloud radiative effect): `rsutcs`,
  `rlutcs`, `rsdscs`, `rsuscs`, `rldscs` (→ `UCSWRFtoa`, `UCLWRFtoa`,
  `DCSWRFsfc`, `UCSWRFsfc`, `DCLWRFsfc`). No `rluscs` — clear-sky and
  all-sky surface upward LW are identical and CMIP6 doesn't publish a
  separate version.
- Real surface pressure: `ps` (→ `PRESsfc`).
- Single-level diagnostics: `ta700` (→ `TMP700`), `wap500` (kept as-is).
- Cloud water-path diagnostics: `clwvi` (total condensed), `clivi`
  (ice-only). Liquid path = `clwvi - clivi` when both are present.

The inventory + task-building code folds the `CFday` queries into the
day-cadence variable set automatically. ``cmip6_source_table(var_id)``
in ``config.py`` returns the table a variable lives on.

### Surface and ocean variables

Variables whose CMIP6 source table varies across datasets (e.g. some
models publish daily `Eday.ts`, others only monthly `Amon.ts`) get
**source-prefixed output names** so multiple cadences/tables can
coexist in a single dataset. The training-side variable-masking
machinery handles which ones are actually populated per model.

The naming convention is `{table}_{var}` lowercased — e.g.
`amon_ts`, `oday_tos`, `omon_zos`. Several variables additionally get
a SHIELD/ERA5-flavoured output name (recorded in `CMIP_TO_OUTPUT_RENAMES`):
``SImon.siconc`` → ``simon_sea_ice_fraction``, ``SIday.siconc`` →
``siday_sea_ice_fraction`` (both rescaled from % to fraction on
[0, 1]), ``Eday.prw`` → ``water_vapor_path``, ``Eday.ts`` →
``surface_temperature`` (matches the SHIELD/ERA5 baseline daily
surface-T name; `amon_ts` keeps its prefix because it's a different
cadence). ``original_name`` is preserved as an attribute on each
renamed variable. The bare atmospheric daily variables (`ua1000`,
`TMP2m`, `PRATEsfc`, …) come from one canonical table (`day` or
`CFday`) for every model and keep their unprefixed (and where
applicable, renamed) names.

**Monthly tables (causal previous-month transform applied).**
Each day receives the *previous calendar month's* mean — strictly
causal, no future leakage. The transform is applied at ingest, so
the stored data is already daily-aligned. ~15–45 day staleness is
inherent at this cadence.

- `amon_ts` from `Amon.ts` — surface temperature (SST over ocean,
  ice-top over sea ice, skin over land). Universal fallback — ~64
  models publish it.
- `simon_sea_ice_fraction` from `SImon.siconc` — sea-ice fraction,
  rescaled from % to [0, 1].
- `simon_sitemptop` from `SImon.sitemptop` — sea-ice top T.
- `omon_zos` from `Omon.zos` — sea surface height (integrates
  full-column ocean density; broad coverage).
- `omon_hfds` from `Omon.hfds` — net downward heat flux at ocean surface.
- `omon_mlotst` from `Omon.mlotst` — mixed layer depth.
- `omon_tob` from `Omon.tob` — ocean bottom temperature.

**Daily tables (drop-in when published).** Source axis already
matches the daily target; we reindex nearest-neighbor.

- `surface_temperature` from `Eday.ts` (renamed via
  `CMIP_TO_OUTPUT_RENAMES`) — same definition as `amon_ts` but
  daily (~21/37 eligible ESGF models, ~1/30 on Pangeo).
- `water_vapor_path` from `Eday.prw` — column-integrated water vapor
  (kg m⁻²). When `clwvi` (CFday) is also present, the pipeline emits
  a derived `total_water_path = water_vapor_path + clwvi` matching the
  CM4/SHIELD baseline.
- `oday_tos` from `Oday.tos` — daily SST (~29/37 ESGF).
- `oday_tossq`, `oday_omldamax`, `oday_sos` — daily SST², daily-max
  MLD, daily SSS (sparser).
- `siday_sea_ice_fraction` (from `SIday.siconc`, rescaled to [0, 1]),
  `siday_sitemptop`, `siday_sithick` — daily sea-ice fraction / top T /
  thickness (~22–25 / 37 ESGF).

**Per-variable masks.** Ocean and sea-ice variables have NaN over
land (and, for sea-ice variables, over ice-free cells). For each
such variable the pipeline writes a companion `{output_name}_mask`
channel (uint8: 1 = valid, 0 = invalid), and **fills the NaN
regions via iterative horizontal diffusion** so the stored value
field is NaN-free. The mask is 2D `(lat, lon)` when the valid
pattern is time-invariant (e.g. ocean-only variables on the static
land mask) and 3D `(time, lat, lon)` when it varies in time (sea-
ice variables). See **Ocean fill** below for the fill scheme.

**Atmospheric surface temperature** variables (`amon_ts`,
`surface_temperature`) are full-surface global fields (model's own
area-weighted composite) and do **not** get a per-cell mask.
Heterogeneity across datasets is handled at the training level
(`allow_variable_masking`).

Coverage on ESGF tiers cleanly: 21/37 eligible models publish
`surface_temperature` (Tier A drop-in for `amon_ts`); 10 more publish
`oday_tos` + `siday_sea_ice_fraction` for a daily SST + ice-mask
composite (Tier B); remaining models fall back to the causal monthly
path. See `process_esgf.py` and the surface-T discussion in
`training.md` for details.

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
  - Pre-1959 (1940–1958): hardcoded NASA GISS pre-Mauna-Loa
    composite (`_GISS_PRE_MAUNA_LOA_CO2` in `external_forcings.py`)
    — Etheridge et al. (1996) Law Dome ice-core record merged with
    the Scripps Mauna Loa series starting March 1958, GISS-recalibrated.
    Values rise from 311.3 ppm (1940) to 315.34 ppm (1958),
    capturing the small WWII-era growth slowdown. Agrees with the
    (retracted) CMIP6-vintage UoM-CMIP-1-2-0 historical file to <1
    ppm at any year.
  - SSP245 / SSP585: UoM input4MIPs annual files (Meinshausen et al.
    2017, 2015–2500). For 2015 onwards the SSP file values supersede
    NOAA.
  - Mapped onto each model's daily axis via **causal previous-year**:
    every day in calendar year `Y` reads year `Y-1`'s value.

- **`input4mips_so2`** — anthropogenic SO2 emission flux
  (kg m⁻² s⁻¹), summed across emission sectors and regridded
  conservatively to F22.5. Drives aerosol cooling with regional
  spatial signatures that differentiate SSPs.

  - Historical: **CMIP7-vintage** CEDS-CMIP-2025-04-18 (gn grid,
    1750-2023). The original CMIP6-vintage CEDS-2017-05-18 dataset
    that the CMIP6 model output was actually forced with has been
    retracted from every ESGF node we tried (LLNL, DKRZ, CEDA,
    ORNL) and isn't readily archived elsewhere. The CMIP7 update is
    a re-run of the same CEDS methodology on the same calendar
    years — values agree with the CMIP6-era version to within a
    few percent at the magnitudes the network sees. Recorded as
    such in `external_forcings.py` so the source vintage is auditable.
  - SSP245 / SSP585: CMIP6-vintage IAMC files (2015-2100, native
    0.5° gridded monthly).
  - Mapped onto each model's daily axis via causal previous-month.

- **`input4mips_bc`** — anthropogenic black carbon emission flux,
  same vintage / source / mapping as SO2.

- **`luh2_forest`** — total forest fraction (sum of primary forested
  land ``primf`` and potentially-forested secondary land ``secdf``)
  from the LUH2 v2 land-state dataset. Gridded annual, fraction in
  [0, 1]. Captures the land-surface-change axis of inter-scenario
  forcing variation.

  - Historical (850-2015): UofMD-landState-2-1-h (Hurtt et al. 2017).
  - SSP245 / SSP585 (2015-2100): UofMD-landState-MESSAGE-ssp245-2-1-f
    and UofMD-landState-MAGPIE-ssp585-2-1-f.
  - LUH2 publishes one ``multiple-states`` netCDF per scenario at
    0.25° resolution with ~12 land-use classes; we extract
    ``primf`` + ``secdf``, sum, replace ocean NaN with 0 (forest
    fraction over ocean is zero by construction), then bilinear-
    regrid to F22.5 (LUH2 lat/lon bounds aren't xesmf-conservative-
    compatible). Mapped onto each model's daily axis via causal
    previous-year.

Other input4MIPs forcings (CH4, N2O, CFC equivalents, ozone, volcanic
aerosol, solar irradiance, biomass burning) are deferred. Most are
either strongly correlated with CO2 across scenarios (CH4, N2O),
identical across all scenarios (solar, volcanic), or of secondary
importance.

#### Source-vintage note

The pilot uses CMIP6 model output as training data, so wherever possible
we use the CMIP6-vintage input4MIPs forcings that those models were
actually forced with. Two exceptions where the original CMIP6-vintage
file isn't available:

| Variable | Period | Used | Reason |
|---|---|---|---|
| `input4mips_co2` | Historical 1959–2014 | NOAA Mauna Loa annual | UoM-CMIP-1-2-0 not indexed on ESGF; NOAA values within <1 ppm at any year |
| `input4mips_co2` | Historical 1940-1958 | NASA GISS Law-Dome + Scripps composite | UoM-CMIP-1-2-0 not indexed on ESGF; the hardcoded annual values (311.3 → 315.34 ppm) agree with the CMIP6 prescribed record to <1 ppm |
| `input4mips_so2` / `_bc` | Historical 1750–2023 | CMIP7-vintage CEDS-CMIP-2025-04-18 | CMIP6-vintage CEDS-2017-05-18 retracted from ESGF |
| `luh2_forest` | All years | bilinear regrid | LUH2 native files don't expose xesmf-compatible lat/lon bounds; bilinear of a fraction field has bounded undershoot/overshoot near sharp land/ocean transitions |

In all three cases the substitute source's values are close enough to
the CMIP6-vintage prescribed values (<1 ppm CO2, <few % SO2/BC at the
scales relevant to a daily emulator) that the substitution is
acceptable for training. The CMIP6-vintage SSP files (UoM, IAMC) are
still on ESGF and used as-is.

### External forcings staging

`external_forcings.py` writes a small per-scenario zarr at
`<external_forcings_directory>/<experiment>.zarr` containing
`co2`, `so2`, `bc`, and `forest` on their native cadence dims
(`time_annual`, `time_monthly`, `time_annual_grid`). `process.py` and
`process_esgf.py` opportunistically attach those forcings to each
per-model dataset at processing time — if the per-scenario zarr is
absent they record a warning and the per-model output simply lacks
the `input4mips_*` / `luh2_*` variables (training handles the
missingness via the existing `allow_variable_masking` machinery).

`external_forcings_directory` is an optional top-level field on both
`ProcessConfig` and `ESGFProcessConfig`. When unset, it defaults to
`<output_directory>/external_forcings/` (the legacy layout). Pointing
several versioned output directories (`v0`, `v0-pilot`, `v1`, …) at
the same `external_forcings_directory` lets them share one staged
copy of the forcings — the inputs are version-independent until
`external_forcings.py` itself changes, so re-staging per version is
wasteful.

The script accepts both local paths and `gs://` URLs (fsspec-backed),
so the same code runs locally and from the argo workflow's stage-
externals template.

Local subset for testing (avoid the 5.8 GB historical LUH2 file):

```
# Minimal end-to-end exercise (~1 GB total download):
python external_forcings.py --output-directory ... --experiments ssp245
# Skip LUH2 entirely (just CO2/SO2/BC):
python external_forcings.py --output-directory ... --variables co2 so2 bc
```

Full staging (one-time ~30 GB download):

```
python external_forcings.py --output-directory ./data/cmip6-daily-pilot/v0
python external_forcings.py --output-directory ... --force  # rebuild
```

Production: the argo workflow's `stage-externals` template runs the
full version once per pilot version with `run_stage_externals: true`;
the resulting zarrs persist in GCS so subsequent process-dataset runs
read them directly without re-staging.

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

- `land_fraction` (from `sftlf`, rescaled from % to [0, 1] and renamed;
  ``original_name`` = `sftlf`). 51 models. Conservative regridding plus
  the `clamp_static_fractions` step gives a clean fraction field.
- `orog` — surface altitude (orography). 47 models.

### Land / ocean / sea-ice fractions

The pipeline emits a (land, ocean, sea-ice) fraction triple that sums
to 1 exactly in every cell:

- `land_fraction` — static, from `sftlf`, on [0, 1].
- `simon_sea_ice_fraction` / `siday_sea_ice_fraction` — time-varying,
  from `SImon.siconc` / `SIday.siconc` (rescaled from % to [0, 1]).
- `simon_ocean_fraction` / `siday_ocean_fraction` — derived as
  ``1 − land_fraction − sea_ice_fraction``. Coastal cells where
  ``land + ice > 1`` (typically from the horizontal-diffusion fill
  spilling sea-ice over the land mask) get the excess **pushed back
  into `sea_ice_fraction`** and `ocean_fraction` clipped to zero, so
  the identity ``land + ice + ocean = 1`` holds exactly. See
  `derive_ocean_and_correct_sea_ice` in `processing.py`. This matches
  the convention used by the ERA5 build pipeline.

### Temperature unit harmonization

CMIP6 publishes `tos` / `tob` / `sitemptop` in °C by spec and the rest
(`tas`, `ts`, `ta`) in K, but publishers occasionally deviate. After
all variables are assembled and renamed, `harmonize_temperature_to_kelvin`
walks every temperature variable in the dataset and converts °C → K
based on the source ``units`` attribute (falling back to the CMOR spec
default when ``units`` is missing). Already-K variables pass through
unchanged. Any conversion or unrecognized unit is recorded in
``index.warnings``.

### Expected model coverage

With the current defaults (`max_core_missing=3`, `max_members_per_f=5`)
and Pangeo + ESGF combined, the pilot covers **~415 eligible
(model, experiment, member) datasets** across the five experiments
(estimate is conservative; the cap-bump from 3 → 5 pulls in more
multi-member ensembles for models like CanESM5 and HadGEM3-GC31-LL):

| Scenario | Datasets (estimate) | Source mix |
|---|---|---|
| historical | ~120 | Pangeo-dominated (103), ESGF adds ~17 |
| ssp126 | ~70 | **ESGF-only** — Pangeo lacks daily 3D state for ssp126 |
| ssp245 | ~80 | Pangeo (69) + ESGF backfill (~11) |
| ssp370 | ~70 | **ESGF-only** — Pangeo lacks daily 3D state for ssp370 |
| ssp585 | ~75 | Pangeo (60) + ESGF backfill (~15) |
| **Total** | **~415** | — |

The multi-member cap retains up to 5 realizations per
`(source_id, experiment, p, f)` label. ESGF-only models (publish daily
3D state on ESGF but not Pangeo) include ACCESS-ESM1-5,
AWI-ESM-1-REcoM, CESM2-WACCM-FV2, FGOALS-f3-L, IPSL-CM6A-LR and
others; the full ssp126 / ssp370 coverage depends entirely on ESGF.

Storage at 338 MB / dataset-year:
- Historical 1940-2014 (75 y): ~3.0 TB
- 4 SSPs 2015-2100 (86 y each): ~8.5 TB
- **Total: ~11.5 TB**

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

  Method is selected on the CMIP6 source name (`pr`, `rsdt`, ...);
  the rename to baseline output names happens at write time, well
  after regrid. The table below lists source-name → output-name where
  relevant.

  | Source variable | Category | Requested | Actual (typical) | Notes |
  |-----------------|----------|-----------|------------------|-------|
  | `ua`, `va`, `hus`, `zg` | core (plev) | bilinear | bilinear | |
  | `tas`, `huss`, `psl` | core (2D) | bilinear | bilinear | renamed `tas`→`TMP2m`, `huss`→`Q2m` |
  | `pr` | core (2D flux) | conservative | conservative | atmos grid has bounds; renamed `pr`→`PRATEsfc` |
  | `rsdt`, `rsut`, `rlut` | optional (TOA, `CFday`/`day`) | conservative | conservative | atmos grid has bounds; renamed to `DSWRFtoa`/`USWRFtoa`/`ULWRFtoa` |
  | `rsutcs`, `rlutcs`, `rsdscs`, `rsuscs`, `rldscs` | optional (clear-sky, `CFday`) | conservative | conservative | renamed `UCSWRFtoa`, `UCLWRFtoa`, `DCSWRFsfc`, `UCSWRFsfc`, `DCLWRFsfc` |
  | `rsds`, `rsus`, `rlds`, `rlus` | optional (sfc rad) | conservative | conservative | atmos grid has bounds; renamed `DSWRFsfc`, `USWRFsfc`, `DLWRFsfc`, `ULWRFsfc` |
  | `hfss`, `hfls` | optional (sfc turb) | conservative | conservative | renamed `SHTFLsfc`, `LHTFLsfc` |
  | `ps` | optional (CFday) | bilinear | bilinear | renamed `PRESsfc` |
  | `ta700`, `wap500`, `clwvi`, `clivi` | optional (CFday diag) | bilinear | bilinear | `ta700`→`TMP700`, others unrenamed |
  | `sfcWind`, `uas`, `vas` | optional (wind) | bilinear | bilinear | `uas`→`UGRD10m`, `vas`→`VGRD10m` |
  | `prw` (Eday) | surface (atmos) | bilinear | bilinear | →`water_vapor_path` |
  | `ts` | surface (Amon/Eday) | bilinear | bilinear | |
  | `siconc` | surface (SImon/SIday) | conservative | **bilinear** | ocean grid; see below. Rescaled %→[0, 1] and renamed `{simon,siday}_sea_ice_fraction` |
  | `sftlf` | static (fx) | conservative | conservative | atmos grid has bounds; rescaled to [0, 1] and renamed `land_fraction` |
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

  **Unstructured ocean grids (FESOM, etc.).** AWI-ESM's FESOM ocean
  grid is published as a 1D `ncells` axis with paired 1D `lat`/`lon`
  coords — there is no rectangular structure for xESMF's
  conservative or bilinear regridders to consume. ``is_unstructured_source``
  in `processing.py` detects this and routes the variable through
  ``nearest_s2d`` with ``locstream_in=True`` (recorded under the
  sentinel method name ``nearest_s2d_locstream``). Because the FESOM
  ocean grid has no land cells, the locstream nearest fills every
  target cell with a valid ocean value; ``apply_target_land_mask``
  then re-applies the target-grid `land_fraction` so the
  NaN-over-land pattern is restored before `emit_mask_and_fill`.

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
    metadata.json                       # sidecar; also serves as the
                                        # "done marker" for resumable
                                        # re-runs
    stats.nc                            # per-dataset multi-period stats
                                        # (written inline by process.py)
  index.csv                             # one row per dataset attempted
  index.parquet                         # ditto, when pyarrow available
  stats.csv                             # cross-dataset tidy aggregate
  stats.parquet                         # ditto, when pyarrow available

<external_forcings_directory>/          # default <output_directory>/external_forcings
  <experiment>.zarr                     # one per scenario
```

One zarr per `(source_id, experiment, variant_label)`. Chunk/shard per
Issue 6 (inner `time=1`, outer `time=365`). One `stats.nc` written
alongside each `data.zarr` containing per-dataset stats over each of
the configured `StatsPeriod`s (see **Multi-period inline stats**).

### Multi-period inline stats

Per-dataset summary statistics are computed **inline** in
`process.py` / `process_esgf.py` right after `write_zarr` — each
per-dataset pod writes its own `stats.nc` next to `data.zarr` while
the dataset is still in memory from the materialize-before-write
step. Stats are computed over one or more named time windows
configured via `defaults.stats_periods` (a tuple of `StatsPeriod`
records in `config.py`). The default set is three periods:

- `full` — the dataset's full time range (always populated).
- `1940-2014` — historical training window (populated on historical
  datasets, all-NaN on pure SSP datasets).
- `1979-2014` — modern reanalysis window aligned with ERA5's
  well-observed era, ending with historical so the stats stay free
  of SSP-scenario drift.

Each per-dataset `stats.nc` carries a `period` dim; stat variables
take the form `{var}__{stat}` with shape `(period,)` or
`(period, plev)`. Periods with no overlap on a given dataset emit
NaN-filled stats.

The standalone `compute_stats.py` is now an **aggregator +
gap-filler**: it scans existing `stats.nc` files, only recomputes
missing ones (or all of them with `--force`), and produces the
cross-dataset tidy aggregate at `<output_directory>/stats.csv` (and
`stats.parquet` when pyarrow is available) with one row per
`(dataset, variable, plev, period)`.

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

1. Drop if more than `max_core_missing` core variables missing.
2. Open each variable's zarr (state from `day`/`CFday`,
   surface-and-ocean from `Amon`/`Eday`/`SImon`/`SIday`/`Omon`/`Oday`,
   static from `fx`). All opens use `chunks={"time": 365}` so the
   subsequent time-subset stays dask-lazy — a correctness fix; without
   chunks, fancy-index time selection can materialise the full
   variable in RAM (~30 GB for ESGF files) and OOM the pod.
3. Validate `cell_methods`.
4. Regrid to F22.5 (Gauss-Legendre 45 x 90) via `xesmf` (bilinear for
   state, conservative for fluxes; streaming `nearest_s2d` for
   unstructured ocean grids like AWI's FESOM).
5. Below-surface nearest-above fill + emit time-varying
   `below_surface_mask(time, plev, lat, lon)` (uint8).
6. Derived `ta_derived_layer_{0..6}` from `zg` + `hus`.
7. Causal-previous-month / -previous-year forcings onto the daily
   axis; attach static fields (`sftlf` → `land_fraction` rescaled to
   [0, 1]).
8. Time-subset per config.
9. Derive `total_water_path` (if `Eday.prw` + `clwvi` both present)
   and `{simon,siday}_ocean_fraction` (with the land+ice>1 excess
   pushed back into sea-ice so the triple sums to 1 exactly).
10. Attach external forcings (`input4mips_*`, `luh2_forest`) from the
    per-scenario zarr under `external_forcings_directory`.
11. Materialize the dataset (xesmf isn't thread-safe; load once
    sequentially).
12. Flatten `plev` dimension into pressure-named 2D variables (see
    **Plev flattening**).
13. Harmonize all temperature variables to K via
    `harmonize_temperature_to_kelvin` (CMIP6 `tos`/`tob`/`sitemptop`
    are spec'd Celsius; converted based on `units` attribute).
14. Apply output renames (`CMIP_TO_OUTPUT_RENAMES`) so `tas`→`TMP2m`,
    radiative fluxes get baseline names, etc.
15. Run sanity checks (advisory; recorded in `warnings`).
16. Write zarr with zarr v3 chunks+shards; drop sidecar `metadata.json`.
17. Compute multi-period stats inline and write `stats.nc` next to
    the zarr.
18. Append a row to the central `index.{csv,parquet}`.

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

Per-dataset summary stats are computed inline by `process.py` /
`process_esgf.py` (see **Multi-period inline stats**). The standalone
`compute_stats.py` aggregates them into the cross-dataset
`stats.csv` / `stats.parquet` and gap-fills any datasets missing an
inline write.

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
- **Issue 3 — Member caps.** `require_i = 1` and `max_members_per_f = 5`
  at ingest, applied per `(source_id, experiment, p, f)` label.
  Deterministic selection by `(variant_f, variant_r)`.
- **Issue 4 — Time subset.** Main training window: `historical`
  1940-01-01 to 2014-12-31 (75 years, aligned with our ERA5 / SHiELD
  AMIP datasets) and all four SSPs 2015-01-01 to 2100-12-31 (86 years
  each, full CMIP6 protocol window). Set `defaults.time_subset: null`
  to expand historical back to its 1850 start.
- **Issue 5 — Pangeo-only vs ESGF.** Both Pangeo and ESGF are used.
  Pangeo provides the bulk of historical and ssp245/ssp585; ESGF is
  the **sole source** for ssp126 / ssp370 (Pangeo lacks daily 3D state
  for those scenarios) and backfills additional models / members for
  the others. `ta` replaced by derived layer-T; `ps` replaced by
  `psl` + topography mask.
- **Issue 8 — Regridding.** `xesmf` targeting Gauss-Legendre F22.5
  (45 x 90). Conservative for fluxes/precip, bilinear for state.
  Weights cached per source grid.
- **Forcings wiring.** Monthly `ts` (`Amon`) + `siconc` (`SImon`)
  causal-previous-month mapped to daily; static `sftlf` (→
  `land_fraction`) + `orog` (`fx`) broadcast. ~21 source_ids have
  full core + forcing + static coverage in Pangeo.
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

- **AWI-CM-1-1-MR non-plev8 pressure grid**: publishes 3D variables
  on a 19-level pressure grid (1, 5, 10, 20, 30, 50, 70, 100, 150,
  200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa) rather
  than the plev8 set the pipeline assumes. The pipeline ingests the
  full 19 levels and produces per-level variables (`ua5`, `ua20`,
  ...) and derived layers (`ta_derived_layer_5_1`, ...) that
  don't match the rest of the cohort. **AWI-CM-1-1-MR is excluded
  from training** via `selection.exclude_source_ids` in both prod
  configs and the v4+ pilot configs.

- **AWI-ESM-1-1-LR and AWI-ESM-1-REcoM corrupted Jan 1 data**:
  every Jan 1 in these models' day-table output is anomalous —
  global-mean psl drops by ~5900 Pa (vs typical 53 Pa day-to-day
  variability, ~110× outlier), TMP2m spikes by ~1.5 K, sfcWind by
  0.4 m/s, DSWRFsfc by ~3 W/m². Surrounding days are normal.
  Pattern is consistent with **each annual file's first day being
  an instantaneous snapshot rather than a daily mean** (the
  highest-magnitude anomalies are in the variables with the
  largest instantaneous-vs-daily-mean gap; precip and humidity are
  unaffected). A CMIP6 publishing bug at the source. Affects
  ~0.27% of timesteps (1 day per 365), but the magnitudes are large
  enough that training samples crossing Jan 1 would see a
  many-sigma input shift. **Both models excluded** via
  `selection.exclude_source_ids`.

- **EC-Earth3/historical/r4i1p1f1 missing plev timesteps**: The last
  13 timesteps (indices 347-359, late December) have all-NaN values
  for all 3D pressure-level variables (ua, va, hus, zg at all levels),
  while surface variables (tas, huss, psl, pr) and derived layer-mean
  temperatures are present. Likely incomplete data upload.
  Other EC-Earth3 members are unaffected. **Excluded** via
  `selection.exclude_variants` in both prod configs and the v4 pilot.

- **HadGEM3-GC31-MM/historical/r2i1p1f3 missing plev timesteps**: 22
  timesteps (indices 40-61, mid-February) have all-NaN values for all
  3D pressure-level variables, while surface variables are present.
  Other HadGEM3-GC31-MM members are unaffected. **Excluded** via
  `selection.exclude_variants`.

- **CESM2-WACCM/ssp585/r1i1p1f1 missing chunk**: 41 consecutive
  timesteps starting 2016-02-15 have all-NaN data across every
  pressure-level variable. Uniform `finite_fraction ≈ 0.944`
  across all plev levels — distinctive signature of a publisher
  missing-chunk rather than a below-surface fill artifact.
  Sibling variants of CESM2-WACCM ssp585 are fine. **Excluded**
  via `selection.exclude_variants`.

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
