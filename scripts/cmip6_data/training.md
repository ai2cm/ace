# CMIP6 Daily Training: Issues and Plan

This document tracks issues blocking or relevant to running a smoke-test
training run on the CMIP6 daily pressure-level data produced by the
`scripts/cmip6_data/` pipeline (`process.py` for Pangeo, `process_esgf.py`
for ESGF).

## Current status

**Data preparation**: scope expanded to ~415 eligible
(model, experiment, member) datasets across historical 1940-2014 and
all four SSPs (126, 245, 370, 585) 2015-2100. Estimated total
on-disk size ~11.5 TB. Production ingest runs through the argo
workflow's `process-dataset` step against the same code that's been
validated end-to-end locally on CanESM5 historical r1i1p1f1.

Previously deferred items, now implemented:
- ✅ Ocean variables (`Oday` daily SST and others, `Omon` zos/hfds/
  mlotst/tob causal-monthly, `SIday`/`SImon` sea-ice). All routed
  through the surface-and-ocean source-prefixed naming convention
  (`oday_tos`, `omon_zos`, `simon_sea_ice_fraction`,
  `siday_sea_ice_fraction`, …) with per-variable masks and
  horizontal-diffusion fill. Sea-ice fractions are emitted on [0, 1]
  (`siconc` rescaled from CMIP6's %).
- ✅ External forcings: `input4mips_co2` (NOAA Mauna Loa + UoM SSP),
  `input4mips_so2`/`input4mips_bc` (CMIP7-vintage CEDS historical +
  CMIP6 IAMC SSPs), `luh2_forest` (UofMD LUH2 v2 multiple-states).
  Staged once per scenario by `external_forcings.py`; argo
  `stage-externals` template runs this in production. Cross-version
  sharing via the optional `external_forcings_directory` config field
  on `ProcessConfig` / `ESGFProcessConfig`.
- ✅ Causal forcing: monthly→daily mapping replaced by strictly
  causal previous-month assignment; annual→daily via causal
  previous-year. Piecewise constant on the daily axis.
- ✅ Daily SST: `oday_tos` joined in alongside `surface_temperature`
  (renamed from `Eday.ts`) where
  published.
- ✅ Baseline-aligned output names: radiative fluxes, near-surface
  state, and `zg500` are renamed at ingest to the SHIELD/ERA5
  convention (`DSWRFsfc`, `TMP2m`, `Q2m`, `h500`, …) so training
  configs can share variable names across data sources. Each renamed
  variable preserves the CMIP6 source name in an `original_name`
  attribute.
- ✅ Per-dataset multi-period stats: each pod writes its own
  `stats.nc` next to `data.zarr` covering the configured
  `StatsPeriod`s (default: `full`, `1940-2014`, `1979-2014`). The
  standalone `compute_stats.py` aggregates / gap-fills these into the
  cross-dataset `stats.csv`.

Remaining work for the actual training run:
- ESGF inventory build + production ingest (argo workflow).
- `make_normalization.py` rerun against the v0 zarrs to produce the
  centering/scaling .nc files for the renamed variable set.
- Decide on heterogeneous-variable training config: which variables
  the smoke-test stepper sees and predicts, and which are mask-only
  inputs.

---

## Blocking issues

These must be resolved before a smoke-test run can start.

### ~~1. Data format: mixed dimensionality in zarr output~~ (resolved)

Resolved: `_flatten_plev_variables` in `process.py` splits 3D variables
into pressure-named 2D variables before writing: `ua1000` through
`ua10`, `below_surface_mask1000` through `below_surface_mask10`, and
`ta_derived_layer_1000_850` through `ta_derived_layer_50_10`. All
variables are now uniformly `(time, lat, lon)` or `(lat, lon)`.

### ~~2. Time-varying below-surface mask~~ (deferred)

Not blocking the smoke test. The processed data includes nearest-above
filled values in below-surface cells, so all variables are continuous
and NaN-free. The smoke test trains on the filled data using the
standard `SingleModuleStepConfig`, omitting mask variables from
`in_names`/`out_names`. The model wastes some capacity learning
extrapolated below-surface values, but the pipeline is valid end-to-end.

See issue 13 for the full mask-aware training plan.

### ~~3. Atmosphere corrector depends on hybrid sigma-pressure coordinates~~ (non-issue)

Not blocking. When no `HybridSigmaPressureCoordinate` is configured,
`DatasetInfo.atmosphere_vertical_coordinate` returns `None`
(`fme/core/dataset_info.py:193`). The corrector features that require
vertical integrals — `conserve_dry_air`, `moisture_budget_correction`,
`total_energy_budget_correction` — each check for `None` and raise a
clear `ValueError` if enabled without a coordinate. All three default
to off. The features that *don't* need a coordinate
(`force_positive_names`, `zero_global_mean_moisture_advection`) work
as-is.

For pressure-level data, vertical integrals are not physically
meaningful (levels, not layers with mass), so the disabled features
are not just a workaround — they are correctly inapplicable.

### ~~4. Normalization statistics~~ (resolved)

Resolved: `make_normalization.py` reads the processed zarrs directly,
selects the first ensemble member per source model, concatenates both
experiments in time, and computes area-weighted global stats. Models
are averaged with equal weight. Outputs:

- `centering.nc` / `scaling.nc` — full-field global mean and std
- `residual_centering.nc` / `residual_scaling.nc` — one-step-
  difference mean and std
- `time_mean_map.nc` — per-variable time-mean spatial field (lat, lon)

All files are scalar-per-variable netCDFs compatible with fme's
`NormalizationConfig(global_means_path=..., global_stds_path=...)`.

### ~~5. Training configuration complexity~~ (resolved)

Resolved: `Cmip6DataConfig` (`fme/ace/data_loading/cmip6.py`) reads
`index.csv` from a data directory, filters by status/source_id/
experiment/realization, and programmatically builds a
`ConcatDatasetConfig` with per-dataset `XarrayDataConfig` entries and
labels.  Added to the `DataLoaderConfig.dataset` union type so it can
be specified directly in training YAML:

```yaml
dataset:
  data_dir: /path/to/cmip6-daily-pilot/v0
  source_ids: [ACCESS-CM2, CESM2]   # optional; null = all
  experiments: [historical, ssp585]
  realizations: [1, 2]              # optional; null = all
```

---

## Important issues

These need workarounds or tracking; not strictly blocking a smoke test with
reduced scope, but should be addressed soon.

### ~~6. Model embedding architecture~~ (resolved)

Resolved on main by PR #1148 ("Add configurable label embedding
dimension to NoiseConditionedSFNO"). `NoiseConditionedSFNOBuilder`
now takes a `label_embed_dim` parameter; when `> 0` a learned
`nn.Linear(n_labels, label_embed_dim)` projects the one-hot labels
into a fixed-width embedding before downstream conditioning, so
embedding capacity is independent of the number of model labels.

### ~~7. Different variables available per model~~ (resolved)

Resolved on main by PR #1160 ("Add per-sample variable masking for
heterogeneous data sources"). `ModuleSelector.allow_variable_masking`
threads a `data_mask` through `BatchData` / `StepArgs` / the loss
chain; missing input variables are zeroed in normalized space and
missing output variables are excluded from loss computation.
`Cmip6DataConfig` and the surface-and-ocean variable schema both
rely on this for per-dataset variable presence.

### 13. Mask-aware training for cells with valid/invalid extents (deferred)

**Current state.** The pipeline already emits per-variable mask
channels for the surface-and-ocean variables that have spatial NaN
extents (e.g. `siday_sea_ice_fraction_mask`, `oday_tos_mask` — 2D
when static, 3D when time-varying), and the `below_surface_mask{hPa}`
channels for the plev variables. The 3D plev variables themselves are
filled with
nearest-above-in-column values; the surface-and-ocean variables are
filled via horizontal diffusion. Both fill schemes give the network
a well-behaved field while the corresponding mask channel preserves
the valid extent.

For the current pilot we **predict** the filled fields directly with
a standard regression loss (no BCE on masks, no NaN in targets).
This is intentional — the smoothing fills are physically reasonable
and the model can learn the filled-region values without a separate
mask loss.

**Deferred generalization.** Add per-variable mask channels for *all*
variables that have NaN regions, not just the surface ones — most
notably the plev `ua` / `va` / `hus` / `zg` would get their own
`{var}{hPa}_mask` per pressure level instead of relying on the
shared `below_surface_mask{hPa}` (whose mask source comes from a
fallback chain — `nan_union` across the 3D vars first, then `zg <
orog`). Storing one mask per variable gives downstream training the
option of a masked regression loss or a BCE mask-loss later without
re-ingesting. Not blocking the production training run.

### 8. No surface pressure variable

**Problem.** The CMIP6 daily data does not include surface pressure (`ps`) —
no CMIP6 model publishes it as a daily variable. The dataset uses `psl`
(mean sea-level pressure) as a proxy. Several existing fme components
reference surface pressure conceptually:

- `HybridSigmaPressureCoordinate.interface_pressure(surface_pressure)` uses
  `ps` to compute layer pressures.
- The corrector's dry-air conservation reconstructs `ps` from the water
  content and `ak`/`bk`.

With constant pressure levels, the dependency on `ps` is removed — layer
pressures are constants. But the model does not have access to the actual
surface pressure as a variable, which limits what physical constraints can
be applied.

**Smoke-test impact.** Not blocking. The corrector features that need `ps`
are already disabled (issue 3). `psl` is included as a prognostic variable.

### ~~14. Causal forcing for monthly boundary conditions~~ (resolved)

Resolved in the pipeline rather than in the loader. ``processing.py``
now provides ``causal_monthly_to_daily`` (each day reads its
calendar-previous-month mean, piecewise constant) and
``causal_annual_to_daily`` (each day reads its calendar-previous-year
value). The surface-and-ocean variable handler (`Amon`/`SImon`/`Omon`
monthly source variables) and the external-forcings handler
(`amon_ts` / `simon_sea_ice_fraction` etc. + annual CO2 + LUH2 forest) both
route through these. The legacy linear ``interp_monthly_to_daily``
function was deleted; no code path can produce the non-causal
interpolation anymore. Where daily-cadence sources exist (`Eday`,
`Oday`, `SIday`) those are used directly via nearest-neighbour time
alignment.

### ~~15. External forcings (CO2, aerosol emissions, land use)~~ (resolved)

All four planned external forcings are now staged and joined into
every per-model dataset:

- ``input4mips_co2`` — annual global-mean CO2 (ppm). NOAA Mauna Loa
  annual mean (≥1959) for historical; UoM input4MIPs (Meinshausen et
  al. 2017) for SSPs 2015 onwards.
- ``input4mips_so2`` / ``input4mips_bc`` — gridded monthly anthropogenic
  emission flux (kg m⁻² s⁻¹). CMIP7-vintage CEDS-CMIP-2025-04-18 for
  historical (the CMIP6-vintage CEDS-2017-05-18 was retracted from
  ESGF); CMIP6 IAMC scenario files for SSPs.
- ``luh2_forest`` — annual gridded total forest fraction (primf +
  secdf, [0, 1]). LUH2 v2 multiple-states files: UofMD-landState-2-1-h
  for historical, UofMD-landState-{IMAGE,MESSAGE,AIM,MAGPIE}-* for
  ssp126/245/370/585.

Staging script: ``external_forcings.py``. Pipeline integration:
``attach_external_forcings`` opens the per-scenario zarr at
``<external_forcings_directory>/<experiment>.zarr`` and projects each
forcing onto the daily time axis via the appropriate causal helper.
``external_forcings_directory`` defaults to
``<output_directory>/external_forcings/`` (legacy behavior) but can be
set explicitly so several versioned output directories share one
staged copy. Argo workflow ships a ``stage-externals`` template that
runs the staging once at the start of a production run.

---

## Lower-priority / future issues

### 9. Calendar diversity

The CMIP6 models use different calendars (`noleap`, `360_day`,
`proleptic_gregorian`). The fme data loader handles calendar types
transparently via xarray/cftime, so this is not blocking. However,
concatenating datasets with different calendars means the effective
number of timesteps per year varies (360 vs 365 vs 366). The training
loop assumes uniform `dt` within each dataset but does not require
uniform `dt` across datasets in a `ConcatDataset`. This should work
but warrants validation during the smoke test.

### 10. Vertical coordinate class for constant pressure levels

As noted in issue 3, a `ConstantPressureLevelCoordinate` would allow
the corrector and other components to reason about pressure-level data
natively. This is not needed for the smoke test (the corrector is
pared down) but is needed for production runs with physical constraints.

### 11. Residual normalization for pressure-level data

The normalization investigation (`normalization/README.md`) proposes
shared per-variable scales pooled across all datasets. For residual
(one-step-difference) normalization the `d1_std` values should be used.
The stratospheric `hus` levels (plev indices 5–7) show 40–55% inter-model
dispersion in `d1_std`, which may require per-model or log-space
normalization. For the smoke test, shared linear normalization is
sufficient; refine after examining training loss by variable.

**Update (2026-05-05).** Per-label validation confirmed that `hus10`,
`hus50`, and `hus100` dominate training loss for roughly half the source
models, accounting for ~95% of those models' excess loss. The root cause
is that models whose stratospheric humidity values are far from the
cross-model centering constant arrive as outlier inputs in network-
normalized space, producing large prediction errors that are further
amplified by the small `residual_scaling` denominators (~5.6e-8 for
`hus10`). This is an interaction between the cross-model normalization
and model-specific stratospheric humidity climatology, not a network
architecture issue per se. Options to address: per-model normalization
for upper-stratospheric humidity, log-space normalization, down-weighting
in the loss, or excluding the uppermost humidity levels.

### 12. INM-CM4-8 exclusion and other data-quality flags

INM-CM4-8 is already excluded from training via `pilot.yaml`
(`selection.exclude_source_ids`). The CESM2-FV2 `sftlf` overshoot at the
south pole and CESM2-WACCM deduplication are handled at ingest.
`Cmip6DataConfig` (issue 5) filters to `status == "ok"` datasets
automatically.

**Update (2026-05-05).** IITM-ESM excluded from training
(`exclude_source_ids` in `train-smoke-test.yaml`). Per-label validation
showed IITM-ESM with a mean loss of 23.3 — 100x worse than well-behaved
models — with negligible variance, indicating consistently wrong
predictions rather than occasional spikes. Root cause: IITM-ESM reports
**negative specific humidity** at 10 and 50 hPa (mean hus10 = −2.8e-6
vs cross-model centering of +2.4e-6). Negative specific humidity is
physically impossible and places this model ~19σ from the multi-model
mean in network-normalized space. This is a data quality issue in the
source model output, not a pipeline bug.
