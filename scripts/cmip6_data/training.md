# CMIP6 Daily Training: Issues and Plan

This document tracks issues blocking or relevant to running a smoke-test
training run on the CMIP6 daily pressure-level data produced by
`scripts/cmip6_data/process.py`.

---

## Blocking issues

These must be resolved before a smoke-test run can start.

### ~~1. Data format: mixed dimensionality in zarr output~~ (resolved)

Resolved: `_flatten_plev_variables` in `process.py` splits 3D variables
into pressure-named 2D variables before writing: `ua1000` through
`ua10`, `below_surface_mask1000` through `below_surface_mask10`, and
`ta_derived_layer_1000_850` through `ta_derived_layer_50_10`. All
variables are now uniformly `(time, lat, lon)` or `(lat, lon)`.

### 2. Time-varying below-surface mask

**Problem.** The CMIP6 data has a time-varying below-surface mask that
varies over time (surface pressure changes move the boundary). After
flattening (issue 1), this becomes per-level variables like
`below_surface_mask1000(time, lat, lon)`. But fme's `MaskProvider`
(`fme/core/mask_provider.py:67`) only supports time-invariant masks — it loads
masks once from the first timestep and broadcasts them.

**Approach.** Use a CMIP6-specific step object (or corrector) that consumes
the per-level `below_surface_mask{hPa}` variables as ordinary time-varying data
variables rather than relying on fme's `MaskProvider`. This step would apply
the mask at each timestep during the forward pass, correctly tracking the
moving below-surface boundary. The mask variables are included in the
training data as forcing inputs (in `in_names` but not `out_names`).

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

### 5. Training configuration complexity

**Problem.** A training run on this data requires listing ~76 individual zarr
paths (one per model/experiment/member), each with its own label, inside a
`ConcatDatasetConfig`. Writing this out manually in YAML is error-prone
and huge.

**Approach.** Create a lightweight configuration frontend
(e.g. `Cmip6DatasetConfig`) that takes the `index.csv` path and
output directory, reads the index, and programmatically builds the
`ConcatDatasetConfig` with per-dataset `XarrayDataConfig` entries and labels.
This keeps the user-facing YAML small while the expansion happens at
config-load time.

---

## Important issues

These need workarounds or tracking; not strictly blocking a smoke test with
reduced scope, but should be addressed soon.

### 6. Model embedding architecture

**Problem.** The current embedding sets `embed_dim_labels = len(all_labels)`
(`fme/ace/registry/stochastic_sfno.py:312`). Each label gets its own
dimension in the one-hot vector, which is passed directly to
`ConditionalLayerNorm` and the positional `label_pos_embed` interaction. With
~20 models (and ~57 physics-distinct labels at full scale), this means
57 independent parameters per conditioning site — the embedding dimension
grows linearly with the number of models and there is no shared structure.

**Desired change.** Replace the one-hot scheme with a fixed-width `n`-dim
embedding:

1. Keep the one-hot encoding in `LabelEncoding` (it already handles
   dynamic label sets well).
2. Add a learned `nn.Linear(n_labels, embed_dim)` projection inside
   `NoiseConditionedModel` and `ConditionalLayerNorm` that maps from
   one-hot space to a compact embedding space.
3. `embed_dim_labels` in the `ContextConfig` becomes the embedding
   dimension (e.g. 8 or 16), independent of the number of models.
4. The `label_pos_embed` parameter in `NoiseConditionedModel` changes shape
   from `(n_labels, embed_dim_pos, H, W)` to
   `(embed_dim_labels, embed_dim_pos, H, W)`, with the one-hot-to-embedding
   projection applied before the einsum.

This decouples model capacity from model count and allows the embedding
space to learn shared structure across similar models.

**Smoke-test workaround.** The current one-hot embedding works as-is for
~20 labels. It's suboptimal but not blocking.

### 7. Different variables available per model

**Problem.** Optional variables (TOA/surface radiation, turbulent fluxes,
surface wind) and forcings (`ts`, `siconc`) are not available for every model.
The `DatasetProperties.update()` method
(`fme/core/dataset/properties.py:55–73`) enforces that concatenated datasets
have consistent metadata, including variable sets. Training on the union of
all variables would fail for models missing some.

**Smoke-test workaround.** Train only on the intersection of variables
available across all (non-excluded) models — the core variables (e.g.
`ua1000`–`ua10`, `va1000`–`va10`, `hus1000`–`hus10`, `zg1000`–`zg10`,
`tas`, `huss`, `psl`, `pr`) plus derived layers
(`ta_derived_layer_1000_850`–`ta_derived_layer_50_10`). Do not include
optional variables.

**Future.** Support heterogeneous variable sets across datasets in the
concat loader. One approach: fill missing optional variables with a
configurable default (e.g. zero or NaN + mask) so that all datasets
present the same variable interface to the model. Another: per-variable
presence flags in the batch so the model and loss can handle missingness.

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

### 12. INM-CM4-8 exclusion and other data-quality flags

INM-CM4-8 is already excluded from training via `pilot.yaml`
(`selection.exclude_source_ids`). The CESM2-FV2 `sftlf` overshoot at the
south pole and CESM2-WACCM deduplication are handled at ingest. No
additional training-side filtering is needed, but the config frontend
(issue 5) should respect the index `status` column and only include
`"ok"` datasets.
