# CMIP6 daily pilot report — `v0-pilot`

**Bucket:** `gs://vcm-ml-intermediate/2026-05-22-cmip6-multimodel-daily-4deg-8plev-1940-2100/v0-pilot/`
**Window:** 1 year per dataset — historical = 2010, ssp* = 2015
**Target grid:** F22.5 Gauss–Legendre 45 × 90
**Successful datasets:** 294 (all `status="ok"`)

This report aggregates per-dataset `metadata.json` sidecars and
`stats.nc` files from the pilot. The headline finding is that the
pipeline works end-to-end across 43 source models, but several
issues are visible in the artifacts that could bite a downstream
training run.

**Status note (2026-05-25).** This report is on the `v0-pilot`
artifacts, generated months ago before several pipeline fixes
landed. Items marked **(fixed)** below have been addressed in the
current code; verification will come from the next pilot
(`v0-pilot-v4+`) which uses prod-style 365-day chunking. Items
marked **(action taken)** are configuration changes (variant
exclusions, etc.) that prevent the issue going forward.

Source artefacts: see [`outliers.csv`](outliers.csv) and
[`stats_aggregated.csv`](stats_aggregated.csv). Plots are in
[`plots/`](plots/) and referenced inline below.

---

## 1. Coverage

![coverage](plots/coverage_matrix.png)

- 43 source models × 5 experiments, with anywhere from 1 to 6
  variants per (model, experiment) cell.
- `historical` is well-populated (90 datasets); the SSP scenarios
  are sparser — `ssp126` and `ssp370` in particular have only ~50
  datasets each because daily 3D state isn't published for many
  models under those scenarios.
- 5 calendar conventions present (`noleap` 107, `proleptic_gregorian`
  93, `standard` 55, `360_day` 36, `julian` 3). `360_day` datasets
  have `n_timesteps = 360`, the rest have `365`.

## 2. Variable presence

![variable_presence](plots/variable_presence.png)

The external-forcing variables (`luh2_forest`, `input4mips_co2/so2/bc`)
are universal — they're attached per-experiment so every dataset
gets them.

The 3D variables behave per-family:

- **Surface met** (`tas`, `huss`, `uas`, `vas`, `pr`, `psl`): present
  in 270–290 of 294 datasets.
- **8-level pressure variables** (`ua/va/hus/zg` × 8 plev): present
  in ~235–260 each. Some models don't publish 3D `Amon` state at the
  pilot's plev set, but the pipeline tolerates up to 3 missing core
  variables (`max_core_missing = 3`) so the dataset still produces
  surface + radiation + externals.
- **Derived layer-mean T** (`ta_derived_layer_*`): present where
  both `zg` and `hus` are published — 197 of 294 datasets.
- **Ocean / sea-ice** (`oday_tos`, `simon_siconc`, `simon_sitemptop`,
  `omon_*`): present in 110–180 datasets each. The rest fail to
  regrid (see §3); a published variable that fails to regrid is
  silently absent in the output, not a hard error.
- **`below_surface_mask*`** appear at variable plev pressures
  (10/50/100/250/500/700/850/1000 hPa) — these are the masks
  for the bottom-bounded levels.

## 3. Pipeline warnings

![warnings](plots/warnings.png)

Top warning categories tell us what's commonly partially-failing:

| Warning prefix                                      | # datasets |
|-----------------------------------------------------|-----------:|
| `oday_tos from Oday.tos failed`                     |        155 |
| `omon_zos from Omon.zos failed`                     |        155 |
| `omon_hfds from Omon.hfds failed`                   |        121 |
| `omon_mlotst from Omon.mlotst failed`               |        108 |
| `simon_siconc from SImon.siconc failed`             |        106 |
| `simon_sitemptop from SImon.sitemptop failed`       |        104 |
| `core variables absent (tolerated)`                 |        118 |
| `<var> sanity: out of range` (per-variable)         |     ~60×N  |

The ocean / sea-ice regrid failures are nearly all
`ESMC_FieldRegridStore failed with rc = 506`, which is xesmf
giving up on curvilinear / tripolar grids when bounds metadata
isn't quite what conservative needs. The result is no ocean
data for those (model, experiment) pairs.

**Implication for training:** every variable in the
surface-and-ocean and 3D-state families is missing for a
substantial fraction of datasets. The training stack needs to
tolerate per-sample variable masking — losing ~30–50% of all
samples for any one ocean variable is too aggressive a filter.

## 4. Below-surface mask + input-NaN distribution

![mask_coverage](plots/mask_coverage.png)

- 144 datasets derive the below-surface mask from the NaN union
  across (ua, va, hus, zg) — publishers that emit NaN under
  topography.
- 99 datasets fall back to `orog_static` because no 3D NaNs were
  observed; the mask comes from `zg < orog`.
- 51 datasets have **no mask** (`mask_source = "none"`) — these
  are datasets without 3D state at all, so there's no notion of
  below-surface.

`n_nan_input_cells` is the count of NaN cells in the regridded 3D
state *before* nearest-above fill. Most datasets have 0 — model
publishers either fill below-surface or don't, and the regrid
preserves the pattern.

Two outliers stand out:

- **HadGEM3-GC31-MM historical r2i1p1f3** — 2,851,200 NaN cells
- **EC-Earth3 historical r4i1p1f1** — 1,684,800 NaN cells

These are full pressure-level slabs of NaN. Both publishers
appear to write whole `(plev, lat, lon)` slabs of NaN for
historical near-surface cells, presumably because their model
doesn't carry a meaningful value below the lowest valid level.
The fill should handle these but the input volume is unusual —
worth keeping an eye on at prod scale.

## 5. Cross-model agreement

![crossmodel_stats](plots/crossmodel_stats.png)

For 16 representative variables, the panel shows per-dataset
(mean, std) coloured by experiment. A few notes:

- Tight cohort for `tas`, `psl`, `zg500`, `luh2_forest`,
  `input4mips_*` (essentially zero CV — the external forcings
  are the *same* file per experiment, so the spread across models
  is a sanity check that all paths read the same source).
- Wider for `pr`, `huss`, `rsds`, `rlut` — expected; precip and
  radiation vary substantially across models.

Quantitative table of per-variable spread (historical only):

| variable          | n  | cohort mean | std       | CV     |
|-------------------|---:|------------:|----------:|-------:|
| `tas`             | 88 | 287.9 K     | 0.57 K    | 0.2%   |
| `amon_ts`         | 90 | 288.7 K     | 0.57 K    | 0.2%   |
| `psl`             | 84 | 101 126 Pa  | 70.5 Pa   | 0.07%  |
| `zg500`           | 70 | 5 659 m     | 20.6 m    | 0.4%   |
| `rlut`            | 71 | 239.3 W/m²  | 2.4 W/m²  | 1.0%   |
| `rsds`            | 74 | 189.2 W/m²  | 4.4 W/m²  | 2.3%   |
| `sfcWind`         | 84 | 6.27 m/s    | 0.21 m/s  | 3.3%   |
| `pr`              | 89 | 3.44e-5     | 1.37e-6   | 4.0%   |
| `ua250`           | 88 | 15.07 m/s   | 0.87 m/s  | 5.7%   |
| `huss`            | 78 | 0.01015     | 6.7e-4    | 6.6%   |
| `luh2_forest`     | 90 | 0.0710      | ≈ 0       | 0%     |
| `input4mips_co2`  | 90 | 387.6 ppm   | ≈ 0       | 0%     |
| `input4mips_so2`  | 90 | 6.56e-12    | ≈ 0       | 0%     |
| `sftlf`           | 79 | 28.71       | 3.29      | **11.5%** |

That last row is the big red flag — see §6.

## 6. ⚠️ Things that could hurt training

### a) `sftlf` is on inconsistent scales **(fixed)**

`sftlf` (land area fraction) showed up in two scales in the pilot:
- 207 datasets had `sftlf` in **percent** (mean ≈ 29, max ≈ 100).
- 1 dataset (FGOALS-f3-L historical r1i1p1f1) had `sftlf` in
  **fraction** (mean = 0.29, max = 1.0).
- 41 datasets *also* exposed `land_fraction` (the post-`clamp_static_fractions`
  renamed/rescaled-to-0..1 variant), all with mean ≈ 0.29.

The clamp step exists in both `process.py` and `process_esgf.py`
(clip → rescale to 0..1 → drop `sftlf` → assign `land_fraction`)
and runs unconditionally on every static set. The pilot's mixed
state was an artefact of being assembled across multiple code
iterations — current code drops `sftlf` cleanly. **Verify on the
v0-pilot-v4 output**: only `land_fraction` (0–1) should be
present, no `sftlf` (0–100).

### b) MIROC6 is a systematic outlier (real, not a bug)

[`outliers.csv`](outliers.csv) shows all 18 of the >3σ outliers
are MIROC6: `psl` is ~280 Pa lower than the cohort mean and
`rlut` is ~11 W/m² lower. This is real model bias, not a
processing artefact — MIROC6 just has a colder, lower-pressure
climatology than the cohort. Worth knowing because the loss
landscape will see MIROC6 as a small but consistent shift.

Suggested mitigations:
- Per-source-id mean/std normalization rather than a single
  cohort normalization (already supported via per-source
  normalization mode added in `Cmip6Step`).
- Or weight MIROC6 samples lower if model diversity matters less
  than cohort consistency.

### c) Heavy NaN-input datasets **(action taken)**

`HadGEM3-GC31-MM historical r2i1p1f3` and `EC-Earth3 historical
r4i1p1f1` have ~10⁶ NaN cells in their input 3D state pre-fill —
full pressure-level slabs (8 plev × 365 days × ~970 cells per
slab). Sibling variants of both models are clean, so the issue is
isolated to those specific publisher files.

**Both are now excluded** via the new
`selection.exclude_variants` schema added to `Selection`. The v4
pilot configs and the prod configs all list these two
`(source_id, experiment, variant_label)` triples. Healthy
sibling variants stay in the cohort.

### d) Ocean / sea-ice gaps are widespread **(fixed — see commit `99acb08d2`)**

~50–60% of pilot datasets lost at least one ocean variable.
Investigation showed the failures were not the ESMF
`rc=506` error we initially thought — they were a `cf_xarray.bounds_to_vertices`
chunking bug: that helper uses `apply_ufunc` with the 4-vertex
dim as a core dim and refuses to run when that dim is chunked.
CMIP6 ocean zarrs ship with `vertices` pre-chunked, breaking
~150 of 294 pilot datasets with the message:

```
ValueError: dimension vertices on 0th function argument to
apply_ufunc with dask='parallelized' consists of multiple
chunks, but is also a core dimension.
```

Fix in `normalize_regrid_source`: `.load()` on the bounds
DataArray before passing to `bounds_to_vertices`. Bounds are
tiny (lat × lon × 4 × 8 bytes ≈ MB) so loading is cheap.
Verified locally on ACCESS-CM2 and CanESM5 Oday.tos sources
that both build conservative regridders. The v4 pilot should
recover most of the missing ocean coverage.

A separate root cause was found in regridding masked variables
(`simon_sitemptop`, `oday_tos` near coastlines): xesmf's default
weighted average treats source NaN cells as 0, polluting target
cells that partially cover NaN regions. Fixed via `skipna=True`
in commit `353e385c7`.

### e) Tight variance on surface T means models disagree little

`tas` has cohort std of 0.57 K. That's smaller than the
ERA5-vs-truth diurnal uncertainty alone, which tells you that all
the historical CMIP6 cohorts roughly agree on the 2010 global
mean. This is good news for stability of training, but means
**training on this cohort doesn't teach a model to handle
plausible inter-model variability** — it just teaches the
ensemble centre. If you want robustness to model biases, you
need either (i) the SSP scenarios mixed in (they have larger
inter-model spread by 2015), or (ii) more recent / more divergent
data.

### f) Finite-cell fraction

![finite_fraction](plots/finite_fraction.png)

Most variables are >99% finite. The mask channels (`*_mask`) sit
where you'd expect (ocean masks finite where ocean is published).
A few variables sit between 0.5 and 0.9 — those are the ocean
variables that survived regrid (sample is the regridded version,
with land NaNs preserved before fill). No variable has an
*unexpected* NaN signature.

### g) Temperature units **(was wrong — now fixed in current code)**

![temperature_units](plots/temperature_units.png)

The plot's at-a-glance read suggested all temperature variables
were in K, but a deeper look at the stats found two pilot-only
issues:

- **`oday_tos`**: 101 of 103 pilot datasets have `units="degC"`
  with means 13–18 °C (un-converted SST). The
  `harmonize_temperature_to_kelvin` step is in the current
  pipeline (commit `4ed9fe7dd`) and converts via the source
  `units` attribute, but the v0-pilot artefacts predate that
  commit.
- **`simon_sitemptop`**: NorESM2-LM and NorESM2-MM show
  cohort-mean ~88 K even with the ice mask applied — well below
  physical sea-ice top temperatures (~243–273 K). Root cause is
  not a units bug but a regrid bug: xesmf's default weighted
  average mixes source NaN cells (no ice) as 0 K into target
  cells that partially cover ice. Fixed in commit `353e385c7`
  (`skipna=True`).

Sibling **`simon_siconc`** issue: all 111 pilot datasets have
the variable in **percent** (0–100, mean ≈ 10), not fraction
(0–1). The `unit_scale=0.01` rescale is in the current code
(commit `7fc609e42`), but again the pilot artefacts predate it.

All three should resolve in `v0-pilot-v4`; regenerate this
section against the new artefacts to verify.

## 7. Recommendations for the 1940–2100 production run

The pilot validates the end-to-end pipeline on a single year per
dataset. Going to 75–86 years per dataset multiplies a few of the
concerns above:

1. **`sftlf` vs `land_fraction` — verify only `land_fraction` lands
   on disk.** The pilot's mixed state should not recur in prod
   because `clamp_static_fractions` runs unconditionally in both
   `process.py` and `process_esgf.py` now, but spot-check after
   the first finished prod datasets.
2. **Ocean regrid failure rate.** ~50% of datasets lose ocean
   variables in the pilot. At prod scale the absolute count of
   missing-ocean datasets will be similar; if downstream training
   wants ocean coverage above some threshold, plan to filter the
   `variant_labels` rather than expecting ESMF to fix itself.
3. **NaN-heavy publishers** (HadGEM3-GC31-MM,
   EC-Earth3 r4) — flag these for a manual look before training.
4. **MIROC6 bias** — per-source normalization (already in place)
   handles this cleanly; just confirm the prod training config
   uses per-source rather than cohort normalization for variables
   like `psl` and `rlut`.
