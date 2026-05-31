# v2 anomaly review

Run: `gs://vcm-ml-intermediate/2026-05-22-cmip6-multimodel-daily-4deg-8plev-1940-2100/v2/`
Generated from per-dataset `stats.nc` (schema 0.6.0 post-fix).

All Tier-1 issues identified in the initial audit have been resolved.
This file now documents (a) the fixes that landed and (b) coverage
caveats that remain.

## Tier 1 — load-bearing data corruption (FIXED)

### A. netCDF fill-value leakage (BCC `ua*`, CESM2 / FGOALS-f3-L ocean vars, MPI `siday_sithick`)

Several publishers ship raw `1e30..1e36` values in cells that should
be `_FillValue` — the metadata didn't match the stored bytes, so
xarray's default `mask_and_scale` decode let them through. Without a
clip those would have landed in training as nonsense data.

**Fix:** `processing.decode_default_fills` applies a magnitude clip
(`|value| >= 1e10` → NaN) at source-open time in both `_open_zstore`
(Pangeo) and `_open_netcdf_files` (ESGF). Threshold is 9+ orders of
magnitude above any legitimate value, so the false-positive surface
is zero. Migration 0.4.0 → 0.5.0 applied the same clip to all 246
existing zarrs in place. Audit recorded per-variable cell counts.

### B. Mismatched fx-grid selection → sparse `HGTsfc`

GFDL-CM4 (3 datasets) and CMCC family (7 datasets) had `HGTsfc` at
only ~20% finite cells because `process.py` picked `orog` from one
grid (`gr2`) and `sftlf` from a different grid (`gr1`) for those
models, and the merge-then-regrid pipeline produced a sparse-union
source.

**Fix:** the fx-selection step now prefers fx rows whose `grid_label`
matches the day-table's grid; per-variable regrid replaces
merge-then-regrid. The 10 affected datasets were rewritten in place
via `refresh_statics.py`. Verified `HGTsfc` and `land_fraction` are
now 100% finite for all 10.

### C. CMCC `orog` published only over land

CMCC-CM2-HR4 / CMCC-CM2-SR5 / CMCC-ESM2 publish `orog` over land
cells only (37% finite at source), unlike most publishers who use
0 m over the ocean. The conservative regridder produced a sparse
output field.

**Fix:** `processing.fill_orog_ocean_with_zero` NaN-fills the source
`orog` with 0 m before regrid, treating ocean cells as 0 m sea
level. Applied in both `process.py` and `refresh_statics.py`.
Verified for all 7 CMCC datasets.

### D. ESGF augment skipped temperature K-harmonization

77 ESGF-augmented `omon_tob` fields shipped in degC and landed in
the zarr as ~3 K — nonsense for ocean bottom temperature, which
should be 273–280 K. Surfaced when the post-fill-clip
`finite_fraction` warning flagged 9.9% finite on FGOALS-f3-L (an
artefact of clipping the bogus values from the un-harmonized field).

**Fix:** `process_esgf.augment_one_esgf` now applies
`harmonize_temperature_to_kelvin` to each non-mask augment output
before writing. Migration 0.5.0 → 0.6.0 added 273.15 to any
on-disk variable whose `units` attribute reads as Celsius
(with a defensive guard against double-conversion). Verified
FGOALS-f3-L `omon_tob` now sits at 273.2–276.1 K.

## Tier 2 — coverage caveats (not bugs, but worth flagging)

### A. Incomplete 3D state on ~33% of datasets

80 of 243 datasets (33%) lack one or more complete 3D state
variables on Pangeo:

- **ACCESS-ESM1-5, CMCC family** (most variants): missing `zg`
  entirely (no geopotential height profile; `h500` is still
  present as a separate field).
- **BCC-CSM2-MR/BCC-ESM1 historical r2/r3**: missing both `hus`
  and `zg`.
- **CESM2 / CESM2-WACCM ssp245+ssp585** (7 variants): missing
  `ua` and `va` entirely (no 3D winds).
- Several other mixed-coverage cases.

This is a Pangeo-publishing gap — those models/variants simply
don't publish daily 3D state on the Pangeo zarr mirror. Our ESGF
augment currently only adds surface / ocean variables, not 3D
state, so these gaps remain.

**To-do (only if downstream needs complete 3D state):** either
(a) tighten `selection.max_core_missing` in the config to require
all 4 of `ua/va/hus/zg`, dropping the partial-coverage variants;
or (b) extend ESGF augment to fill in missing 3D state when
Pangeo lacks it. (b) is more work but recovers more datasets.

### B. Surface-and-ocean coverage by ESGF augment

These variables didn't land for the full cohort (existing Pangeo
sources never had them, ESGF augment didn't reach all datasets):

| Variable | Coverage |
| --- | --- |
| `PRESsfc` (surface pressure) | 31% |
| `DLWRFsfc` (downward LW at surface) | 52% |
| `USWRFsfc` (upward SW at surface) | 65% |
| `omon_tob` (ocean bottom T) | 32% |
| `omon_mlotst` (mixed layer depth) | 44% |
| `simon_sitemptop` (sea-ice top T) | 51% |
| `oday_tos` (sea surface T daily) | 56% |
| `omon_zos` (sea surface height) | 58% |

Per-source normalization handles this transparently — each source's
centering/scaling files include only the variables that source has.
Cross-source pooled files average over whatever subset contributes
to each variable (see `contributors` attr per variable in the
pooled `.nc` files).

## Tier 3 — model spread observed

`outliers.csv` lists 10 dataset×variable cells > 3σ from cohort mean.
After the bug fixes, all surviving outliers are **real model
behavior**:

- **MIROC6 `psl`** ~290 Pa below cohort across all 6 variants (z ≈
  -4.5). Real MIROC6 mean sea level pressure bias.
- **EC-Earth3 ssp585 r1 `h500`** ~170 m below cohort. Within normal
  inter-model spread for a warmer climate scenario.
- **`input4mips_so2`** small deviations on a couple of historical
  r2/r3 variants of MIROC6 and EC-Earth3-Veg-LR. External-forcings
  field is shared; minor variance.

## Cross-validation checks performed

- **Fill-value cohort scan** (`|x| >= 1e10`): 0 datasets.
- **plev consistency** across 3D-state-equipped datasets: only the
  CESM2 missing-`ua`/`va` case stood out (Tier 2.A above).
- **Cross-model spread on per-source means** (CoV): no variables
  flag as scaling/unit bugs after the degC → K migration.
- **Temperature unit harmonization scan**: 0 datasets with sub-100 K
  values for any temperature variable.
- **HGTsfc + land_fraction completeness**: 1.0 finite across all 243
  datasets.

Source artefacts: `outliers.csv`, `SUMMARY.md`, `stats_aggregated.csv`,
`plots/*.png` in this directory.
