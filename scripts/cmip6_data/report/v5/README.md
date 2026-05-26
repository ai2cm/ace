# CMIP6 daily v5 pilot — partial report

**Bucket:** `gs://vcm-ml-intermediate/2026-05-22-cmip6-multimodel-daily-4deg-8plev-1940-2100/v0-pilot-v5/`
**Window:** 2 years per dataset (historical 2010-2011, ssp* 2015-2016)
**Chunking:** prod-style `chunk_time=365`, `variable_batch_size=8`
**Datasets reported:** 350 successful (cmip6-daily-pilot-bfzkw, mid-run; 6 ESGF-stuck pods still pending)

This is a **partial** report generated while bfzkw still has ~6 ESGF-stuck pods running. The plots and stats below are computed against the 350 datasets that landed sidecars before the snapshot.

## What's fixed since v0-pilot

Every issue I flagged in the v0-pilot report has been addressed in code and is verified in v5 on-disk artefacts.

| v0-pilot issue | Status in v5 |
|---|---|
| `sftlf` on inconsistent scales (some 0–100 %, some 0–1) | **Fixed.** Only `land_fraction` (0–1) is present; `sftlf` is dropped at write time. |
| `oday_tos` in °C for 101/103 datasets | **Fixed.** `oday_tos` cohort mean = 291.1 K (~18 °C SST in K). |
| `simon_siconc` in percent (mean 10, max 100) | **Fixed.** Renamed to `simon_sea_ice_fraction` and rescaled to 0–1 (mean 0.039, max 0.089). |
| `simon_sitemptop` cohort mean = 88 K (NaN-as-zero contamination) | **Fixed.** Cohort mean now 269.1 K (~–4 °C, plausible sea-ice top temp). Root cause was xesmf's default `skipna=False`; now `skipna=True`. |
| Ocean regrid 50–60% failure rate (looked like ESMF `rc=506`, really `bounds_to_vertices` chunking bug) | **Fixed.** `normalize_regrid_source` now loads bounds eagerly before `bounds_to_vertices`. |
| HadGEM3-GC31-MM hist r2i1p1f3 and EC-Earth3 hist r4i1p1f1 NaN-slab variants | **Excluded** via `exclude_variants`. |
| AWI-CM-1-1-MR 19-level plev confusion | **Excluded** via `exclude_source_ids` (both Pangeo and ESGF configs). |

Per the partial cohort:

![coverage](plots/coverage_matrix.png)

The coverage map looks similar to v0-pilot but with **more realizations per cell** (max_members_per_f bumped 3 → 5), e.g. CanESM5 now has up to 6 historical members, HadGEM3-GC31-LL has more ssp variants.

## ⚠️ New issues to flag

### a) FGOALS-f3-L `land_fraction` is double-scaled

`land_fraction` cohort mean is 0.29 across 115 datasets — physically correct (global mean land area fraction). FGOALS-f3-L historical r1i1p1f1 shows **0.003** (z = -16.8 from cohort mean). 100× too small.

Root cause: `clamp_static_fractions` in processing.py unconditionally clips `sftlf` to [0, 100] and divides by 100 to get `land_fraction`. FGOALS-f3-L publishes `sftlf` natively in **0–1 fraction units** (the rest of the cohort publishes 0–100 %). Our clamp doesn't sniff the input scale, so we divide a fraction by 100 and get ≈ 0.003.

Fix proposal: in `clamp_static_fractions`, check the input `sftlf` max — if `max ≤ 1.0`, treat as already-fraction and don't rescale. Otherwise rescale by 100. (Or read the `units` attribute.) Either way, this dataset's stored `land_fraction` is wrong and needs reprocessing after the fix.

### b) MIROC6 outlier pattern persists

[`outliers.csv`](outliers.csv): 27 of 29 outliers are MIROC6 — same systematic ~280 Pa lower `psl` and ~11 W/m² lower `ULWRFtoa` ("ULWRFtoa" is the renamed `rlut`). Real model bias, not pipeline. Per-source normalization (already in `Cmip6Step`) handles it.

### c) AWI-ESM Jan 1 publisher artifact (now excluded)

`AWI-ESM-1-1-LR` and `AWI-ESM-1-REcoM` ship corrupted Jan 1 data
every year: psl drops by ~5900 Pa (vs typical 53 Pa day-to-day,
**~110× outlier**), TMP2m spikes by 1.5 K, sfcWind by 0.4 m/s,
DSWRFsfc by ~3 W/m². PRATEsfc and Q2m are unaffected. Pattern is
consistent with each annual file's first day being an
instantaneous snapshot rather than a daily mean — variables whose
instantaneous values differ most from their daily means (slow-
varying psl, fast-varying surface met) show the biggest jump;
stochastic-mean variables don't budge.

Surfaced via the existing `_GLOBAL_MEAN_JUMP_TOL` sanity check
in `processing.py`, which flagged all 5 AWI-ESM v5 datasets in
`index.warnings`. **Both models added to `exclude_source_ids` in
all configs.** ~10–15 datasets removed from the prod cohort.

### d) Per-variable stats are now multi-period

Stats files (`stats.nc`) now ship 3 periods (`full`, `1940-2014`, `1979-2015`). The aggregator picks `full` for this report. Downstream training that reads stats needs to be explicit about which period it wants — the prior single-period assumption no longer holds.

## Distributional plots

![variable_presence](plots/variable_presence.png)

Variable presence is similar to v0-pilot — most variables present in 80–95% of datasets, ocean variables sparser.

![warnings](plots/warnings.png)

The dominant warning pattern is gone — previously ~150 of 294 datasets had `oday_tos from Oday.tos failed`. The new top warnings are mostly per-variable sanity-range hits (e.g. `SHTFLsfc out of expected range` from a few extreme-flux models) — informational, not blocking.

![crossmodel_stats](plots/crossmodel_stats.png)

Per-dataset mean/std for 16 key variables, coloured by experiment. `oday_tos`, `simon_sitemptop`, `simon_sea_ice_fraction` are all in their expected units now and showing physically plausible distributions.

![finite_fraction](plots/finite_fraction.png)

Most variables are >99% finite. The few variables below that line are the ocean/sea-ice variables that legitimately have NaN over land (kept under the per-variable mask channels).

![mask_coverage](plots/mask_coverage.png)

Below-surface mask sourcing — about evenly split between `nan_union` (publisher-supplied below-surface NaN pattern) and `orog_static` (computed from regridded zg vs orog). A handful of datasets have `none` (no 3D state to compute mask from, e.g. CMCC-CM2-SR5 which ships only `ua`).

![temperature_units](plots/temperature_units.png)

All four temperature-like variables sit cleanly in K. No leftover °C, no NorESM-style bogus low values.

## Caveats

- This is a **partial** report — 350 of an expected ~410+ ok datasets. The remaining ~60 are still in-flight or stuck on ESGF download stalls.
- The new pilot uses `chunk_time=365` (prod setting), so write-path code is being exercised as it will be in prod. No write failures observed across the 350 datasets.
- Source artefacts: [`stats_aggregated.csv`](stats_aggregated.csv) (5.2 MB, 350 rows × 1420 cols), [`outliers.csv`](outliers.csv) (29 rows).
