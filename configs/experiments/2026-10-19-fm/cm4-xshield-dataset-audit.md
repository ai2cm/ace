# CM4 / X-SHiELD dataset audit

Audit (2026-10-21) of the datasets requested as additional training data for a
planned `ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield.yaml`
(based on `base_configs/ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96.yaml`).
The config has not been written yet because of the blocking issues below.

Weka (`/climate-default`) is not mounted locally, so the zarrs were inspected
at their GCS copies in `gs://vcm-ml-intermediate`, assuming the weka names
match. Each zarr was opened with xarray and checked for: required stepper
`in_names`/`out_names`, timestep and time range, calendar, horizontal grid
against the c96 AMIP reference, `ak`/`bk`, CO2 variable and units, surface
fractions, and NaNs.

## Requested paths

| Dataset | Timestep | Steps / range | `ocean_fraction` | CO2 variable |
|---|---|---|---|---|
| `2026-06-19-CM4-piControl-atmosphere-land-4deg-8layer-200yr.zarr` | **6 h** | 292,000; 0151 to 0351, noleap | present | `carbon_dioxide`, 284 ppmv (constant) |
| `2026-06-19-CM4-1pctCO2-atmosphere-land-4deg-8layer-140yr.zarr` | **6 h** | 204,400; 0001 to 0141, noleap | present | `carbon_dioxide`, 284 to 1,145 ppmv |
| `2026-10-11-CM4-like-AM4-random-CO2-4deg/random-CO2-{1x,2x,4x}CO2-ic_{0001,0002,0003}.zarr` | **6 h** | 7,668 each; 0152-10-01 to 0158-01-01, noleap | present | `carbon_dioxide`, 104 to 3,570 ppmv |
| `2025-09-11-X-SHiELD-AMIP-4deg-daily-8layer-11yr-fme-dataset/2025-09-11-X-SHiELD-AMIP-4deg-8layer-11yr.zarr` | 1 day | 4,020; 2013-01-02 to 2024-01-04, julian | present | `global_mean_co2` (fraction) |

## Daily CM4 copies (not the requested paths)

- piControl: `gs://vcm-ml-intermediate/2025-03-21-CM4-piControl-atmosphere-land-4deg-8layer-200yr-daily/2026-06-19-CM4-piControl-atmosphere-land-4deg-8layer-200yr.zarr`, 73,000 days.
- 1pctCO2: `gs://vcm-ml-intermediate/2025-06-18-CM4-1pctCO2-atmosphere-land-4deg-8layer-140yr-daily/2026-06-19-CM4-1pctCO2-atmosphere-land-4deg-8layer-140yr.zarr`, 51,100 days.
- Random CO2: `gs://vcm-ml-intermediate/2026-10-11-CM4-like-AM4-random-CO2-4deg-daily/`, 9 zarrs of 1,917 days each.

None of the daily copies has `ocean_fraction`. All of them have
`carbon_dioxide` in ppmv instead of `global_mean_co2`.

## Blocking issues

1. **The requested CM4 paths are 6-hourly; the model is daily.** The base
   config sets `strict: false` on the train concat, so
   `DatasetProperties.update` only warns ("Inconsistent timesteps between
   datasets") and training would silently treat 6 h steps as days. The daily
   copies must be used instead.
2. **No `global_mean_co2` in any CM4 dataset.** CM4 has `carbon_dioxide` in
   ppmv; the other datasets use a volume mixing ratio (1e6 times smaller). It
   needs renaming and rescaling by 1e-6. `XarrayDataConfig` has no rename or
   derive option.
3. **No `ocean_fraction` in the daily CM4 copies.** It can be derived as
   `1 - land_fraction - sea_ice_fraction`: in the 6-hourly data land + sea ice
   + ocean is exactly 1, and land + sea ice never exceeds 1 in the daily data.
   CM4 `land_fraction` is static `(lat, lon)`; in the other datasets it is
   time-varying.
4. **No normalization stats for the new data.** Grouped normalization only has
   `c96` and `era5` groups. New `cm4` and `xshield` groups would need stats
   under `/climate-default/alexeyy/norm_ablation_0/fm/groups/<label>/`. The
   network and residual stats (`norm_ablation_0/fm/*.nc`) were presumably
   computed from c96 and era5 only. Per-dataset daily stats exist on GCS
   (`…-daily-stats`) but not in the group layout.

## Non-blocking issues

5. **NaNs at the first timestep of the 6-hourly CM4 data.**
   `tendency_of_total_water_path_due_to_advection` is all NaN (4,050 cells)
   at time index 0 in every 6-hourly zarr, and clean at indices 1 to 4, the
   midpoint and the last step. The daily copies have no NaNs at index 0.
6. **Vertical coordinate mismatch.** CM4 `ak`/`bk` differ from c96 and ERA5;
   X-SHiELD matches c96. With `strict: false`, fme uses the vertical
   coordinate of the first training dataset (c96) for all data, so
   `conserve_dry_air` would use the wrong levels for CM4 (as it already does
   for ERA5).
7. **CO2 well outside the current range.** Random CO2 reaches 3,570 ppm and
   1pctCO2 1,145 ppm.
8. **Spin-up periods.** The X-SHiELD stats exclude 2013 as spin-up. The
   random-CO2 data starts 0152-10-01 but its stats start 0153-01-01. Using
   all timesteps includes both.
9. **Random CO2 `ic_0003` was excluded from its stats** (`exclude_runs` in
   `scripts/data_process/configs/CM4-like-AM4-random-CO2-ensemble-atmosphere-4deg-8layer.yaml`),
   which suggests it was meant to be held out.
10. **Mixed calendars.** CM4 is noleap with years 0001 to 0351, X-SHiELD is
    julian, ERA5 and c96 are standard. Expected to be fine for training but
    not tested.

## Checks that passed

- All datasets are on the same 45×90 Gaussian grid (X-SHiELD latitude differs
  from c96 by at most 3.8e-6 degrees).
- Units match apart from CO2.
- No NaNs at the first timestep of the daily copies or X-SHiELD.
- Land, ocean and sea-ice fractions are within [0, 1].

## Needed before writing the config

- Daily CM4 zarrs on weka with `global_mean_co2` (volume mixing ratio) and
  `ocean_fraction`, or rename/derive support added to fme in its own PR.
- A decision on computing normalization stats for `cm4` and `xshield` groups.
