# Masked, per-land-area snow channels

Experimental data processing for the ACE2S prognostic-snow arms. Writes, for each daily
parent store, a zarr store of masked snow channels that a training config attaches to the
parent with the data loader's `merge` key, plus the matching normalization statistics.
The parents are never modified.

## Why

The two parents define snow differently:

| | CM4 (`2025-03-21-...-daily`) | ERA5 (`2026-08-07-...-daily`) |
|---|---|---|
| `surface_snow_amount` | per unit **land** area (`cell_measures: area: land_area`) | per unit **cell** area: the native ERA5 snow depth is a grid-box mean, diluted by the native land fraction, and the conservative regrid keeps it so |
| `surface_snow_area_fraction` | percent, per unit land area | fraction, computed natively as `min(1, depth / 0.1 m)` from the diluted depth, then regridded |

Checked on the data (2026-09-21): ERA5 target SWE retention per unit frozen precipitation
falls in proportion to the store's `land_fraction` across mask cells (0.84 at full land to
0.47 at half land, also in cells too cold to melt); on native quarter-degree points that are
part land, snow depth is proportional to the point's land fraction; CM4 shows no dependence.
Half of the cells where the ERA5 masked-naive model's snowpack ran away have land fraction
below 0.8, against 8% of mask cells overall.

## Definition written here

Both datasets: snow per unit land area, cover as a fraction in [0, 1], NaN outside the
shared static mask (`snow_mask.nc`: `land_fraction >= 0.5` and not ice sheet; unchanged from
the 2026-08-12 masked arms, see `build_snow_mask.py`).

- **ERA5:** `SWE / land_fraction`; `cover = min(1, cover / land_fraction)`. The SWE division
  is exact (`regrid(lsm * sd_land) / regrid(lsm)`). The cover division is exact for one native
  point and approximate across the native points of a one-degree cell, because the native
  clip at 1 happens before regridding. Exact cover needs the native fields.
- **CM4:** SWE unchanged; `cover / 100`.

The transform is one function, `masked_snow.land_snow_fields`, used by both the store builder
and the stats script.

## Files

| file | role |
|---|---|
| `masked_snow.py` | parent definitions (`era5`, `cm4`, `cm4-1pctco2`, `cm4-picontrol-2026`, the nine `cm4-randco2-*` members), the scenario-training `SOURCE_SETS`, mask loading, the shared transform |
| `build_masked_snow_channels.py` | writes `store-out/<parent>-land-snow-masked.zarr` (4 variables, parent time coordinate, chunk 1 / shard 360) |
| `fit_masked_snow_stats.py` | copies the parent's (or pooled) stats files and adds `_masked` entries (mean, std, one-day residual std over valid cells) and valid-domain time-mean maps; `--pool <source set>` fits over several parents with running moments |
| `pool_daily_stats.py` | pools the per-store daily stats of a source set with `combine_stats.combine_stats` (the control arms' stats and the base of the treatment arms') |
| `run_data_pipeline.sh` | both original stores, both stats, GCS uploads, Beaker stats datasets |
| `run_scenario_pipeline.sh` | the scenario-training stores (2026-06-19 piControl, nine random-CO2 members), pooled and masked stats per source set, uploads |
| `build_snow_mask.py`, `snow_mask.nc` | mask definition and the committed mask (identical to the 2026-08-12 one) |
| `test_masked_snow.py` | unit tests of the transform |

## Outputs

| | store (GCS, beside the parent; copy to weka `/climate-default`) | Beaker stats dataset |
|---|---|---|
| ERA5 | `gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025/2026-08-07-era5-1deg-8layer-daily-1940-2025-land-snow-masked.zarr` | `2026-08-07-era5-1deg-8layer-daily-1940-2025-land-snow-masked-stats-1990-2019` |
| CM4 | `gs://vcm-ml-intermediate/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily/2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-land-snow-masked.zarr` | `2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-land-snow-masked-stats` |
| CM4 1pctCO2 | `gs://vcm-ml-intermediate/2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr-daily/2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr-daily-land-snow-masked.zarr` | pooled, see below |
| CM4 piControl (2026-06-19) | `gs://vcm-ml-intermediate/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily/2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-land-snow-masked.zarr` | pooled, see below |
| CM4 random-CO2 members | `gs://vcm-ml-intermediate/2026-06-19-CM4-like-AM4-random-CO2-1deg-daily/random-CO2-<level>-ic_<n>-land-snow-masked.zarr` | pooled, see below |

### Scenario-training stats

Training on piControl + 1pctCO2 (`pic-1pct`) or + the random-CO2 ensemble (`pic-1pct-randco2`)
normalizes with stats pooled over the training windows of every parent in the source set
(piControl 0156-0235, 1pctCO2 0046-0125, random-CO2 ic_0001 and ic_0002 from 0153), which is
what gives `carbon_dioxide` a usable standard deviation. `pool_daily_stats.py` pools the
per-store daily stats the data pipeline computed over those windows; `fit_masked_snow_stats.py
--pool` adds the `_masked` entries fit over all the parents' valid cells, weighting each parent by
its sampled cells. Two Beaker datasets per source set: `<date>-cm4-<set>-daily-stats` (control
arms) and `<date>-cm4-<set>-daily-land-snow-masked-stats` (treatment arms).

The earlier stores (`...-snow-masked.zarr`, per-cell-area ERA5, percent CM4 cover) and their
stats datasets stay in place so the 2026-08-12 and 2026-09-17 masked-naive checkpoints remain
evaluable against the data they were trained on.

## Training-config changes

Point the `merge` entry at the new store name and the stats mount at the new dataset. The
`_masked` variable names, `mask_` fields, `stepper.input_masking` and everything else are
unchanged.

## Eventual home

This stays experimental until the training methodology is settled. The permanent fix belongs
in the upstream 6-hourly generation: `scripts/era5` should divide native snow depth by the
native land-sea mask on land points before deriving cover and regridding (and regrid both
with land weighting), and `compute_dataset.py` should carry the mask and masked channels the
way it carries `mask_soil_moisture`. `get_stats.py` then covers the masked channels with no
special code, since its reductions skip NaN.
