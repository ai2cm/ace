# ACE2S masked-snow training, 1-deg daily

Four models testing whether excluding ocean/sea-ice and ice sheets from the snow variables fixes
the drift and misallocated-effort pathologies seen in the unmasked arms. The snow channels are
replaced by `_masked` variants that are NaN outside a static validity domain; ACE's existing
masking machinery (the coupled/ocean-model pattern) does the rest: outputs are NaN-filled outside
the mask every step, the loss zeroes prediction and target where the target is NaN, and per-variable
mask-aware area weights remove masked cells from every metric, so RMSE/bias/channel-mean and
best-checkpoint selection stay finite. No `fme` code changes.

## Mask

Valid (=1) where `land_fraction ≥ 0.5` and NOT ice sheet, with ice sheet = any of: CM4
`glac_fraction ≥ 0.1` (from the 6-hourly store), ERA5 local-summer mean snow cover > 0.5, or ERA5
climatological max SWE > 2000 kg/m2 (removes deep ice-sheet margins approaching the 10000 cap).
One shared mask for both datasets (identical grids), built by `build_snow_mask.py` → `snow_mask.nc`
(committed) + `snow_mask.png`. 14236 valid cells = 65.1% of land; matches the snow-distribution
report's non-ice census (14258) with the max-SWE criterion trimming the ~90 deep-margin cells.

## Data: sidecar zarrs, parents untouched

`build_masked_snow_channels.py <dataset>` writes a standalone sidecar zarr per dataset with exactly
four variables — `surface_snow_amount_masked`, `surface_snow_area_fraction_masked` (parent channels
with NaN where mask=0) and their static `mask_<name>` variables — time coordinate copied from the
parent, chunking matched (time chunk 1, shard 360). The loaders merge parent + sidecar at load time
(`dataset: {merge: [...]}`); the sidecar's `mask_*` variables bind through the merged dataset
properties. Rollback = delete the sidecar.

| sidecar (GCS beside parent; weka /climate-default) |
|---|
| `2026-08-07-era5-1deg-8layer-daily-1940-2025-snow-masked.zarr` |
| `2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-snow-masked.zarr` |

## Arms

Configs are generated from their parents; the diff is exactly: snow names → `_masked` (in/out
names, transforms keys, force_positive), loader `dataset:` → `merge: [parent, sidecar]`,
`stepper.input_masking: {mask_value: 0, fill_value: 0.0}` (fed-back input filled with physical 0),
and the stats mount. No `fill_nans_on_normalize` anywhere.

| arm | parent | isolates |
|---|---|---|
| masked-naive | 2026-08-07 naive snowprog | masking alone (does it remove the σ pathology by itself?) |
| masked-log1p | 2026-08-09 log1p | masking on top of the best current encoding |

Stats are fit over the valid domain by `fit_masked_snow_stats.py` (entries keyed under the
`_masked` names; valid-domain time-mean maps added to the patched `time-mean.nc`). Masked-domain
snow metrics are NOT comparable to the unmasked arms' — the `_masked` W&B keys make accidental
overlays impossible, which is intentional.

## Launch

```bash
./run-ace-train.sh                    # ai2/jupiter, 8 GPUs; all four
CLUSTER=ai2/titan ./run-ace-train.sh  # ai2/titan, 4 GPUs
```

Expected cosmetics in W&B: snow power-spectrum and zonal-mean panels are blank (SHT/plain-mean of
NaN fields) and snow histograms are junk for the `_masked` channels; all reductions that matter are
mask-aware. First-epoch checks: finite snow val losses and `time_mean_norm/rmse/channel_mean`,
bias maps NaN over the masked domain, `masked-naive` snow σ at seasonal scale in its stats.
