# Blowup investigation

Tooling to reproduce and characterize the long-rollout blowup of the model trained
in beaker run `01KTCTN1TCJ89NGQCMTJ9GWC2K`
(`train-4deg-daily-v1-era5-only-residual-rs0`), a 4-degree, 8-layer, daily ERA5
`NoiseConditionedSFNO` with residual prediction trained with an ensemble CRPS loss.

## What blows up

The `long_46year` inference (1979 IC, 16794 daily steps) that runs inline during
training does not stay stable. The annual-mean evolution logged to wandb
(`a0tt7761`) shows a monotonic runaway that begins in the first year and compounds
roughly exponentially:

| variable | 1979 | 1982 | 1985 | 1988 | 1990 | target |
|---|---|---|---|---|---|---|
| `air_temperature_7` (K) | 285 | 299 | 362 | 490 | 609 | ~285 |
| `surface_temperature` (K) | 287 | 293 | 311 | 352 | 392 | ~288 |
| `TMP2m` (K) | 287 | 294 | 325 | 388 | 439 | ~287 |
| `specific_total_water_7` | 0.0087 | 0.011 | 0.017 | 0.034 | 0.051 | ~0.009 |

So it is a warming/moistening climate drift, not a sudden NaN spike, and it is
worst in the near-surface layers (level 7). Several of the 8 ICs eventually go
NaN. This tooling reproduces the drift from a single IC at daily resolution to
pin down which variables leave the training distribution first and what spatial
and vertical form the drift takes early on.

## Layout

- `download_checkpoint.sh` — fetch `best_inference_ckpt.tar` (best-validation EMA
  checkpoint) from the run's result dataset into `checkpoint/`.
- `prepare_data.py` — stage the variables and time range the rollout needs from
  the GCS ERA5 zarr into a local netCDF in `data/`.
- `inference.yaml` — standalone `fme.ace.inference` config: single 1979-01-01 IC,
  reads the staged local netCDF, full prognostic+diagnostic fields saved every
  daily step.
- `run_inference.sh` — run the rollout, writing `output/run/`.
- `diagnose_blowup.py` — compute and plot the diagnostics into
  `output/diagnostics/` (figures + `summary.md`).

## Usage

This investigation does not need a weka mount; it only needs GCS read access.

```bash
cd scripts/blowup_investigation
./download_checkpoint.sh
python prepare_data.py       # stage 8 years -> data/era5_4deg_blowup_slice.nc
./run_inference.sh 2922      # 8-year rollout from the staged data
python diagnose_blowup.py
```

The default horizon is 8 years (2922 daily steps, through ~1987). That is long
enough for the drift to develop into a clear blowup but far short of the full
46-year rollout. At ~2.4 s/step on CPU the rollout takes a couple of hours, which
a GPU cuts to minutes.

`prepare_data.py` and the config write into this folder (`data/`, `output/`,
`checkpoint/`), all of which are gitignored.

## Why staged local data (and the direct-GCS alternative)

Reading the GCS zarr through the fme forcing loader requires a working forkserver
(GCSFS is not fork-safe, so the zarr code path always spawns a forkserver worker;
there is no `num_workers=0` escape for the zarr engine). `prepare_data.py` avoids
that dependency entirely by downloading the needed variables and time range into
a local netCDF, which the loader reads single-process (`num_data_workers: 0`,
`engine: h5netcdf`). This is the default and was validated on a box with no weka
mount and a broken forkserver.

To stream directly from GCS instead (no staging step, but requires a healthy
forkserver), edit the `forcing_loader` and `initial_condition` blocks of
`inference.yaml` per the comment there to use the zarr engine.

## Data source

The model was trained on `gs://vcm-ml-intermediate/2026-04-17-era5-4deg-8layer-daily-1940-2025/2026-03-19-era5-4deg-8layer-1940-2025.zarr`
(the source rsync'd to `/climate-default` for training; see
`scripts/data_process/configs/era5-4deg-8layer-1940-2025.yaml`). Normalization
statistics are baked into the checkpoint, so loading the stepper needs no external
stats files.

## Diagnostics

`diagnose_blowup.py` normalizes each predicted field by the training mean and
full-field std stored in the checkpoint, giving a z-score that is roughly N(0,1)
over the grid for in-distribution states. It then reports:

- per-variable onset of drift, ordered, where onset is the first step the
  area-weighted global-mean z-score has moved more than 1 std unit from its
  day-0 (in-sample) value (`summary.md`);
- `oos_onset.png` — grid-max |z| and global-mean z vs time for the earliest
  variables to leave the training distribution;
- `global_means.png` — prediction vs ERA5 global means in physical units;
- `level_structure.png` — global-mean z per model level for temperature, total
  water, and winds, showing which levels drift first;
- `spatial_z.png` — maps of z for the earliest-diverging variable, to show
  whether the drift is broad or a localized grid-point instability.
