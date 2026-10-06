# UFS GEFSv13 Replay Ocean Pipeline

xarray-beam pipeline for processing the NOAA UFS GEFSv13 replay ocean (MOM6)
and atmosphere (FV3) dataset into a training-ready zarr store for SamudrACE.

This pipeline follows the same runner/infrastructure pattern as `scripts/era5/`.

## Data Sources

| Component | URL | Resolution | Frequency |
|-----------|-----|------------|-----------|
| MOM6 ocean | `gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/06h-freq/zarr/mom6.zarr` | 0.25° | 6-hourly |
| FV3 atmosphere | `gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr` | 0.25° | 3-hourly |

## Pipeline Steps

The ocean and atmosphere are processed in two independent Beam streams
(mirroring the multi-stream setup of `scripts/era5/`) that write to the same
output zarr store.  The 3-hourly atmosphere times are validated up front to
exactly interleave the 6-hourly ocean times, so the streams align purely by
integer chunk offsets — no per-chunk time matching is needed.

The pipeline always emits data at the ocean model's native 6-hourly cadence —
no time coarsening happens here.  Coarser products (daily, 5-day, etc.) are
produced downstream from this raw output by
[`scripts/data_process/time_coarsen.py`](../data_process/time_coarsen.py), a
separate, much cheaper post-processing step (plain zarr region writes, no
regridding) — see "Downstream time coarsening" below.

Ocean stream (6-hourly MOM6 chunks):

1. Thickness-weighted (`ho`) vertical coarsening (75 → 19 levels matching
   CM4) at native horizontal resolution, splitting 3-D fields into
   per-level 2-D variables — done before horizontal regridding since
   thickness-weighted vertical averaging and horizontal regridding don't
   commute, matching the CM4 convention (and cheaper, since only 19
   levels then need to go through the regridder instead of 75)
2. Regrid to Gaussian grid (F90 = 1°) via xESMF
3. Derive additional variables (sst, ssu/ssv, wfo, hfds, etc.)
4. Insert NaN on land, nearest-neighbour fill residual coastal NaN

Atmosphere stream (3-hourly FV3 chunks):

1. Derive frozen precipitation rate from bucket accumulations
2. Average 3-hourly fields to the 6-hourly ocean cadence, before
   regridding, to minimize how many timesteps pass through the regridder
3. Regrid to Gaussian grid via xESMF
4. Mask sea-ice variables to the ocean

## Downstream time coarsening

The pipeline always outputs raw 6-hourly data; producing a coarser product
(daily, 5-day, etc.) is a separate, much cheaper step (plain zarr region
writes, no xESMF regridding) run on that raw output, via
[`scripts/data_process/time_coarsen.py`](../data_process/time_coarsen.py)
(`make coarsen_daily`; config at
[`configs/ufs-replay-ocean-1deg-19level-daily.yaml`](../data_process/configs/ufs-replay-ocean-1deg-19level-daily.yaml)).

It's config-driven, and distinguishes ocean/sea-ice *state* variables
(per-level `thetao`/`so`/`uo`/`vo`, `sst`/`ssu`/`ssv`/`zos`, sea-ice
fraction/thickness), which are **snapshotted** (the last native timestep of
each day, not averaged), from atmosphere forcing and ocean surface flux
variables, which are **window-averaged** over the day — matching the
convention used by the SHiELD-family `time_coarsen` configs elsewhere in
`data_process`. This is the physically appropriate choice for training on
daily data: it preserves actual model states rather than smoothing them,
while still correctly time-integrating forcing. (A plain uniform mean over
every variable, with no state/forcing distinction, is also possible by
putting every time-varying variable in `window_names` and leaving
`snapshot_names` empty.)

Note the resulting time label: with any snapshot variables present (as in
our config), `time_coarsen.py` labels each day at 18Z (the snapshotted
state's own valid time), not the 09Z center-of-day label a pure mean — or
this pipeline's daily output before it was simplified to always emit raw
6-hourly data — would produce.

## Quick Start

```bash
# Create local conda environment
make create_environment

# Local test run (DirectRunner)
make ufs_replay_direct_test_run

# Or run directly with Python
cd pipeline
python3 ufs-replay-pipeline.py \
    gs://vcm-ml-scratch/test.zarr \
    2023-12-01T06:00:00 \
    2023-12-31T18:00:00 \
    --output_grid F90 \
    --runner DirectRunner \
    --save_main_session
```

## Production Runs (Dataflow)

```bash
# Build and push Docker image
make build_dataflow push_dataflow

# Submit production job
make ufs_replay_dataflow

# Submit test job on Dataflow
make ufs_replay_dataflow_test_run
```

## Key CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output_grid` | `F90` | Gaussian grid spec (`F90`=1°, `F22.5`=4°) |
| `--process_time_chunksize` | `4` | Native 6-hourly ocean timesteps per Beam chunk |
| `--output_time_shardsize` | `360` | Times per zarr shard (~90 days at 6-hourly) |
| `--vertical_coarsening_indices` | built-in | JSON list of [start,end) pairs |

## Comparison with `scripts/era5/`

| Aspect | ERA5 | UFS Replay |
|--------|------|------------|
| Source | ARCO-ERA5 (0.25°) | NOAA UFS replay (0.25°) |
| Streams | 4 parallel (flux, surface, pressure, model) | 2 parallel (ocean, atmosphere) |
| Vertical | Pressure-weighted (137→8 layers) | Thickness-weighted (75→19 levels) |
| Time step | 6-hourly output | 6-hourly (native; coarsen downstream if needed) |
| Runner | Dataflow / DirectRunner | Dataflow / DirectRunner |
