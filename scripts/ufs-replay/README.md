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

1. Read MOM6 ocean variables (3-D + surface + layer thickness `ho`)
2. Read FV3 atmosphere forcing and sea-ice variables
3. Average 3-hourly FV3 atmosphere to 6-hourly ocean cadence
4. Regrid ocean and atmosphere to Gaussian grid (F90 = 1°) via xESMF
5. Thickness-weighted vertical coarsening (75 → 19 levels matching CM4)
6. Split 3-D fields into per-level 2-D variables
7. Derive additional variables (SST, ssu/ssv, deptho, wfo, hfds, etc.)
8. Insert NaN on land, nearest-neighbour fill residual coastal NaN
9. Coarsen in time (6-hourly → daily)

## Quick Start

```bash
# Create local conda environment
make create_environment

# Local test run (DirectRunner)
make ufs_replay_direct_test_run

# Or run directly with Python
cd pipeline
python3 xr-beam-pipeline.py \
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
| `--time_coarsen_factor` | `4` | Temporal coarsening (4 = 6h→daily) |
| `--process_time_chunksize` | `4` | Ocean timesteps per Beam chunk |
| `--output_time_shardsize` | `360` | Times per zarr shard (~1 year daily) |
| `--vertical_coarsening_indices` | built-in | JSON list of [start,end) pairs |

## Comparison with `scripts/era5/`

| Aspect | ERA5 | UFS Replay |
|--------|------|------------|
| Source | ARCO-ERA5 (0.25°) | NOAA UFS replay (0.25°) |
| Streams | 4 parallel (flux, surface, pressure, model) | 1 merged (ocean + atmo per chunk) |
| Vertical | Pressure-weighted (137→8 layers) | Thickness-weighted (75→19 levels) |
| Time step | 6-hourly output | 6-hourly → daily |
| Runner | Dataflow / DirectRunner | Dataflow / DirectRunner |
