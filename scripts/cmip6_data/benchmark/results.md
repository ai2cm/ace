# Data Loading Benchmark: Zarr vs NetCDF

## Summary

NetCDF with fork-based data workers is **8x faster** per batch than zarr
without workers (our previous default). Zarr with forkserver workers is
slower than zarr without workers due to forkserver overhead.

## Setup

- **Hardware**: Single GCP VM with T4 GPU, local SSD storage
- **Dataset**: CMIP6 daily pilot v0 -- 112 datasets across 29 source
  models, each ~364 daily timesteps on a 45x90 lat/lon grid with 66
  variables (~310 MB per zarr store, 36 GB total)
- **NetCDF conversion**: Each zarr store converted to yearly `.nc` files
  with 1-day (1-timestep) internal HDF5 chunking via `zarr_to_netcdf.py`
- **Benchmark**: 100 batches, batch_size=4, n_timesteps=2 (matching
  training config), no simulated GPU compute (sleep=0)
- **Date**: 2026-05-04

## Results

| Configuration       | Init (s) | Mean batch (s) | Total 100 batches (s) | Speedup vs baseline |
|---------------------|----------:|---------------:|----------------------:|--------------------:|
| zarr, 0 workers     |     76.5 |         0.9946 |                 99.46 |           1.0x      |
| zarr, 4 workers     |     73.8 |         1.8945 |                189.45 |           0.5x      |
| netcdf, 0 workers   |    182.6 |         8.2858 |                828.58 |           0.1x      |
| netcdf, 4 workers   |     18.4 |         0.1236 |                 12.36 |         **8.0x**    |
| netcdf, 8 workers   |     18.0 |         0.1264 |                 12.64 |         **7.9x**    |

Baseline is zarr with 0 workers, which is the configuration used for
the ongoing CMIP6 daily pilot smoke-test training run.

## Analysis

**Why zarr workers hurt**: Zarr uses async I/O internally, which is not
fork-safe. PyTorch must use the `forkserver` multiprocessing context
instead of `fork`. Forkserver spawns fresh Python processes (no
copy-on-write memory sharing), so each worker re-imports all modules and
re-initializes all 112 datasets. The coordination overhead exceeds the
parallelism benefit for local storage where zarr's async I/O already
provides some overlap.

**Why netcdf without workers is slow**: NetCDF4 uses synchronous file
reads. Without workers, every batch blocks the main process on disk I/O
with no overlap or prefetching. Each read opens a file handle,
seeks to the requested time chunk, reads, and closes -- all serially.

**Why netcdf with workers wins**: NetCDF4 reads are fork-safe, so
PyTorch can use the default `fork` context. Workers start instantly
via copy-on-write (18s init vs 77s for zarr), share the parent's
memory pages, and prefetch batches in parallel while the main process
runs the forward/backward pass on GPU. The 1-day internal HDF5
chunking means each random 2-timestep read touches only ~2 MB of data
instead of scanning the full ~310 MB file.

**Why 8 workers is not faster than 4**: Local SSD bandwidth is already
saturated at 4 workers. Additional workers add scheduling overhead
without improving I/O throughput.

## Recommendation

Use `engine: netcdf4` with `num_data_workers: 4` and
`prefetch_factor: 4` for local training. Convert zarr stores to netCDF
using `scripts/cmip6_data/zarr_to_netcdf.py` before training.

## Reproducing

```bash
# Convert zarr to netCDF (one-time, ~1 hour for 120 datasets)
python scripts/cmip6_data/zarr_to_netcdf.py \
    scripts/cmip6_data/data/cmip6-daily-pilot/v0 \
    scripts/cmip6_data/data/cmip6-daily-pilot/v0-nc \
    --workers 4

# Run benchmark
bash scripts/cmip6_data/benchmark/run_benchmark.sh
```
