# GFDL OM4/CM4 ocean dataset pipeline

Produces training-reference datasets from native-grid 0.25° tripolar OM4/CM4
model output, following the operational pattern of `scripts/era5/`
(xarray_beam on Google Cloud Dataflow, DirectRunner for local subset runs).

Current contents — the regridding library layer:

- `pipeline/grids.py`: analytic Gaussian target grids (`F90` = 1°, `F22.5` =
  4°) with exact quadrature-weight cell areas.
- `pipeline/ocean_emulators_port.py`: utilities ported from the ai2cm fork of
  [ocean_emulators](https://github.com/ai2cm/ocean_emulators) — supergrid
  conversion, vector rotation, C-grid→tracer-center interpolation, and
  wetmask-normalized conservative regridding.
- `pipeline/weights.py`: one-time setup step that precomputes xESMF
  conservative weights for a source×target grid pair and stores them as a
  versioned GCS artifact, plus the per-process cached regridder loader used
  by workers.

## Setup

```
make create_environment      # conda env gfdl-om4-ingestion
make generate_weights_one_degree
```
