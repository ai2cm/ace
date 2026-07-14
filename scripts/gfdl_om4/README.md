# GFDL OM4/CM4 ocean dataset pipeline

Produces training-reference datasets from native-grid 0.25° tripolar OM4/CM4
model output, following the operational pattern of `scripts/era5/`
(xarray_beam on Google Cloud Dataflow, DirectRunner for local subset runs).

Each invocation is driven by a YAML config (see `configs/`) naming the
source stores, the streams and variables to process, and the output layout,
and writes one templated, sharded zarr v3 store.

Contents:

- `pipeline/run.py`: the pipeline itself — opens the configured streams,
  builds the output template (statics stamped in and written by the
  driver), and runs one beam branch per stream through per-chunk transforms
  (C-grid→tracer-center interpolation, vector rotation, wetmask-normalized
  regridding, level splitting) into the output store. Every output variable
  carries `source_store`/`source_variable` (and, for derived variables,
  `derivation`) provenance attrs.
- `pipeline/config.py`: YAML→dataclass configuration (dacite). Stream
  options cover source-dim renaming (e.g. ice-model `xT/yT/xB/yB` onto the
  ocean `xh/yh/xq/yq` conventions), time subsampling to the shared snapshot
  instants, full-cell (per-total-cell-area) regridding for selected
  variables, and named postprocess transforms.
- `pipeline/postprocess.py`: named post-regrid transforms selectable per
  stream — Kelvin `sst`, `hfds_total_area`, and the sea-ice conventions
  (ice-velocity masking, thickness zeroing, `sea_ice_volume`).
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
- `pipeline/face_masks.py`: one-time setup step for sources whose staggered
  velocities carry remap-born zeros over land (MOM6's online z\*-remap
  leaves coastal velocity faces valid with value exactly 0.0 where the
  native vertical grid masks them as land). Scans the source, flags faces
  that are structurally zero with a dry tracer neighbor, and publishes the
  masks as a versioned GCS artifact; streams opt in via `face_mask_url`.
  During center interpolation a wet cell whose faces on an axis are all
  land is a wall for that axis, and that grid-relative component is set
  to the no-normal-flow value 0 before rotation, so every velocity output
  keeps the tracer wetmask footprint (`mask_k`).

## Setup

One-time setup to prepare the conda environment and precompute regridding
weights:

```
make create_environment      # conda env gfdl-om4-ingestion
make generate_weights        # precompute all conservative regridding weights
make generate_face_masks     # scan sources for remap-born zero velocity faces
```

## Running

### Test runs

A local DirectRunner subset run (a few timesteps, writes to a scratch
store):

```
make test_run
```

or directly, with any beam pipeline options after the script's own
arguments:

```
python -m pipeline.run --config configs/om4-picontrol-1deg.yaml \
    --num-timesteps 6 --output-path <url> --runner DirectRunner
```

### Production runs

A production run on Google Cloud Dataflow — build and push the worker
image, then launch (the config's output path is used as-is):

```
make push_dataflow
make dataflow                # or CONFIG=configs/<other>.yaml make dataflow
```

`make dataflow` invokes `run-dataflow.sh`, which supplies the Dataflow
resource flags (project, region, temp location, worker shape, container
image). Unlike the era5 pipeline, the pipeline code here is a package, so
the worker image copies `pipeline/` in and puts it on `PYTHONPATH` rather
than relying on `--save_main_session`. Workers authenticate to GCS with
the project service account; no S3/OSN credentials are involved.
