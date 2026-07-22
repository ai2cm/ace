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

One-time setup to prepare the conda environment and precompute the durable
pipeline inputs (published under the dated, immutable
`INPUTS_URL_ROOT` prefix in the Makefile — regenerating requires a new
prefix or explicit overwrite flags):

```
make create_environment      # conda env gfdl-om4-ingestion
make generate_weights        # conservative regridding weights (F90 + F22.5)
make generate_face_masks     # per-simulation remap-born-zero face masks
```

Face-mask artifacts are per-simulation (see `pipeline/face_masks.py`): each
`generate_face_masks_*` target scans that source's first year against the
independently-censused expected surface-face counts baked into the target.

## Configs

One config per output store, under `configs/`: {piControl, 1pctCO2} ×
{1° `F90`, 4° `F22.5`}. All input artifacts (weights, face masks) come from
the permanent inputs prefix, sources are read from
`vcm-ml-raw-flexible-retention`, and outputs are flat dated zarrs in
`gs://vcm-ml-intermediate/`; nothing consumed by a production run lives
under `vcm-ml-scratch`.

## Running

### Smoke tests

Each config has a local DirectRunner smoke test that runs a few timesteps
into a throwaway scratch store, asserts the wetmask conform step is a no-op
(`--max-conformed-cells 0`), and checks the output opens with the expected
variable set:

```
make smoke_tests             # all four configs + the checks below
make smoke_test_picontrol_1deg   # or any single config
```

`make smoke_tests` additionally verifies that the piControl and 1pctCO2
configs derive identical wetmasks (`make check_wetmask_equivalence`;
downstream training/analysis assumes the stores share one mask — a
difference is a stop-and-report finding, not something to conform around),
and that a repeat run against an existing output store refuses to
initialize into it (`make smoke_test_repeat_fails`).

The pipeline can also be invoked directly, with any beam pipeline options
after the script's own arguments:

```
python -m pipeline.run --config configs/om4-picontrol-1deg.yaml \
    --num-timesteps 6 --output-path <url> --runner DirectRunner
```

### Production launch checklist

1. **Smoke test** — `make smoke_tests` (or the single-config target for the
   config being launched) against the exact configs to be launched.
2. **Launch** — build and push the worker image, then launch on Google
   Cloud Dataflow (the config's output path is used as-is, and the run
   aborts if a store already exists there). Like the smoke tests, the
   launch targets pass `--max-conformed-cells 0`: the production sources
   have a static footprint, so any wetmask conforming is a failure, not
   something to repair silently. A source that genuinely needs the
   conform step's fill-from-above repair is a deliberate opt-in (invoke
   `run-dataflow.sh` directly without the flag):

   ```
   make push_dataflow
   make dataflow_picontrol_1deg     # one target per config
   ```

   Expect a silent multi-minute setup phase before workers start: the
   driver builds the template (processing the first timestep of every
   stream) and writes its metadata serially — accepted behavior, not a
   hang.
3. **Inspect** — after the job completes, check the store:

   ```
   python -m pipeline.check_output --config configs/om4-picontrol-1deg.yaml
   ```

   and review the job's Dataflow console page for stage-level errors
   before consuming the output.

`make dataflow*` invokes `run-dataflow.sh`, which supplies the Dataflow
resource flags (project, region, temp location, worker shape, container
image). Unlike the era5 pipeline, the pipeline code here is a package, so
the worker image copies `pipeline/` in and puts it on `PYTHONPATH` rather
than relying on `--save_main_session`. Workers authenticate to GCS with
the project service account; no S3/OSN credentials are involved.
