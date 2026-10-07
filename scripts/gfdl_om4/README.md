# GFDL 0.25° tripolar ingestion pipelines

Two pipelines that produce training-reference datasets from native-grid
0.25° tripolar GFDL model output, following the operational pattern of `scripts/era5/`
(xarray_beam on Google Cloud Dataflow, DirectRunner for local subset runs).

Each invocation is driven by a YAML config (see `configs/`) naming the
source stores, the streams and variables to process, and the output layout,
and writes one templated, sharded zarr v3 store.

| Pipeline | Package | Configs | Make targets | Output |
| --- | --- | --- | --- | --- |
| OM4 | `pipeline/om4/` | `configs/om4-*.yaml`, `configs/cm4-*-1daily.yaml` | `om4.mk` (unprefixed) | OM4/CM4 ocean stores, one per config |
| randco2 | `pipeline/randco2/` | `configs/cm4-like-am4-randco2-*.yaml` | `randco2.mk` (`randco2_*`) | CM4-like-AM4 random-CO2 sea-surface stores, one per member |

Both run from `scripts/gfdl_om4/` through the one `Makefile`, which holds
the shared targets and includes `om4.mk` and `randco2.mk`.

## Shared core

Top-level modules of `pipeline/`, used by both pipelines; neither
subpackage imports the other.

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
- `pipeline/postprocess.py`: the postprocess contract (`Postprocess`,
  `ChunkContext`) and provenance-attr helpers.
- `pipeline/config_io.py`: YAML loading (with optional `{member}`
  substitution) and the `OutputConfig`/`WetmaskConfig` dataclasses.
- `pipeline/zarr_io.py`: store opening, the output-store-absent guard, the
  wetmask footprint assertion, the read-width helpers
  (`source_time_chunk_size`, `shard_aligned_chunk_size`), and the sharded
  zarr write.

### Read strategy

Both pipelines read each source at its own time chunk width
(`source_time_chunk_size`, from the source's `preferred_chunks`), so each
source chunk is fetched and decoded once rather than once per timestep it
holds. OM4 reads 3D variables one level at a time. Before consolidating
into output shards, read chunks are split to `gcd(read width, shard)`
(`shard_aligned_chunk_size`), which puts every chunk boundary on a shard
boundary; where the read width divides the shard the split is a no-op. The
output layout (time chunk and shard from the config's `output`) does not
depend on the read width.

### Postprocess config

Each entry of a stream's `postprocess` list is a transform name, or
`{name: <transform>, sources: {<argument>: <variable>}}` naming the output
variables it reads. `kelvin_sst` needs `sources: {celsius_sst: <name>}`
(`tos` in the OM4 configs, `SST` in randco2's). `sea_ice` and
`sea_ice_fraction_consistency` take `sea_ice_fraction` and
`ocean_sea_ice_fraction`, defaulting to those names. A name the stream does
not produce fails at config load.

### Environment, weights and worker image

One conda env, one `Dockerfile`, one worker image (`VERSION`/`IMAGE_NAME`
in the `Makefile`) for both pipelines:

```
make create_environment      # conda env gfdl-om4-ingestion
make generate_weights        # conservative regridding weights (F90 + F22.5), read by both
make build_dataflow          # local image build; smoke-imports pipeline.om4.run and pipeline.randco2.run
make push_dataflow           # build and push to Artifact Registry
make enter                   # shell in the image with this directory mounted
```

The weights are published under the dated, immutable `INPUTS_URL_ROOT`
prefix in the `Makefile` — regenerating requires a new prefix or explicit
overwrite flags.

`run-dataflow.sh {om4|randco2} RUNNER CONFIG [FLAGS...]` launches either
pipeline; the first argument selects the module (`pipeline.<name>.run`),
the worker machine type and the temp location, and the rest (project,
region, container image, worker counts) is shared. A `DataflowRunner`
launch refuses to start without `SDK_CONTAINER_IMAGE`, which the make
targets set to `$(IMAGE_NAME)`. Unlike the era5 pipeline, the pipeline code
here is a package, so the worker image copies `pipeline/` in and puts it on
`PYTHONPATH` rather than relying on `--save_main_session`. Workers
authenticate to GCS with the project service account; no S3/OSN
credentials are involved.

## OM4 pipeline: OM4/CM4 ocean

Make targets in `om4.mk`, unprefixed (`make smoke_tests`, `make dataflow_*`).

Contents, under `pipeline/om4/`:

- `pipeline/om4/run.py`: the pipeline itself — opens the configured streams,
  builds the output template (statics stamped in and written by the
  driver), and runs one beam branch per stream through per-chunk transforms
  (C-grid→tracer-center interpolation, vector rotation, wetmask-normalized
  regridding, level splitting) into the output store, read as in
  [Read strategy](#read-strategy). Every output variable
  carries `source_store`/`source_variable` (and, for derived variables,
  `derivation`) provenance attrs.
- `pipeline/om4/config.py`: YAML→dataclass configuration (dacite). Stream
  options cover source-dim renaming (e.g. ice-model `xT/yT/xB/yB` onto the
  ocean `xh/yh/xq/yq` conventions), time subsampling to the shared snapshot
  instants, full-cell (per-total-cell-area) regridding for selected
  variables, and named postprocess transforms.
- `pipeline/om4/postprocess.py`: named post-regrid transforms selectable per
  stream — Kelvin `sst`, `hfds_total_area`, and the sea-ice conventions
  (ice-velocity masking, thickness zeroing, `sea_ice_volume`).
- `pipeline/om4/face_masks.py`: one-time setup step for sources whose staggered
  velocities carry remap-born zeros over land (MOM6's online z\*-remap
  leaves coastal velocity faces valid with value exactly 0.0 where the
  native vertical grid masks them as land). Scans the source, flags faces
  that are structurally zero with a dry tracer neighbor, and publishes the
  masks as a versioned GCS artifact; streams opt in via `face_mask_url`.
  See `run._rotate_pairs` for how the flagged faces and the wall-zero
  fill keep every velocity output on the tracer wetmask footprint
  (`mask_k`).
- `pipeline/om4/check_output.py`, `pipeline/om4/check_wetmask_equivalence.py`:
  post-run and cross-config checks used by `om4.mk`.

### Setup

After the shared setup above, precompute the OM4 face masks (published
under `INPUTS_URL_ROOT` like the weights):

```
make generate_face_masks     # per-simulation remap-born-zero face masks
```

Face-mask artifacts are per-simulation (see `pipeline/om4/face_masks.py`): each
`generate_face_masks_*` target scans that source's first year and checks the
flagged surface-face counts against the independent census baked into the
target.

### Configs

One config per output store, under `configs/`: {piControl, 1pctCO2} ×
{1° `F90`, 4° `F22.5`}, plus the 1° sea-ice budget companions
`om4-{picontrol,1pctco2}-1deg-5daily-budget.yaml`: 5-daily block means
(`time_block_mean: 20`) of the SIS2 surface fluxes and ice/snow transport,
full-cell only, on the 5-daily stores' time axis, to merge with them at
training time. All input artifacts (weights, face masks) come from
the permanent inputs prefix, sources are read from
`vcm-ml-raw-flexible-retention`, and outputs are flat dated zarrs in
`gs://vcm-ml-intermediate/`; nothing consumed by a production run lives
under `vcm-ml-scratch`.

### Running

#### Smoke tests

Each config has a local DirectRunner smoke test that runs a few timesteps
into a throwaway scratch store — the run itself fails unless every chunk's
valid-data footprint equals the wetmask — and checks the output opens with
the expected variable set:

```
make smoke_tests             # all four configs + the checks below
make smoke_test_picontrol_1deg   # or any single config
```

`make smoke_tests_budget` covers the budget companions; `make test` runs
the unit tests.

`make smoke_tests` additionally runs:

- `make check_wetmask_equivalence` — the piControl and 1pctCO2 configs must
  derive identical wetmasks. Downstream training/analysis assumes the
  stores share one mask.
- `make smoke_test_repeat_fails` — a repeat run against an existing output
  store must refuse to initialize into it.

The pipeline can also be invoked directly, with any beam pipeline options
after the script's own arguments:

```
python -m pipeline.om4.run --config configs/om4-picontrol-1deg-5daily.yaml \
    --num-timesteps 6 --output-path <url> --runner DirectRunner
```

#### Production launch checklist

1. **Smoke test** — `make smoke_tests` (or the single-config target for the
   config being launched) against the exact configs to be launched.
2. **Launch** — build and push the worker image, then launch on Google
   Cloud Dataflow (the config's output path is used as-is, and the run
   aborts if a store already exists there). Like the smoke tests, a launch
   fails on any chunk whose valid-data footprint differs from the wetmask:
   the production sources have a static footprint, so a difference is a
   stop-and-report finding, not something to repair silently.

   ```
   make push_dataflow
   make dataflow_picontrol_1deg     # one target per config
   ```

   `make dataflow*` first checks that `$(IMAGE_NAME)` exists in Artifact
   Registry (`make check_dataflow_image`).

   Expect a silent multi-minute setup phase before workers start: the
   driver builds the template (processing the first timestep of every
   stream) and writes its metadata serially — accepted behavior, not a
   hang.
3. **Inspect** — after the job completes, check the store:

   ```
   python -m pipeline.om4.check_output --config configs/om4-picontrol-1deg-5daily.yaml
   ```

   and review the job's Dataflow console page for stage-level errors
   before consuming the output.

`make dataflow*` invokes `run-dataflow.sh om4` (shared core above).

## randco2 pipeline: CM4-like-AM4 random-CO2 sea surface

Carried over from PR 1444's `scripts/gfdl_cm4_like_am4_randco2/README.md`.
Make targets in `randco2.mk`, prefixed `randco2_` (`make randco2_smoke_tests`,
`make randco2_dataflow_*`); its variables are prefixed `RANDCO2_`
(`RANDCO2_CONFIG`, `RANDCO2_MEMBERS`, ...) apart from `MEMBER`.

Produces 1° sea-surface training-reference stores from the nine members of
the CM4-like-AM4 random-CO2 ensemble, whose ice-model 6-hourly snapshot
output sits on the native 0.25° tripolar tracer grid. Same operational
pattern as `scripts/era5/` (xarray_beam on Google Cloud Dataflow,
DirectRunner for local subset runs).

One invocation — one config plus one `--member` — writes one templated,
sharded zarr v3 store. The nine members differ only in a name inside URLs,
so there is one config carrying a `{member}` placeholder rather than nine
near-identical files, and `randco2.mk` holds the member list
(`RANDCO2_MEMBERS`).

Each output store carries exactly five variables on the target grid:

| Variable | Meaning |
| --- | --- |
| `SST` | sea surface temperature, °C, wetmask-normalized regrid of the source `SST` |
| `sst` | the same field in K (`SST + 273.15`) |
| `sea_ice_fraction` | ice area per total cell area |
| `ocean_sea_ice_fraction` | ice area per ocean area |
| `sea_surface_fraction` | time-invariant ocean fraction of each target cell |

The source `sea_ice_fraction` is ice area per native cell area, and native
tracer cells are wholly wet or wholly dry, so on the source grid the
full-cell and ocean-relative quantities coincide. They separate on the target
grid, where a coastal cell mixes ocean and land source area: the full-cell
regrid keeps the source name and the wetmask-normalized one is renamed. The
two are redundant by construction (`sea_ice_fraction` =
`ocean_sea_ice_fraction` × `sea_surface_fraction`), and the
`sea_ice_fraction_consistency` postprocess asserts that per chunk.

Contents, under `pipeline/randco2/` (the shared core above supplies the
grid, weights loader, regridding, config loading and zarr I/O):

- `pipeline/randco2/run.py`: the pipeline itself — opens the configured
  stream, builds the output template (the static ocean fraction stamped in
  and written by the driver), and runs one beam branch through the per-chunk
  transform into the output store, read as in
  [Read strategy](#read-strategy). Every output variable carries
  `source_store`/`source_variable` (and, for derived variables,
  `derivation`) provenance attrs.
- `pipeline/randco2/config.py`: YAML→dataclass configuration (dacite); the
  `{member}` substitution is `pipeline.config_io.load_yaml`'s.
- `pipeline/randco2/postprocess.py`: named post-regrid transforms — Kelvin
  `sst` and the sea-ice-fraction consistency assertion.
- `pipeline/randco2/check_output.py`: post-run assertion that a store opens
  with the variable set the config implies.
- `pipeline/randco2/check_wetmask_equivalence.py`: assertion that all nine
  members derive the same ocean wetmask.

The regridding weights are the published OM4 tripolar artifact
(`pipeline/weights.py` loader only): these sources sit on the same native
tracer grid, so no weights are generated for them.

The source fields are 2D surface quantities on the tracer grid, so nothing
here splits vertical levels, interpolates from staggered points, or rotates
vectors, and there is no statics source store: `sea_surface_fraction` is the
only time-invariant output, and the target grid's cell geometry comes from
the analytic grid constructor.

### Masking

Every time-varying field is NaN over land, and the NaN pattern is the same at
every timestep: the wetmask is the first timestep's NaN pattern of the
source's own `SST`, every chunk's valid-data footprint is asserted to equal
it before regridding, and the normalized regrid NaNs target cells with no
ocean source area. `sea_surface_fraction` is 0 over land rather than NaN, and
is the only variable exempt from the land-NaN convention.

### Setup

The conda env and worker image are the shared ones (`make
create_environment`, `make push_dataflow`). The regridding weights come
from the dated, immutable inputs prefix named in the config and are read
as published; there is no generation step.

### Running

#### Smoke tests

Each member has a local DirectRunner smoke test that runs a few timesteps
into a throwaway scratch store and checks the output opens with the expected
variable set:

```
make randco2_smoke_tests                                   # all nine, plus the checks below
make randco2_smoke_test_random-CO2-1xCO2-ic_0001            # or any single member
```

`make randco2_smoke_tests` additionally verifies that all nine members derive
identical wetmasks (`make randco2_check_wetmask_equivalence`; downstream training and
analysis assume the stores share one mask — a difference is a
stop-and-report finding about the sources, not something to conform around),
and that a repeat run against an existing output store refuses to initialize
into it (`make randco2_smoke_test_repeat_fails`), and that a source chunk straddling
an output shard boundary is written correctly (`make
randco2_smoke_test_shard_boundary`; the stream is read at the source's own time chunk
width, which need not divide the output shard, so the boundary case is not
reachable from a short run at the production shard size — the target shrinks
the shard instead of lengthening the run).

The nine members were written by one ensemble with identical chunking, so one
member's smoke-test wall clock is the budget for the rest; a member running
several times over it is a finding about that store rather than something to
wait out.

The pipeline can also be invoked directly, with any beam pipeline options
after the script's own arguments:

```
python -m pipeline.randco2.run \
    --config configs/cm4-like-am4-randco2-sea-surface-1deg.yaml \
    --member random-CO2-1xCO2-ic_0001 \
    --num-timesteps 6 --output-path <url> --runner DirectRunner
```

#### Production launch checklist

1. **Smoke test** — `make randco2_smoke_tests`, or the single-member target for the
   member being launched, against the exact config to be launched.
2. **Launch** — build and push the worker image, then launch on Google Cloud
   Dataflow. Each member's output path is the config's, with `{member}`
   filled; the run aborts if a store already exists there.

   ```
   make push_dataflow
   make randco2_dataflow_all                            # all nine jobs
   make randco2_dataflow_random-CO2-1xCO2-ic_0001       # or one member
   ```

   Unlike the flat dated store per config that `scripts/era5/` and the OM4
   ocean pipeline write, the nine stores here are subdirectories of one dated
   parent prefix, one per member.

   Expect a silent multi-minute setup phase before workers start: the driver
   builds the template (processing the first timestep) and writes its
   metadata serially — accepted behavior, not a hang.
3. **Inspect** — after each job completes, check its store:

   ```
   python -m pipeline.randco2.check_output \
       --config configs/cm4-like-am4-randco2-sea-surface-1deg.yaml \
       --member random-CO2-1xCO2-ic_0001
   ```

   and review the job's Dataflow console page for stage-level errors before
   consuming the output.

The 4° stores (`configs/cm4-like-am4-randco2-sea-surface-4deg.yaml`, target
grid `F22.5`) run through the same targets with that config:
`make randco2_4deg_smoke_tests` and `make randco2_4deg_dataflow_all`, or any
single-member target with `RANDCO2_CONFIG` set to it.

`make randco2_dataflow*` checks that `$(IMAGE_NAME)` exists in Artifact
Registry (`make check_dataflow_image`), then invokes `run-dataflow.sh
randco2` (shared core above).
