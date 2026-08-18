# CM4-like-AM4 random-CO2 sea-surface dataset pipeline

Produces 1° sea-surface training-reference stores from the nine members of
the CM4-like-AM4 random-CO2 ensemble, whose ice-model 6-hourly snapshot
output sits on the native 0.25° tripolar tracer grid. Same operational
pattern as `scripts/era5/` (xarray_beam on Google Cloud Dataflow,
DirectRunner for local subset runs).

One invocation — one config plus one `--member` — writes one templated,
sharded zarr v3 store. The nine members differ only in a name inside URLs,
so there is one config carrying a `{member}` placeholder rather than nine
near-identical files, and the Makefile holds the member list.

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

Contents:

- `pipeline/run.py`: the pipeline itself — opens the configured stream,
  builds the output template (the static ocean fraction stamped in and
  written by the driver), and runs one beam branch through the per-chunk
  transform into the output store. Every output variable carries
  `source_store`/`source_variable` (and, for derived variables,
  `derivation`) provenance attrs.
- `pipeline/config.py`: YAML→dataclass configuration (dacite), including the
  `{member}` substitution.
- `pipeline/postprocess.py`: named post-regrid transforms — Kelvin `sst` and
  the sea-ice-fraction consistency assertion.
- `pipeline/grids.py`: analytic Gaussian target grids (`F90` = 1°) with exact
  quadrature-weight cell areas.
- `pipeline/ocean_emulators_port.py`: wetmask-normalized conservative
  regridding, ported from the ai2cm fork of
  [ocean_emulators](https://github.com/ai2cm/ocean_emulators).
- `pipeline/weights.py`: loader for the published conservative weight
  artifact, with the per-process regridder cache workers use. These sources
  sit on the same native tracer grid the OM4 tripolar artifact was built
  from, so no weights are generated here.
- `pipeline/check_output.py`: post-run assertion that a store opens with the
  variable set the config implies.
- `pipeline/check_wetmask_equivalence.py`: assertion that all nine members
  derive the same ocean wetmask.

The source fields are 2D surface quantities on the tracer grid, so nothing
here splits vertical levels, interpolates from staggered points, or rotates
vectors, and there is no statics source store: `sea_surface_fraction` is the
only time-invariant output, and the target grid's cell geometry comes from
the analytic grid constructor.

## Masking

Every time-varying field is NaN over land, and the NaN pattern is the same at
every timestep: the wetmask is the first timestep's NaN pattern of the
source's own `SST`, every chunk's valid-data footprint is asserted to equal
it before regridding, and the normalized regrid NaNs target cells with no
ocean source area. `sea_surface_fraction` is 0 over land rather than NaN, and
is the only variable exempt from the land-NaN convention.

## Setup

```
make create_environment      # conda env cm4-randco2-ingestion
```

The regridding weights come from the dated, immutable inputs prefix named in
the config and are read as published; there is no generation step.

## Running

### Smoke tests

Each member has a local DirectRunner smoke test that runs a few timesteps
into a throwaway scratch store and checks the output opens with the expected
variable set:

```
make smoke_tests                                   # all nine, plus the checks below
make smoke_test_random-CO2-1xCO2-ic_0001            # or any single member
```

`make smoke_tests` additionally verifies that all nine members derive
identical wetmasks (`make check_wetmask_equivalence`; downstream training and
analysis assume the stores share one mask — a difference is a
stop-and-report finding about the sources, not something to conform around),
and that a repeat run against an existing output store refuses to initialize
into it (`make smoke_test_repeat_fails`).

The nine members were written by one ensemble with identical chunking, so one
member's smoke-test wall clock is the budget for the rest; a member running
several times over it is a finding about that store rather than something to
wait out.

The pipeline can also be invoked directly, with any beam pipeline options
after the script's own arguments:

```
python -m pipeline.run \
    --config configs/cm4-like-am4-randco2-sea-surface-1deg.yaml \
    --member random-CO2-1xCO2-ic_0001 \
    --num-timesteps 6 --output-path <url> --runner DirectRunner
```

### Production launch checklist

1. **Smoke test** — `make smoke_tests`, or the single-member target for the
   member being launched, against the exact config to be launched.
2. **Launch** — build and push the worker image, then launch on Google Cloud
   Dataflow. Each member's output path is the config's, with `{member}`
   filled; the run aborts if a store already exists there.

   ```
   make push_dataflow
   make dataflow_all                            # all nine jobs
   make dataflow_random-CO2-1xCO2-ic_0001       # or one member
   ```

   Unlike the flat dated store per config that `scripts/era5/` and the OM4
   ocean pipeline write, the nine stores here are subdirectories of one dated
   parent prefix, one per member.

   Expect a silent multi-minute setup phase before workers start: the driver
   builds the template (processing the first timestep) and writes its
   metadata serially — accepted behavior, not a hang.
3. **Inspect** — after each job completes, check its store:

   ```
   python -m pipeline.check_output \
       --config configs/cm4-like-am4-randco2-sea-surface-1deg.yaml \
       --member random-CO2-1xCO2-ic_0001
   ```

   and review the job's Dataflow console page for stage-level errors before
   consuming the output.

`make dataflow*` invokes `run-dataflow.sh`, which supplies the Dataflow
resource flags (project, region, temp location, worker shape, container
image). Unlike the era5 pipeline, the pipeline code here is a package, so the
worker image copies `pipeline/` in and puts it on `PYTHONPATH` rather than
relying on `--save_main_session`. Workers authenticate to GCS with the
project service account; no S3/OSN credentials are involved.
