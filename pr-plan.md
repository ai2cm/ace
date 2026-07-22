# Add GFDL OM4/CM4 ocean dataset pipeline on xarray_beam + Dataflow

New self-contained pipeline in `scripts/gfdl_om4/`, modeled on `scripts/era5/`:
reads native-grid 0.25° tripolar OM4/CM4 zarrs, applies per-chunk transforms
(vector rotation, wetmask-normalized conservative regridding to a Gaussian
grid, level splitting, derived variables, masking), and writes one templated
sharded zarr v3 store per run, driven by a YAML config. First target: the
5-daily 1° ocean (Samudra) training store; the same machinery supports a
6-hourly sea-ice store and 4° outputs as follow-on configs.

Directory layout mirrors `scripts/era5/` (Makefile, Dockerfile,
environment/requirements, `pipeline/`, plus `configs/`).

---

## `scripts/gfdl_om4/pipeline/ocean_emulators_port.py` (new)

Utilities ported from the ai2cm fork of ocean_emulators so the pipeline has no
dependency on that repo. Only what the pre-coarsened sources still need — no
vertical coarsening (sources are already on the 19-level Samudra grid).

```python
def convert_supergrid(hgrid: xr.Dataset) -> xr.Dataset:
    """ocean_hgrid.nc supergrid -> tracer-cell centers, corners, and rotation angle."""

def interpolate_to_cell_centers(
    u: xr.DataArray, v: xr.DataArray, wetmask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """C-grid velocity points -> tracer centers, wetmask-normalized (NOT the
    legacy fillna(0), which biases coastal speeds low). Skipped if sources are
    verified already tracer-colocated."""

def rotate_vectors(
    u: xr.DataArray, v: xr.DataArray, angle: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate native grid-relative vector components to eastward/northward.
    Requires u, v, angle colocated (tracer centers)."""

def build_wetmask_regridder(
    source_grid: xr.Dataset, target_grid: xr.Dataset, weights_path: str
) -> xe.Regridder:
    """Conservative xESMF regridder loading precomputed weights."""

OCEAN_FRACTION_THRESHOLD = 0.0  # legacy na_thres=1 equivalent: any ocean overlap keeps the cell

def regrid_normalized(
    ds: xr.Dataset, regridder: xe.Regridder, wetmask: xr.DataArray
) -> xr.Dataset:
    """Wetmask-normalized conservative regrid: mask, regrid field and mask,
    divide by regridded mask, NaN where target ocean fraction <=
    OCEAN_FRACTION_THRESHOLD. Equivalent to legacy xESMF skipna/na_thres=1
    behavior, but explicit (required anyway: precomputed weights are raw
    conservative weights, so skipna renormalization is unavailable). The
    regridded ocean-fraction field is kept as an output static."""
```

Ported from the **ai2cm fork** of ocean_emulators (the on-disk m2lines checkout
lacks the fork's `wetmask_name` API — verify the fork's actual normalization
before porting). Known legacy defects fixed rather than ported (see PRD
"Legacy defects"): coastal `fillna(0)` velocity bias, land-zero dilution of
`sea_ice_fraction`, polar-radius cell areas, `sea_ice_volume` units label.

## `scripts/gfdl_om4/pipeline/grids.py` (new)

Analytic Gaussian target grids (port of the era5 `F<N>` grid code).

```python
GAUSSIAN_GRID_N = {"F22.5": 22.5, "F90": 90}

def make_target_grid(name: str) -> xr.Dataset:
    """Gaussian grid (true leggauss latitudes, era5 port) with lat/lon centers
    and bounds for xESMF, by name."""

def target_cell_areas(grid: xr.Dataset) -> xr.DataArray:
    """Exact cell areas from Gaussian quadrature weights, mean radius
    6371.0 km. Replaces legacy xe.util.cell_area(r=6356) (polar radius,
    ~0.5% low) used for the old store's areacello."""
```

## `scripts/gfdl_om4/pipeline/weights.py` (new)

One-time weight generation plus the worker-side cached loader. Weights are a
versioned GCS artifact so workers start fast and regridding is deterministic.

```python
def generate_weights(
    hgrid_url: str, target_grid_name: str, output_url: str
) -> None:
    """Setup entry point (`python -m ... weights`): compute conservative xESMF
    weights for the 0.25 deg tripolar source x named target grid and write to a
    versioned GCS path."""

_REGRIDDER_CACHE: dict[str, xe.Regridder]  # one regridder per worker process

def get_regridder(config: RegriddingConfig) -> xe.Regridder:
    """Load the weight artifact into a cached regridder (era5 cache pattern)."""
```

## `scripts/gfdl_om4/pipeline/config.py` (new)

YAML -> dacite -> dataclasses, following `scripts/data_process` conventions.
Streams are data, not hard-coded flow: a new simulation or the future 6-hourly
sea-ice store is a config change.

```python
@dataclasses.dataclass
class VariableConfig:
    source_name: str
    output_name: str | None = None      # rename (e.g. ice SSH collision)
    rotate_with: str | None = None      # partner var for vector rotation

@dataclasses.dataclass
class StreamConfig:
    name: str                           # for logging / beam stage labels
    source_url: str
    variables: list[VariableConfig]
    time_subsample: int = 1             # e.g. 20 = every 20th 6-hourly step
    dim_renames: Mapping[str, str] = ...    # ice xT/yT/xB/yB -> xh/yh/xq/yq
    sea_ice_conventions: bool = False   # NaN->0 pre-regrid, zero thickness where no ice
    process_time_chunksize: int = 1

@dataclasses.dataclass
class RegriddingConfig:
    target_grid: str                    # "F90" | "F22.5"
    weights_url: str                    # versioned GCS weight artifact
    hgrid_url: str                      # GCS mirror of ocean_hgrid.nc

@dataclasses.dataclass
class PipelineConfig:
    output_url: str
    streams: list[StreamConfig]
    statics: StreamConfig               # written eagerly via the template
    regridding: RegriddingConfig
    n_levels: int = 19
    shift_means_to_midpoint: bool = False  # legacy convention; off in all planned configs
    output_time_chunksize: int = 1
    output_time_shardsize: int = 365    # multiple of the 73-step year
    start_time: str | None = None       # subset controls for DirectRunner runs
    end_time: str | None = None

    def __post_init__(self) -> None:
        """Validate chunk/shard divisibility, unique output names, known grid."""

def load_config(path: str) -> PipelineConfig: ...
```

## `scripts/gfdl_om4/pipeline/transforms.py` (new)

Per-chunk transforms applied inside beam stages; each returns provenance attrs
(`source_store`, `source_variable`, and a `derivation` description for derived
fields).

```python
def split_levels(ds: xr.Dataset, n_levels: int) -> xr.Dataset:
    """3D fields -> name_0..name_18 2D fields."""

def apply_land_nan(ds: xr.Dataset, wetmask: xr.DataArray) -> xr.Dataset:
    """NaN over land for all fields except masks / idepth_*."""

def apply_sea_ice_conventions(ds: xr.Dataset, wetmask_2d: xr.DataArray) -> xr.Dataset:
    """Two ice-fraction outputs: sea_ice_fraction = full-cell fraction
    (legacy convention: NaN->0 everywhere incl. land, plain conservative
    regrid, land-NaN post) and ocean_sea_ice_fraction = ice/ocean-area
    (NaN->0 over ocean only, wetmask-normalized regrid). Identity
    sea_ice_fraction ~= ocean_sea_ice_fraction * ocean_fraction asserted.
    Thickness zeroed where no ice."""

def derive_sea_ice_volume(ds: xr.Dataset, areacello: xr.DataArray) -> xr.DataArray:
    """thickness * fraction * area, with units attr matching the divisor
    (legacy store labels km^3 values as "1000 km^3" — do not reproduce)."""
def derive_sst_kelvin(ds: xr.Dataset) -> xr.DataArray: ...

def derive_hfds_total_area(ds: xr.Dataset) -> xr.DataArray:
    """hfds * sea_surface_fraction (full-cell flux; wetmask-normalized hfds is
    per-ocean-area). Hoisted from create_coupled_datasets.py, which derives it
    downstream today. (ocean_sea_ice_fraction is likewise hoisted — see
    apply_sea_ice_conventions. The SST-threshold sea-ice masks are NOT: they
    need a full-run time-mean, which doesn't fit per-chunk processing.)"""

def process_stream_chunk(
    key: xbeam.Key, ds: xr.Dataset, *, stream: StreamConfig,
    regridding: RegriddingConfig, wetmask: xr.DataArray, n_levels: int
) -> tuple[xbeam.Key, xr.Dataset]:
    """The beam MapTuple body: dim renames -> C-grid->tracer interpolation
    (if needed) -> rotation -> pre-regrid masking -> regrid_normalized ->
    level split -> derived vars -> land-NaN -> renames -> provenance attrs."""
```

## `scripts/gfdl_om4/pipeline/xr_beam_pipeline.py` (new)

Driver, following the era5 pattern: eager one-timestep template build, statics
stamped into the template and written by the driver, each stream writing its
disjoint variable set through ConsolidateChunks -> ChunksToZarr (zarr v3 with
shards).

```python
def open_stream(stream: StreamConfig, time_slice: slice | None) -> xr.Dataset:
    """Open source zarr via obstore, select vars, subsample."""

def derive_wetmask(source: xr.Dataset) -> xr.DataArray:
    """3D wetmask from the 19-level data (NaN pattern or deptho — resolved at
    implementation time against the old store's per-level masks)."""

def build_template(
    config: PipelineConfig, streams: Mapping[str, xr.Dataset],
    statics_regridded: xr.Dataset, output_time: pd.DatetimeIndex
) -> xr.Dataset:
    """Process one timestep per stream eagerly; expand to full time coordinate;
    stamp statics; set store-level history attr listing all input URLs."""

def validate_time_alignment(streams: Mapping[str, xr.Dataset]) -> None:
    """Assert exact time-coordinate equality across streams and
    snapshot-instant / mean-interval-end coincidence. Fail fast."""

def main() -> None:
    """CLI: config path + optional --start_time/--end_time/--output_url
    overrides + beam pipeline args. Stage-labeled one-line logging (config
    summary, template build, per-stream chunk counts, weight load, write
    completion)."""
```

### Critical detail — masking and naming conventions

- Wetmask-normalized conservative regridding for every field; NaN over land
  except masks/`idepth_*` (exemptions by explicit list, not the legacy
  substring match). Per-variable valid-data footprints asserted against the
  wetmask — the old code silently imposed `thetao`'s NaN pattern on
  everything.
- Legacy defects deliberately fixed, not reproduced (PRD "Legacy defects"):
  `sea_ice_volume` units label (10^3 off), polar-radius `areacello` (~0.5%
  low), coastal velocity `fillna(0)` bias. Validation against the old store
  must expect these diffs. `sea_ice_fraction` keeps the legacy full-cell
  semantics; a new `ocean_sea_ice_fraction` (ice/ocean-area) is added
  alongside it.
- Physically distinct fields are never merged: `tos` and ice-model `SST`,
  `so_0` and `SSS`, and all three SSH variants (`zos`, `SSH`, renamed ice SSH)
  are kept under distinct names.
- Snapshot streams keep raw timestamps verbatim; the store has a single time
  coordinate (the snapshot instants, coinciding with the means'
  end-of-interval labels). Ice fields enter at the snapshot instants by
  subsampling every 20th 6-hourly step — no 6-hourly -> 5-daily time
  coarsening anywhere.

## `scripts/gfdl_om4/configs/piControl-ocean-5daily-1deg.yaml` (new)

First production config: 5-daily 1° (`F90`) ocean store from the piControl
sample year — snapshot ocean state, 5-day-mean surface fluxes, ice fields at
snapshot instants, and statics (per-level masks, `idepth_*`, `areacello`,
`deptho`, `sea_surface_fraction`, and other regriddable `ocean_static`
fields).

## `scripts/gfdl_om4/` operational files (new)

`Makefile` (create_environment, build/push image, weight-generation target,
DirectRunner test-run target, Dataflow production target), `Dockerfile`,
`environment.yaml`, `dataflow-requirements.txt`, `pipeline/run-dataflow.sh`,
`README.md` — same operational pattern as `scripts/era5/`.

---

## Tests

No pytest suite (matching `scripts/era5/`). Verification is built in:

- Inline assertions on load-bearing assumptions (time alignment across
  streams, chunk/shard divisibility, expected variables and level counts,
  wetmask consistency) that fail fast and loudly.
- A Makefile DirectRunner test-run target exercising the full pipeline on a
  few timesteps before any Dataflow launch.
- Systematic validation of the produced store against the previous processed
  store is follow-on work outside this PR.

---

## Open Questions

- Weight artifact format: xESMF native weight NetCDF vs a plain sparse-matrix
  zarr — any preference for long-term stability?
- Per-stream `process_time_chunksize` defaults: 3D snapshot fields likely want
  per-timestep chunks while 2D flux/ice fields can batch several — tune during
  implementation or pin in config from the start?
