"""xarray_beam pipeline producing a training-reference ocean dataset.

Reads native-grid 0.25 degree tripolar OM4/CM4 output stores, applies
per-chunk transforms (C-grid to tracer-center interpolation, vector rotation,
wetmask-normalized conservative regridding to a Gaussian grid, level
splitting), and writes one templated, sharded zarr v3 store per invocation,
driven by a YAML config (see pipeline/config.py and configs/).

Run locally on a subset with the DirectRunner (see the Makefile test_run
target), or on Google Cloud Dataflow by passing the corresponding beam
pipeline options after the script's own arguments.

Masking conventions of the output store:

- Every time-varying field is NaN over land. This falls out of the
  wetmask-normalized regrid: target cells with ocean fraction <=
  OCEAN_FRACTION_THRESHOLD are NaN.
- Per-level binary masks ``mask_0..N`` (and ``mask_2d``, equal to
  ``mask_0``) mark the regridded wetmask footprint: 1 where a target cell
  overlaps any ocean source area, else 0. The wetmask is taken at the
  reference (first) timestep; instantaneous ocean coverage drifts from it
  by a handful of sub-surface cells (see MAX_FOOTPRINT_DRIFT_FRACTION), so
  a cell with ``mask_k == 1`` can occasionally be NaN at a given time.
- ``sea_surface_fraction`` is the regridded surface ocean fraction (0 over
  land, not NaN), usable to weight coastal cells.
- Variables listed in a stream's ``full_cell_variables`` (e.g.
  ``sea_ice_fraction``) are regridded with full-cell semantics — the value
  is per total cell area, land counted as zero — with land-NaN applied
  after; their wetmask-normalized (per-ocean-area) twins are written under
  renamed outputs (e.g. ``ocean_sea_ice_fraction``).
- Masks, ``idepth_*``, ``areacello``, and ``sea_surface_fraction`` are
  exempt from land-NaN by explicit list (see land_nan_exempt_names).
"""

import argparse
import logging

import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam
from apache_beam.options.pipeline_options import PipelineOptions
from obstore.store import from_url
from zarr.storage import ObjectStore

from .config import PipelineConfig, StreamConfig, load_config
from .grids import make_target_grid
from .ocean_emulators_port import (
    OCEAN_FRACTION_THRESHOLD,
    interpolate_to_cell_centers,
    regrid_normalized,
    rotate_vectors,
)
from .postprocess import DERIVATION_ATTR, POSTPROCESS, ChunkContext, provenance_attrs
from .weights import get_regridder, open_source_grid

logger = logging.getLogger(__name__)

TIME_DIM = "time"
LEVEL_DIM = "z_l"
OUTPUT_DTYPE = np.float32

# Tracer-center dimension for each staggered (right/north-edge) dimension.
STAGGERED_TO_TRACER_DIM = {"xq": "xh", "yq": "yh"}


def land_nan_exempt_names(level_count: int) -> list[str]:
    """Output variables exempt from the NaN-over-land convention.

    An explicit list rather than a name-pattern match, so that a new
    variable whose name happens to contain e.g. "mask_" is still masked.
    """
    return (
        [f"mask_{k}" for k in range(level_count)]
        + ["mask_2d"]
        + [f"idepth_{k}" for k in range(level_count + 1)]
        + ["areacello", "sea_surface_fraction"]
    )


def _make_zarr_store(url: str, read_only: bool = True):
    """Create a zarr store from a URL using obstore. If local, return the path."""
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


# ---------------------------------------------------------------------------
# Source opening and load-bearing assertions
# ---------------------------------------------------------------------------


def open_stream(stream: StreamConfig, config: PipelineConfig) -> xr.Dataset:
    """Open a stream's variables lazily, time-subset, with fail-fast checks."""
    ds = xr.open_zarr(
        _make_zarr_store(stream.store), chunks=None, decode_timedelta=False
    )
    missing = set(stream.variables) - set(ds.data_vars)
    if missing:
        raise AssertionError(
            f"stream {stream.name!r}: variables missing from {stream.store}: "
            f"{sorted(missing)}"
        )
    ds = ds[stream.variables]
    if stream.dim_renaming:
        ds = ds.rename(stream.dim_renaming)
        for staggered, tracer in STAGGERED_TO_TRACER_DIM.items():
            if (
                staggered in ds.dims
                and tracer in ds.dims
                and ds.sizes[staggered] == ds.sizes[tracer] + 1
            ):
                # Symmetric staggering carries both edges; drop the first
                # point to match the right/north-edge convention of the
                # shared interpolation and rotation machinery.
                ds = ds.isel({staggered: slice(1, None)})
    if stream.time_subsample_stride is not None:
        # Keep the last instant of each stride-length block, anchored to the
        # source store's raw time grid (hence before any time-range subset),
        # so block ends land on the shared snapshot instants; the
        # cross-stream time-alignment assertion enforces the coincidence.
        stride = stream.time_subsample_stride
        ds = ds.isel({TIME_DIM: slice(stride - 1, None, stride)})
    if config.start_time is not None or config.end_time is not None:
        ds = ds.sel({TIME_DIM: slice(config.start_time, config.end_time)})
    if ds.sizes[TIME_DIM] == 0:
        raise AssertionError(
            f"stream {stream.name!r}: no timesteps in "
            f"[{config.start_time}, {config.end_time}]"
        )
    for name, da in ds.data_vars.items():
        if LEVEL_DIM in da.dims and da.sizes[LEVEL_DIM] != config.expected_level_count:
            raise AssertionError(
                f"stream {stream.name!r}: {name} has {da.sizes[LEVEL_DIM]} "
                f"levels; expected {config.expected_level_count}"
            )
    return ds


def load_wetmask(config: PipelineConfig) -> xr.DataArray:
    """The 3D ocean wetmask: the NaN pattern of the reference variable's
    first timestep (True over ocean)."""
    ds = xr.open_zarr(
        _make_zarr_store(config.wetmask.store), chunks=None, decode_timedelta=False
    )
    if config.wetmask.variable not in ds.data_vars:
        raise AssertionError(
            f"wetmask variable {config.wetmask.variable!r} missing from "
            f"{config.wetmask.store}"
        )
    da = ds[config.wetmask.variable].isel({TIME_DIM: 0}).load()
    if LEVEL_DIM not in da.dims:
        raise AssertionError(
            f"wetmask variable {config.wetmask.variable!r} has no {LEVEL_DIM} "
            "dimension"
        )
    if da.sizes[LEVEL_DIM] != config.expected_level_count:
        raise AssertionError(
            f"wetmask variable has {da.sizes[LEVEL_DIM]} levels; expected "
            f"{config.expected_level_count}"
        )
    wetmask = da.notnull().reset_coords(drop=True)
    wetmask.attrs = {}
    return wetmask


# Instantaneous ocean coverage may drift slightly from the reference-time
# wetmask (cells with time-varying validity at sub-surface levels); observed
# drift is ~0.01% of wet cells. Anything larger indicates an inconsistent
# source (wrong grid, dropped mask) and fails the run.
MAX_FOOTPRINT_DRIFT_FRACTION = 0.001


def _assert_footprint(da: xr.DataArray, footprint: xr.DataArray, context: str) -> None:
    """Assert a variable's valid-data footprint exactly equals the chunk's
    shared footprint.

    Guards against a source variable whose land pattern disagrees with the
    others', which the normalized regrid would otherwise silently average
    as zeros.
    """
    valid, expected = xr.broadcast(da.notnull(), footprint)
    mismatches = int((valid != expected).sum())
    if mismatches:
        raise AssertionError(
            f"{context}: valid-data footprint of {da.name!r} differs from the "
            f"chunk footprint at {mismatches} cells"
        )


def _assert_footprint_drift(
    footprint: xr.DataArray, wetmask: xr.DataArray, context: str
) -> None:
    """Assert the chunk's instantaneous footprint stays within
    MAX_FOOTPRINT_DRIFT_FRACTION of the reference-time wetmask."""
    drifted, expected = xr.broadcast(footprint, wetmask)
    drift = int((drifted != expected).sum())
    limit = MAX_FOOTPRINT_DRIFT_FRACTION * max(int(expected.sum()), 1)
    if drift > limit:
        raise AssertionError(
            f"{context}: instantaneous footprint differs from the reference "
            f"wetmask at {drift} cells (limit {limit:.0f})"
        )


def _assert_time_alignment(datasets: dict[str, xr.Dataset]) -> xr.DataArray:
    """Assert all streams share an identical time coordinate; return it."""
    names = list(datasets)
    reference = datasets[names[0]][TIME_DIM]
    for name in names[1:]:
        other = datasets[name][TIME_DIM]
        if (
            reference.sizes[TIME_DIM] != other.sizes[TIME_DIM]
            or not (reference.values == other.values).all()
        ):
            raise AssertionError(
                f"time coordinate of stream {name!r} differs from stream "
                f"{names[0]!r}"
            )
    return reference


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------

# One rotation-angle array per weight artifact per worker process.
_ANGLE_CACHE: dict[str, xr.DataArray] = {}


def _get_angle(weights_url: str) -> xr.DataArray:
    if weights_url not in _ANGLE_CACHE:
        _ANGLE_CACHE[weights_url] = open_source_grid(weights_url)["angle"].load()
    return _ANGLE_CACHE[weights_url]


# One exact cell-area array per target grid per worker process.
_AREACELLO_CACHE: dict[str, xr.DataArray] = {}


def _get_areacello(target_grid_name: str) -> xr.DataArray:
    if target_grid_name not in _AREACELLO_CACHE:
        _AREACELLO_CACHE[target_grid_name] = make_target_grid(target_grid_name)[
            "areacello"
        ]
    return _AREACELLO_CACHE[target_grid_name]


def _rotate_pairs(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
) -> xr.Dataset:
    """Interpolate each configured C-grid vector pair to tracer centers and
    rotate it to geographic components, preserving names and attrs.

    Ocean cells whose staggered neighbors are all invalid (some vector
    fields, e.g. surface stresses, carry a stricter staggered-point mask
    than the tracer wetmask at a small number of coastal cells) are set to
    zero, so the pair keeps the chunk's shared valid-data footprint.
    """
    pairs = [pair for pair in stream.rotated_pairs if pair[0] in ds.data_vars]
    if not pairs:
        return ds
    angle = _get_angle(weights_url)
    for u_name, v_name in pairs:
        u_attrs, v_attrs = dict(ds[u_name].attrs), dict(ds[v_name].attrs)
        u, v = interpolate_to_cell_centers(ds[u_name], ds[v_name], wetmask)
        u, v = rotate_vectors(u, v, angle)
        u = u.fillna(0.0).where(wetmask)
        v = v.fillna(0.0).where(wetmask)
        rotation_note = (
            "interpolated from C-grid points to tracer centers "
            "(normalizing by valid ocean neighbors; ocean cells with no "
            "valid neighbor set to 0) and rotated from grid-relative to "
            "geographic components"
        )
        ds = ds.drop_vars([u_name, v_name])
        ds[u_name] = u.assign_attrs(u_attrs, **{DERIVATION_ATTR: rotation_note})
        ds[v_name] = v.assign_attrs(v_attrs, **{DERIVATION_ATTR: rotation_note})
    return ds


def _process_chunk(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
    target_grid_name: str,
    level_index: int | None,
) -> xr.Dataset:
    """Transform one in-memory chunk: rotate, check footprints, regrid, and
    (for 3D chunks) split the level into suffixed 2D variables.

    The regrid is normalized by the chunk's own instantaneous footprint
    (the shared NaN pattern of its tracer-centered variables), not the
    reference-time wetmask, because ocean coverage varies slightly in time;
    every variable is asserted to share that footprint exactly, and the
    footprint is asserted to stay close to the reference wetmask.
    """
    context = f"stream {stream.name!r}"
    if level_index is not None:
        context += f" level {level_index}"

    tracer_names = [
        name for name in ds.data_vars if set(wetmask.dims).issubset(set(ds[name].dims))
    ]
    if tracer_names:
        footprint = ds[tracer_names[0]].notnull().reset_coords(drop=True)
        footprint.attrs = {}
    else:
        footprint = wetmask

    ds = _rotate_pairs(ds, stream, footprint, weights_url)
    for name in ds.data_vars:
        _assert_footprint(ds[name], footprint, context)
    _assert_footprint_drift(footprint, wetmask, context)

    regridder = get_regridder(weights_url, target_grid_name)
    regridded, ocean_fraction = regrid_normalized(ds, regridder, footprint)

    output = xr.Dataset()
    for name, da in regridded.data_vars.items():
        out_name = stream.renaming.get(name, name)
        derivation = da.attrs.get(DERIVATION_ATTR)
        if level_index is not None:
            depth = float(ds[LEVEL_DIM].item())
            level_note = f"level {level_index} (depth {depth:g} m) of {name}"
            derivation = f"{level_note}; {derivation}" if derivation else level_note
            da = da.squeeze(LEVEL_DIM, drop=True)
            out_name = f"{out_name}_{level_index}"
        output[out_name] = da.assign_attrs(
            provenance_attrs(stream.store, name, derivation)
        )

    for name in stream.full_cell_variables:
        if name not in ds.data_vars:
            continue
        full = regridder(ds[name].fillna(0.0), keep_attrs=True)
        full = full.where(ocean_fraction > OCEAN_FRACTION_THRESHOLD)
        output[name] = full.assign_attrs(
            provenance_attrs(
                stream.store,
                name,
                "NaN filled with 0 over the full grid (land included) and "
                "conservatively regridded without ocean-fraction "
                "normalization, giving the per-total-cell-area quantity; "
                "NaN over land applied after",
            )
        )

    if stream.postprocess:
        chunk_context = ChunkContext(
            ocean_fraction=ocean_fraction,
            areacello=_get_areacello(target_grid_name),
            store=stream.store,
        )
        for postprocess_name in stream.postprocess:
            spec = POSTPROCESS[postprocess_name]
            if all(v in output.data_vars for v in spec.requires):
                output = spec.fn(output, chunk_context)

    for name in output.data_vars:
        output[name] = output[name].astype(OUTPUT_DTYPE)
    return output


def process_chunk(
    key: xbeam.Key,
    ds: xr.Dataset,
    # beam.MapTuple requires keyword-only side inputs to carry defaults.
    stream: StreamConfig | None = None,
    wetmask: xr.DataArray | None = None,
    weights_url: str | None = None,
    target_grid_name: str | None = None,
) -> tuple[xbeam.Key, xr.Dataset]:
    """Beam entry point: process one (time, level) chunk of a stream."""
    assert stream is not None
    assert wetmask is not None
    assert weights_url is not None
    assert target_grid_name is not None
    level_index = key.offsets.get(LEVEL_DIM)
    if level_index is not None:
        wetmask = wetmask.isel({LEVEL_DIM: level_index}, drop=True)
    elif LEVEL_DIM in wetmask.dims:
        # 2D (surface) variables are checked and normalized against the
        # surface level of the wetmask.
        wetmask = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    output = _process_chunk(
        ds, stream, wetmask, weights_url, target_grid_name, level_index
    )
    new_key = xbeam.Key(
        {TIME_DIM: key.offsets[TIME_DIM], "lat": 0, "lon": 0},
        vars=frozenset(output.data_vars),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Statics
# ---------------------------------------------------------------------------


def _interface_depths(level_centers: np.ndarray) -> np.ndarray:
    """Interface depths recovered from level centers by the midpoint
    recurrence (interface k+1 = 2 * center k - interface k, starting at 0)."""
    interfaces = [0.0]
    for center in level_centers:
        interfaces.append(2 * float(center) - interfaces[-1])
    interfaces = np.array(interfaces)
    if not (np.diff(interfaces) > 0).all():
        raise AssertionError(
            "level centers are not midpoints of a valid interface grid: "
            f"recovered interfaces {interfaces} are not increasing"
        )
    return interfaces


def build_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Eagerly build all static output fields on the target grid."""
    regridder = get_regridder(config.weights_url, config.target_grid)
    statics = xr.Dataset()

    # Per-level ocean fraction and binary masks from the regridded wetmask.
    level_count = wetmask.sizes[LEVEL_DIM]
    surface_fraction = None
    for k in range(level_count):
        fraction = regridder(
            wetmask.isel({LEVEL_DIM: k}, drop=True).astype("float64"), keep_attrs=False
        ).fillna(0.0)
        if k == 0:
            surface_fraction = fraction
        mask = (fraction > OCEAN_FRACTION_THRESHOLD).astype(OUTPUT_DTYPE)
        statics[f"mask_{k}"] = mask.assign_attrs(
            long_name=f"ocean mask level-{k}",
            units="0 if land, 1 if ocean",
            **provenance_attrs(
                config.wetmask.store,
                config.wetmask.variable,
                f"1 where the regridded level-{k} wetmask (NaN pattern of "
                f"{config.wetmask.variable}) exceeds "
                f"{OCEAN_FRACTION_THRESHOLD:g}, else 0",
            ),
        )
    statics["mask_2d"] = statics["mask_0"].copy()

    assert surface_fraction is not None
    statics["sea_surface_fraction"] = surface_fraction.astype(
        OUTPUT_DTYPE
    ).assign_attrs(
        long_name="fraction of cell area overlapping ocean source cells",
        units="0-1",
        **provenance_attrs(
            config.wetmask.store,
            config.wetmask.variable,
            "conservative regrid of the surface wetmask (NaN pattern of "
            f"{config.wetmask.variable}); usable to weight coastal cells",
        ),
    )

    # Interface depths from the level-center coordinate.
    interfaces = _interface_depths(wetmask[LEVEL_DIM].values)
    for k, depth in enumerate(interfaces):
        statics[f"idepth_{k}"] = xr.DataArray(
            depth,
            attrs={
                "units": "meters",
                "long_name": f"Depth at interface level-{k}",
                **provenance_attrs(
                    config.wetmask.store,
                    LEVEL_DIM,
                    "interface depths recovered from level-center depths by "
                    "the midpoint recurrence",
                ),
            },
        )

    # Exact Gaussian-quadrature cell areas, not regridded from the source.
    target_grid = make_target_grid(config.target_grid)
    statics["areacello"] = (
        target_grid["areacello"]
        .astype(OUTPUT_DTYPE)
        .assign_attrs(
            provenance_attrs(
                "analytic",
                "areacello",
                f"exact {config.target_grid} Gaussian-quadrature cell areas "
                "(mean Earth radius 6371.0 km); differs by ~0.5% from stores "
                "whose areas used the polar radius",
            )
        )
    )

    # Regriddable static source fields, wetmask-normalized like the streams.
    source = xr.open_zarr(
        _make_zarr_store(config.statics.store), chunks=None, decode_timedelta=False
    )
    missing = set(config.statics.variables) - set(source.data_vars)
    if missing:
        raise AssertionError(
            f"static variables missing from {config.statics.store}: "
            f"{sorted(missing)}"
        )
    fields = source[config.statics.variables].load()
    surface_wetmask = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    regridded, _ = regrid_normalized(fields, regridder, surface_wetmask)
    for name, da in regridded.data_vars.items():
        statics[name] = da.astype(OUTPUT_DTYPE).assign_attrs(
            provenance_attrs(config.statics.store, name)
        )

    # Enforce the land-NaN convention on everything not explicitly exempt.
    exempt = set(land_nan_exempt_names(level_count))
    land = statics["mask_2d"] == 0
    for name in statics.data_vars:
        if name not in exempt and statics[name].ndim > 0:
            statics[name] = statics[name].where(~land)
    return statics


# ---------------------------------------------------------------------------
# Template and driver
# ---------------------------------------------------------------------------


def _split_stream(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Split a stream dataset into its 3D (level-bearing) and 2D parts."""
    names_3d = [name for name in ds.data_vars if LEVEL_DIM in ds[name].dims]
    names_2d = [name for name in ds.data_vars if LEVEL_DIM not in ds[name].dims]
    return ds[names_3d], ds[names_2d]


def build_template(
    config: PipelineConfig,
    stream_datasets: dict[str, xr.Dataset],
    statics: xr.Dataset,
    wetmask: xr.DataArray,
    output_time: xr.DataArray,
    input_urls: list[str],
) -> xr.Dataset:
    """Eagerly process the first timestep of every stream to build the
    output template, with statics stamped in for the driver to write."""
    logger.info("[template] processing first timestep of each stream")
    streams_by_name = {stream.name: stream for stream in config.streams}
    pieces = []
    for name, ds in stream_datasets.items():
        stream = streams_by_name[name]
        ds_3d, ds_2d = _split_stream(ds)
        first = {TIME_DIM: slice(0, 1)}
        if ds_2d.data_vars:
            key = xbeam.Key({TIME_DIM: 0})
            _, out = process_chunk(
                key,
                ds_2d.isel(first).load(),
                stream,
                wetmask,
                config.weights_url,
                config.target_grid,
            )
            pieces.append(out)
        for k in range(ds_3d.sizes.get(LEVEL_DIM, 0)):
            key = xbeam.Key({TIME_DIM: 0, LEVEL_DIM: k})
            _, out = process_chunk(
                key,
                ds_3d.isel({**first, LEVEL_DIM: slice(k, k + 1)}).load(),
                stream,
                wetmask,
                config.weights_url,
                config.target_grid,
            )
            pieces.append(out)
    merged = xr.merge(pieces).squeeze(TIME_DIM, drop=True)
    merged = merged.drop_encoding()

    template = xbeam.make_template(merged)
    template = template.expand_dims({TIME_DIM: output_time.values}, axis=0)
    template = template.assign_coords({TIME_DIM: output_time})

    # Statics are eager (numpy-backed), so ChunksToZarr writes them from the
    # driver when it sets up the store.
    template.update(statics.drop_encoding())
    template.attrs["history"] = (
        "Dataset computed by ace/scripts/gfdl_om4/pipeline, using the "
        f"following input sources: {input_urls}."
    )
    return template


def _expected_output_names(
    config: PipelineConfig, stream_datasets: dict[str, xr.Dataset]
) -> set[str]:
    names: set[str] = set()
    for stream in config.streams:
        ds = stream_datasets[stream.name]
        for name in ds.data_vars:
            out_name = stream.renaming.get(name, name)
            if LEVEL_DIM in ds[name].dims:
                names.update(f"{out_name}_{k}" for k in range(ds.sizes[LEVEL_DIM]))
            else:
                names.add(out_name)
        names.update(stream.full_cell_variables)
        for postprocess_name in stream.postprocess:
            names.update(POSTPROCESS[postprocess_name].adds)
    return names


def _shift_timestamps_to_midpoint(time: xr.DataArray) -> xr.DataArray:
    """Shift end-of-interval time labels back by half the (uniform) step."""
    steps = np.diff(time.values)
    if not (steps == steps[0]).all():
        raise AssertionError("cannot midpoint-shift a non-uniform time coordinate")
    shifted = time.copy(data=time.values - steps[0] / 2)
    shifted.attrs["long_name"] = "time, avg interval midpoint"
    return shifted


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce a training-reference ocean dataset store. "
        "Unrecognized arguments are passed to beam as pipeline options."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument(
        "--start-time",
        help="Override the config's inclusive time-range start (e.g. 0151-01-06)",
    )
    parser.add_argument(
        "--end-time", help="Override the config's inclusive time-range end"
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Process only the first N timesteps (for subset test runs)",
    )
    parser.add_argument("--output-path", help="Override the config's output path")
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()

    config = load_config(args.config)
    if args.start_time is not None:
        config.start_time = args.start_time
    if args.end_time is not None:
        config.end_time = args.end_time
    if args.output_path is not None:
        config.output.path = args.output_path

    logger.info(
        "[config] streams=%s target_grid=%s output=%s time_chunk=%d time_shard=%d",
        [stream.name for stream in config.streams],
        config.target_grid,
        config.output.path,
        config.output.time_chunk_size,
        config.output.time_shard_size,
    )

    logger.info(
        "[wetmask] deriving wetmask from NaN pattern of %s in %s",
        config.wetmask.variable,
        config.wetmask.store,
    )
    wetmask = load_wetmask(config)
    logger.info(
        "[wetmask] %d ocean cells at the surface, %d at the deepest level",
        int(wetmask.isel({LEVEL_DIM: 0}).sum()),
        int(wetmask.isel({LEVEL_DIM: -1}).sum()),
    )

    stream_datasets = {
        stream.name: open_stream(stream, config) for stream in config.streams
    }
    output_time = _assert_time_alignment(stream_datasets)
    if args.num_timesteps is not None:
        output_time = output_time.isel({TIME_DIM: slice(0, args.num_timesteps)})
        stream_datasets = {
            name: ds.isel({TIME_DIM: slice(0, args.num_timesteps)})
            for name, ds in stream_datasets.items()
        }
    if config.shift_timestamps_to_avg_interval_midpoint:
        output_time = _shift_timestamps_to_midpoint(output_time)
        stream_datasets = {
            name: ds.assign_coords({TIME_DIM: output_time.values})
            for name, ds in stream_datasets.items()
        }
    logger.info(
        "[config] %d timesteps: %s .. %s",
        output_time.sizes[TIME_DIM],
        output_time.values[0],
        output_time.values[-1],
    )

    logger.info("[weights] loading weight artifact from %s", config.weights_url)
    get_regridder(config.weights_url, config.target_grid)

    logger.info("[statics] building static fields")
    statics = build_statics(config, wetmask)

    input_urls = sorted(
        {stream.store for stream in config.streams}
        | {config.statics.store, config.wetmask.store, config.weights_url}
    )
    template = build_template(
        config, stream_datasets, statics, wetmask, output_time, input_urls
    )
    expected = _expected_output_names(config, stream_datasets) | set(statics.data_vars)
    if expected != set(template.data_vars):
        raise AssertionError(
            "template variables disagree with the expected output set; "
            f"missing={sorted(expected - set(template.data_vars))} "
            f"unexpected={sorted(set(template.data_vars) - expected)}"
        )
    expected_coords = {TIME_DIM, "lat", "lon"}
    if set(template.coords) != expected_coords:
        raise AssertionError(
            "unexpected coordinates leaked into the output template: "
            f"{sorted(set(template.coords) - expected_coords)}"
        )
    logger.info("[template] %d output variables", len(template.data_vars))

    output_chunks = {TIME_DIM: config.output.time_chunk_size}
    output_shards = {TIME_DIM: config.output.time_shard_size}
    output_store = _make_zarr_store(config.output.path, read_only=False)

    streams_by_name = {stream.name: stream for stream in config.streams}
    logger.info("[pipeline] starting; writing to %s", config.output.path)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        for name, ds in stream_datasets.items():
            stream = streams_by_name[name]
            branches = []
            ds_3d, ds_2d = _split_stream(ds)
            if ds_3d.data_vars:
                branches.append((f"{name}_3d", ds_3d, {TIME_DIM: 1, LEVEL_DIM: 1}))
            if ds_2d.data_vars:
                branches.append((f"{name}_2d", ds_2d, {TIME_DIM: 1}))
            for label, branch_ds, chunks in branches:
                n_chunks = int(
                    np.prod(
                        [branch_ds.sizes[dim] // size for dim, size in chunks.items()]
                    )
                )
                logger.info(
                    "[stream:%s] %d variables in %d chunks",
                    label,
                    len(branch_ds.data_vars),
                    n_chunks,
                )
                (
                    p
                    | f"{label}_to_chunks"
                    >> xbeam.DatasetToChunks(branch_ds, chunks=chunks)
                    | f"{label}_process"
                    >> beam.MapTuple(
                        process_chunk,
                        stream=stream,
                        wetmask=wetmask,
                        weights_url=config.weights_url,
                        target_grid_name=config.target_grid,
                    )
                    | f"{label}_consolidate" >> xbeam.ConsolidateChunks(output_shards)
                    | f"{label}_to_zarr"
                    >> xbeam.ChunksToZarr(
                        output_store,
                        template,
                        zarr_chunks=output_chunks,
                        zarr_shards=output_shards,
                        zarr_format=3,
                    )
                )
    logger.info("[write] pipeline complete: %s", config.output.path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # apache_beam may have already configured the root logger on import,
    # making basicConfig a no-op; raise the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    main()
