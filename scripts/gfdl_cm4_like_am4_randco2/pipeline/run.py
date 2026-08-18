"""xarray_beam pipeline producing a 1-degree sea-surface dataset.

Reads a native-grid 0.25 degree tripolar ice-model snapshot store, applies
wetmask-normalized conservative regridding to a Gaussian grid per chunk, and
writes one templated, sharded zarr v3 store per invocation, driven by a YAML
config and an ensemble-member name (see pipeline/config.py and configs/).

Run locally on a subset with the DirectRunner (see the Makefile smoke_test
target), or on Google Cloud Dataflow by passing the corresponding beam
pipeline options after the script's own arguments.

The source fields are two-dimensional surface quantities on the tracer grid,
so nothing here splits vertical levels, interpolates from staggered points,
or rotates vectors, and there is no statics store: the only time-invariant
output is the regridded ocean fraction.

Masking conventions of the output store:

- Every time-varying field is NaN over land. This falls out of the
  wetmask-normalized regrid: target cells with ocean fraction <=
  OCEAN_FRACTION_THRESHOLD are NaN. The NaN pattern is the same at every
  timestep because each chunk's valid-data footprint is asserted to equal
  the reference-time wetmask before regridding (see _assert_footprint).
- ``sea_surface_fraction`` is the regridded ocean fraction (0 over land, not
  NaN), usable to weight coastal cells, and is exempt from land-NaN by
  explicit list (see land_nan_exempt_names).
- Variables listed in the stream's ``full_cell_variables`` (here
  ``sea_ice_fraction``) are regridded with full-cell semantics — the value
  is per total cell area, land counted as zero — with land-NaN applied
  after; their wetmask-normalized (per-ocean-area) twins are written under
  renamed outputs (``ocean_sea_ice_fraction``).
"""

import argparse
import logging
import math

import apache_beam as beam
import fsspec
import numpy as np
import xarray as xr
import xarray_beam as xbeam
from apache_beam.options.pipeline_options import PipelineOptions
from obstore.store import from_url
from zarr.storage import ObjectStore

from .config import PipelineConfig, StreamConfig, load_config
from .ocean_emulators_port import OCEAN_FRACTION_THRESHOLD, regrid_normalized
from .postprocess import DERIVATION_ATTR, POSTPROCESS, ChunkContext, provenance_attrs
from .weights import get_regridder

logger = logging.getLogger(__name__)

TIME_DIM = "time"
OUTPUT_DTYPE = np.float32


def land_nan_exempt_names() -> list[str]:
    """Output variables exempt from the NaN-over-land convention.

    An explicit list rather than a name-pattern match, so a new variable
    whose name happens to resemble one of these is still masked.
    """
    return ["sea_surface_fraction"]


def _make_zarr_store(url: str, read_only: bool = True):
    """Create a zarr store from a URL using obstore. If local, return the path."""
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    else:
        return url


def _assert_output_store_absent(path: str) -> None:
    """Refuse to initialize into a pre-existing output store.

    Output stores are written once and treated as immutable; initializing
    the template into an existing store would corrupt or silently overwrite
    it. Delete the store explicitly or pick a new output path.
    """
    fs, root = fsspec.url_to_fs(path)
    if fs.exists(root):
        raise FileExistsError(
            f"output store already exists at {path}; refusing to initialize "
            "into it. Delete it explicitly or choose a new output path."
        )


# ---------------------------------------------------------------------------
# Source opening and load-bearing assertions
# ---------------------------------------------------------------------------


def open_stream(stream: StreamConfig, config: PipelineConfig) -> xr.Dataset:
    """Open the stream's variables lazily, time-subset, with fail-fast checks."""
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
    if config.start_time is not None or config.end_time is not None:
        ds = ds.sel({TIME_DIM: slice(config.start_time, config.end_time)})
    if ds.sizes[TIME_DIM] == 0:
        raise AssertionError(
            f"stream {stream.name!r}: no timesteps in "
            f"[{config.start_time}, {config.end_time}]"
        )
    for name, da in ds.data_vars.items():
        if set(da.dims) != {TIME_DIM, "yh", "xh"}:
            raise AssertionError(
                f"stream {stream.name!r}: {name} has dimensions {da.dims}; "
                f"expected the tracer-grid surface dimensions "
                f"({TIME_DIM}, yh, xh) — check dim_renaming"
            )
    return ds


def source_time_chunk_size(ds: xr.Dataset) -> int:
    """The stream's own time chunk width, used as the beam read width.

    Reading a narrower slice than the source is chunked re-fetches and
    re-decompresses the whole chunk once per slice, so a one-timestep read
    against a ten-timestep chunk costs ten times the bytes it uses. Matching
    the source width makes every chunk pay for itself once. Widening beyond
    it would buy nothing and cost worker memory.
    """
    widths = {
        int(da.encoding["preferred_chunks"][TIME_DIM])
        for da in ds.data_vars.values()
        if TIME_DIM in da.encoding.get("preferred_chunks", {})
    }
    if len(widths) != 1:
        raise AssertionError(
            f"expected one source time chunk width across the stream's "
            f"variables; found {sorted(widths)}"
        )
    return widths.pop()


def shard_aligned_chunk_size(read_chunk_size: int, shard_size: int) -> int:
    """The width to split read chunks to before consolidating into shards.

    ``ConsolidateChunks`` groups a chunk by ``shard_size * (offset //
    shard_size)`` and asserts the group's first offset is the group key, so
    every chunk boundary must fall on a shard boundary. A read chunk wider
    than that alignment straddles one: a 10-timestep read against a
    365-timestep shard puts a chunk at offset 360 in the group for offset 0
    and the next, at 370, in a group keyed 365.

    The greatest common divisor is the widest split that lands every boundary
    on both a read and a shard boundary, and is the read width itself when the
    read width already divides the shard, leaving the split a no-op.
    """
    return math.gcd(read_chunk_size, shard_size)


def load_wetmask(config: PipelineConfig) -> xr.DataArray:
    """The 2D ocean wetmask: the NaN pattern of the reference variable's
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
    if da.ndim != 2:
        raise AssertionError(
            f"wetmask variable {config.wetmask.variable!r} is not 2D after "
            f"selecting the first timestep; dimensions {da.dims}"
        )
    da = da.rename(
        {k: v for k, v in config.stream.dim_renaming.items() if k in da.dims}
    )
    wetmask = da.notnull().reset_coords(drop=True)
    wetmask.attrs = {}
    return wetmask


def _assert_footprint(da: xr.DataArray, wetmask: xr.DataArray, context: str) -> None:
    """Assert a variable's valid-data footprint exactly equals the wetmask.

    Guards against a source variable whose land pattern disagrees with the
    wetmask's, which the normalized regrid would otherwise silently average
    as zeros — and guarantees the output NaN pattern is the same at every
    timestep, which training assumes (a finite target at a masked cell NaNs
    the loss; a NaN target at an unmasked cell NaNs metrics).
    """
    valid, expected = xr.broadcast(da.notnull(), wetmask)
    mismatches = int((valid != expected).sum())
    if mismatches:
        raise AssertionError(
            f"{context}: valid-data footprint of {da.name!r} differs from the "
            f"wetmask at {mismatches} cells"
        )


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------


def _process_chunk(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
    target_grid_name: str,
) -> xr.Dataset:
    """Transform one in-memory chunk: check footprints, regrid, rename, and
    apply the configured postprocess transforms."""
    context = f"stream {stream.name!r}"
    for name in ds.data_vars:
        _assert_footprint(ds[name], wetmask, context)

    regridder = get_regridder(weights_url, target_grid_name)
    regridded, ocean_fraction = regrid_normalized(ds, regridder, wetmask)

    output = xr.Dataset()
    for name, da in regridded.data_vars.items():
        out_name = stream.renaming.get(name, name)
        output[out_name] = da.assign_attrs(
            provenance_attrs(stream.store, name, da.attrs.get(DERIVATION_ATTR))
        )

    for name in stream.full_cell_variables:
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
        chunk_context = ChunkContext(ocean_fraction=ocean_fraction, store=stream.store)
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
    """Beam entry point: process one time chunk of the stream."""
    assert stream is not None
    assert wetmask is not None
    assert weights_url is not None
    assert target_grid_name is not None
    output = _process_chunk(ds, stream, wetmask, weights_url, target_grid_name)
    new_key = xbeam.Key(
        {TIME_DIM: key.offsets[TIME_DIM], "lat": 0, "lon": 0},
        vars=frozenset(output.data_vars),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Statics, template, and driver
# ---------------------------------------------------------------------------


def build_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Eagerly build the time-invariant output fields on the target grid.

    The only one is ``sea_surface_fraction``: the conservative regrid of the
    wetmask, i.e. the fraction of each target cell covered by ocean source
    cells. The target grid's own cell geometry comes from the analytic grid
    constructor and is not stored, and a sea-surface dataset has no use for
    bathymetry or geothermal flux, so there is no statics source store.
    """
    regridder = get_regridder(config.weights_url, config.target_grid)
    fraction = regridder(wetmask.astype("float64"), keep_attrs=False).fillna(0.0)
    statics = xr.Dataset()
    statics["sea_surface_fraction"] = fraction.astype(OUTPUT_DTYPE).assign_attrs(
        long_name="fraction of cell area overlapping ocean source cells",
        units="0-1",
        **provenance_attrs(
            config.wetmask.store,
            config.wetmask.variable,
            "conservative regrid of the wetmask (NaN pattern of "
            f"{config.wetmask.variable}); usable to weight coastal cells",
        ),
    )
    return statics


def build_template(
    config: PipelineConfig,
    stream_dataset: xr.Dataset,
    statics: xr.Dataset,
    wetmask: xr.DataArray,
    output_time: xr.DataArray,
    input_urls: list[str],
) -> xr.Dataset:
    """Eagerly process the first timestep to build the output template, with
    statics stamped in for the driver to write."""
    logger.info("[template] processing the first timestep")
    _, first = process_chunk(
        xbeam.Key({TIME_DIM: 0}),
        stream_dataset.isel({TIME_DIM: slice(0, 1)}).load(),
        config.stream,
        wetmask,
        config.weights_url,
        config.target_grid,
    )
    merged = first.squeeze(TIME_DIM, drop=True).drop_encoding()

    template = xbeam.make_template(merged)
    template = template.expand_dims({TIME_DIM: output_time.values}, axis=0)
    template = template.assign_coords({TIME_DIM: output_time})

    # Statics are eager (numpy-backed), so ChunksToZarr writes them from the
    # driver when it sets up the store.
    template.update(statics.drop_encoding())
    template.attrs["history"] = (
        "Dataset computed by ace/scripts/gfdl_cm4_like_am4_randco2/pipeline, "
        f"using the following input sources: {input_urls}."
    )
    return template


def _expected_output_names(
    config: PipelineConfig, stream_dataset: xr.Dataset
) -> set[str]:
    stream = config.stream
    names = {stream.renaming.get(name, name) for name in stream_dataset.data_vars}
    names.update(stream.full_cell_variables)
    for postprocess_name in stream.postprocess:
        names.update(POSTPROCESS[postprocess_name].adds)
    return names


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce a 1-degree sea-surface dataset store. "
        "Unrecognized arguments are passed to beam as pipeline options."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument(
        "--member",
        help="Ensemble-member name substituted into the config's URLs",
    )
    parser.add_argument(
        "--start-time",
        help="Override the config's inclusive time-range start (e.g. 0152-10-01)",
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
    parser.add_argument(
        "--time-shard-size",
        type=int,
        help=(
            "Override the config's output time shard size. Exists so a subset "
            "test run can cross a shard boundary within a few source chunks"
        ),
    )
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()

    config = load_config(args.config, args.member)
    if args.start_time is not None:
        config.start_time = args.start_time
    if args.end_time is not None:
        config.end_time = args.end_time
    if args.output_path is not None:
        config.output.path = args.output_path
    if args.time_shard_size is not None:
        config.output.time_shard_size = args.time_shard_size
        config.output.__post_init__()
    _assert_output_store_absent(config.output.path)

    logger.info(
        "[config] stream=%s target_grid=%s output=%s time_chunk=%d time_shard=%d",
        config.stream.name,
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
    logger.info("[wetmask] %d ocean cells", int(wetmask.sum()))

    stream_dataset = open_stream(config.stream, config)
    output_time = stream_dataset[TIME_DIM]
    if args.num_timesteps is not None:
        stream_dataset = stream_dataset.isel({TIME_DIM: slice(0, args.num_timesteps)})
        output_time = stream_dataset[TIME_DIM]
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

    input_urls = sorted({config.stream.store, config.wetmask.store, config.weights_url})
    template = build_template(
        config, stream_dataset, statics, wetmask, output_time, input_urls
    )
    expected = _expected_output_names(config, stream_dataset) | set(statics.data_vars)
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

    chunks = {TIME_DIM: source_time_chunk_size(stream_dataset)}
    logger.info(
        "[stream:%s] %d variables, %d timesteps in %d chunks of %d",
        config.stream.name,
        len(stream_dataset.data_vars),
        stream_dataset.sizes[TIME_DIM],
        math.ceil(stream_dataset.sizes[TIME_DIM] / chunks[TIME_DIM]),
        chunks[TIME_DIM],
    )
    split_chunks = {
        TIME_DIM: shard_aligned_chunk_size(
            chunks[TIME_DIM], config.output.time_shard_size
        )
    }
    if split_chunks[TIME_DIM] != chunks[TIME_DIM]:
        logger.info(
            "[stream:%s] splitting to %d before consolidating into %d-wide "
            "shards; the %d-wide read does not align with the shard boundary",
            config.stream.name,
            split_chunks[TIME_DIM],
            config.output.time_shard_size,
            chunks[TIME_DIM],
        )
    logger.info("[pipeline] starting; writing to %s", config.output.path)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | "to_chunks" >> xbeam.DatasetToChunks(stream_dataset, chunks=chunks)
            | "process"
            >> beam.MapTuple(
                process_chunk,
                stream=config.stream,
                wetmask=wetmask,
                weights_url=config.weights_url,
                target_grid_name=config.target_grid,
            )
            | "split" >> xbeam.SplitChunks(split_chunks)
            | "consolidate" >> xbeam.ConsolidateChunks(output_shards)
            | "to_zarr"
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
