"""xarray_beam pipeline producing a 1-degree sea-surface dataset.

Reads a native-grid 0.25 degree tripolar ice-model snapshot store, applies
wetmask-normalized conservative regridding to a Gaussian grid per chunk, and
writes one templated, sharded zarr v3 store per invocation, driven by a YAML
config and an ensemble-member name (see pipeline/randco2/config.py and
configs/).

Run locally on a subset with the DirectRunner (see the randco2.mk
randco2_smoke_test target), or on Google Cloud Dataflow by passing the
corresponding beam pipeline options after the script's own arguments.

The source fields are two-dimensional surface quantities on the tracer grid,
so nothing here splits vertical levels, interpolates from staggered points,
or rotates vectors. The weight artifact is a native-area one
(pipeline.native_area_weights): its ``ŵ`` and ``A_J`` are in MOM6 cell
areas. The statics store (``statics.store``, the OM4 ocean statics on the
same tripolar grid) supplies native ``areacello`` (``a_i``) and ``sftof``
by fixed name.

Masking conventions of the output store:

- Every time-varying field is NaN over land. This falls out of the
  wetmask-normalized regrid: target cells with ocean fraction <=
  OCEAN_FRACTION_THRESHOLD are NaN. The NaN pattern is the same at every
  timestep because each chunk's valid-data footprint is asserted to equal
  the reference-time wetmask before regridding (see pipeline.zarr_io.assert_footprint).
- The statics ``areacello`` (``A_J``, m², 0 on target rows no native cell
  covers), ``sftof`` (%) and ``sea_surface_fraction`` (``sftof / 100``) are
  0 over land, not NaN, and exempt from land-NaN by explicit list (see
  land_nan_exempt_names). ``areacello · sftof / 100`` is the ocean area.
- Variables listed in the stream's ``full_cell_variables`` (here
  ``sea_ice_fraction``) are regridded with full-cell semantics — the value
  is per total cell area, land counted as zero — with land-NaN applied
  after; their wetmask-normalized (per-ocean-area) twins are written under
  renamed outputs (``ocean_sea_ice_fraction``).

Values stay float64 until the ``OUTPUT_DTYPE`` cast, where the area
identities of pipeline.native_area run (rule ``|lhs − rhs| ≤ rtol · S``).
"""

import argparse
import logging
import math

import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam
from apache_beam.options.pipeline_options import PipelineOptions

from ..native_area import (
    LEVEL_DIM,
    NATIVE_AREA_CHECKS,
    NativeAreaStatics,
    assert_native_area_statics,
    check_field_identities,
    check_static_identities,
    regrid_full_cell,
)
from ..ocean_emulators_port import OCEAN_FRACTION_THRESHOLD, regrid_normalized
from ..postprocess import (
    DERIVATION_ATTR,
    ChunkContext,
    add_provenance_args,
    assert_postprocess_inputs,
    check_provenance_args,
    provenance_attrs,
    provenance_run_attrs,
)
from ..weights import get_regridder, open_areacello, read_artifact_attrs
from ..zarr_io import (
    OUTPUT_DTYPE,
    TIME_DIM,
    WriteShardedZarr,
    assert_footprint,
    assert_output_store_absent,
    make_zarr_store,
    shard_aligned_chunk_size,
    source_time_chunk_size,
)
from .config import PipelineConfig, StreamConfig, load_config

logger = logging.getLogger(__name__)


def land_nan_exempt_names() -> list[str]:
    """Output variables exempt from the NaN-over-land convention.

    An explicit list rather than a name-pattern match, so a new variable
    whose name happens to resemble one of these is still masked.
    """
    return ["areacello", "sftof", "sea_surface_fraction"]


# ---------------------------------------------------------------------------
# Source opening and load-bearing assertions
# ---------------------------------------------------------------------------


def open_stream(stream: StreamConfig, config: PipelineConfig) -> xr.Dataset:
    """Open the stream's variables lazily, time-subset, with fail-fast checks."""
    ds = xr.open_zarr(
        make_zarr_store(stream.store), chunks=None, decode_timedelta=False
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


def load_wetmask(config: PipelineConfig) -> xr.DataArray:
    """The 2D ocean wetmask: the NaN pattern of the reference variable's
    first timestep (True over ocean)."""
    ds = xr.open_zarr(
        make_zarr_store(config.wetmask.store), chunks=None, decode_timedelta=False
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


NATIVE_STATICS = ("areacello", "sftof")


def load_native_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Native ``areacello`` (``a_i``, m²) and ``sftof`` (%) from
    ``config.statics.store``, float64, on the wetmask's dims and coords.

    Read by fixed name and placed by position: the statics store and the
    stream share the tripolar grid, not their coordinate labels."""
    source = xr.open_zarr(
        make_zarr_store(config.statics.store), chunks=None, decode_timedelta=False
    )
    missing = set(NATIVE_STATICS) - set(source.data_vars)
    if missing:
        raise AssertionError(
            f"native statics missing from {config.statics.store}: {sorted(missing)}"
        )
    native = xr.Dataset()
    for name in NATIVE_STATICS:
        da = source[name].reset_coords(drop=True)
        if set(da.dims) != set(wetmask.dims) or da.sizes != wetmask.sizes:
            raise AssertionError(
                f"{name} in {config.statics.store} has sizes {dict(da.sizes)}; "
                f"the wetmask has {dict(wetmask.sizes)}"
            )
        native[name] = xr.DataArray(
            da.transpose(*wetmask.dims).values.astype(np.float64),
            dims=wetmask.dims,
            coords=wetmask.coords,
        )
    return native


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------


def _process_chunk(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
    target_grid_name: str,
    native_areacello: xr.DataArray,
) -> xr.Dataset:
    """Transform one in-memory chunk: check footprints, regrid in float64,
    check the area identities, rename, and apply the configured postprocess
    transforms."""
    context = f"stream {stream.name!r}"
    for name in ds.data_vars:
        assert_footprint(ds[name], wetmask, context)
    ds = ds.astype(np.float64)

    regridder = get_regridder(weights_url, target_grid_name)
    regridded, ocean_fraction = regrid_normalized(ds, regridder, wetmask)
    area = open_areacello(weights_url, target_grid_name)

    full_cell = xr.Dataset()
    for name in stream.full_cell_variables:
        full_cell[name] = regrid_full_cell(regridder, ds[name], wetmask, ocean_fraction)

    check_field_identities(
        ds,
        regridder,
        wetmask,
        area,
        ocean_fraction,
        native_areacello,
        context,
        per_ocean_area=regridded,
        full_cell=full_cell,
    )

    output = xr.Dataset()
    for name, da in regridded.data_vars.items():
        out_name = stream.renaming.get(name, name)
        output[out_name] = da.assign_attrs(
            provenance_attrs(stream.store, name, da.attrs.get(DERIVATION_ATTR))
        )

    for name, full in full_cell.data_vars.items():
        output[name] = full.assign_attrs(
            provenance_attrs(
                stream.store,
                name,
                "zeroed over land (wetmask) and conservatively regridded "
                "without ocean-fraction normalization, giving the "
                "per-total-cell-area quantity; NaN over land applied after",
            )
        )

    if stream.postprocess:
        chunk_context = ChunkContext(
            ocean_fraction=ocean_fraction, store=stream.store, areacello=area
        )
        for spec in stream.postprocess_specs():
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
    native_areacello: xr.DataArray | None = None,
) -> tuple[xbeam.Key, xr.Dataset]:
    """Beam entry point: process one time chunk of the stream."""
    assert stream is not None
    assert wetmask is not None
    assert weights_url is not None
    assert target_grid_name is not None
    assert native_areacello is not None
    output = _process_chunk(
        ds, stream, wetmask, weights_url, target_grid_name, native_areacello
    )
    new_key = xbeam.Key(
        {TIME_DIM: key.offsets[TIME_DIM], "lat": 0, "lon": 0},
        vars=frozenset(output.data_vars),
    )
    return new_key, output


# ---------------------------------------------------------------------------
# Statics, template, and driver
# ---------------------------------------------------------------------------


def build_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Eagerly build the time-invariant output fields on the target grid, in
    native (MOM6) cell areas:

        areacello_J            = A_J                       artifact, 0 on Z
        sftof_J                = Σ_i O_Ji sftof_i / A_J    full-cell regrid
        sea_surface_fraction_J = sftof_J / 100 = W'_J0 / A_J

    each 0 on Z = {J : A_J = 0}. Native ``areacello``, ``sftof`` come from
    ``config.statics.store`` by fixed name.
    """
    if OCEAN_FRACTION_THRESHOLD != 0:
        raise AssertionError(
            f"OCEAN_FRACTION_THRESHOLD = {OCEAN_FRACTION_THRESHOLD}; the "
            "native-area statics assume 0"
        )
    regridder = get_regridder(config.weights_url, config.target_grid)
    native = load_native_statics(config, wetmask)
    assert_native_area_statics(wetmask, native["areacello"], native["sftof"])
    fraction = regridder(wetmask.astype(np.float64), keep_attrs=False)
    area = open_areacello(config.weights_url, config.target_grid)
    sftof = regridder(native["sftof"].where(wetmask, 0.0), keep_attrs=False).fillna(0.0)
    sea_surface_fraction = sftof / 100.0
    ocean_area = (area * fraction).fillna(0.0)
    stored = xr.Dataset(
        {
            "areacello": area,
            "sftof": sftof,
            "sea_surface_fraction": sea_surface_fraction,
            "mask_0": (ocean_area > 0).astype(np.float64),
        }
    )
    check_static_identities(
        NativeAreaStatics(stored, ocean_area),
        regridder,
        wetmask.expand_dims(LEVEL_DIM),
        native["areacello"],
    )

    statics = xr.Dataset()
    statics["areacello"] = area.astype(OUTPUT_DTYPE).assign_attrs(
        long_name="target cell area covered by native ocean-grid cells",
        units="m2",
        **provenance_attrs(
            config.weights_url,
            "areacello",
            "A_J = Σ_i O_Ji, native (MOM6) cell area overlapping the target "
            "cell, from the weight artifact; 0 where no native cell overlaps",
        ),
    )
    statics["sftof"] = sftof.astype(OUTPUT_DTYPE).assign_attrs(
        long_name="sea area percentage, regridded with native-area overlaps O",
        units="%",
        **provenance_attrs(
            config.statics.store,
            "sftof",
            "full-cell conservative regrid of native sftof with overlaps in "
            "native cell areas; 0 where no native cell overlaps",
        ),
    )
    statics["sea_surface_fraction"] = sea_surface_fraction.astype(
        OUTPUT_DTYPE
    ).assign_attrs(
        long_name="fraction of cell area that is ocean",
        units="0-1",
        **provenance_attrs(config.statics.store, "sftof", "sftof / 100"),
    )
    return statics


def build_template(
    config: PipelineConfig,
    stream_dataset: xr.Dataset,
    statics: xr.Dataset,
    wetmask: xr.DataArray,
    output_time: xr.DataArray,
    input_urls: list[str],
    native_areacello: xr.DataArray,
    run_attrs: dict,
) -> xr.Dataset:
    """Eagerly process the first timestep to build the output template, with
    statics stamped in for the driver to write; ``run_attrs``: root attrs
    added to the store."""
    logger.info("[template] processing the first timestep")
    _, first = process_chunk(
        xbeam.Key({TIME_DIM: 0}),
        stream_dataset.isel({TIME_DIM: slice(0, 1)}).load(),
        config.stream,
        wetmask,
        config.weights_url,
        config.target_grid,
        native_areacello,
    )
    merged = first.squeeze(TIME_DIM, drop=True).drop_encoding()

    template = xbeam.make_template(merged)
    template = template.expand_dims({TIME_DIM: output_time.values}, axis=0)
    template = template.assign_coords({TIME_DIM: output_time})

    # Statics are eager (numpy-backed), so ChunksToZarr writes them from the
    # driver when it sets up the store.
    template.update(statics.drop_encoding())
    template.attrs["history"] = (
        "Dataset computed by ace/scripts/gfdl_om4/pipeline/randco2, "
        f"using the following input sources: {input_urls}."
    )
    template.attrs.update(run_attrs)
    return template


def _expected_output_names(
    config: PipelineConfig, stream_dataset: xr.Dataset
) -> set[str]:
    stream = config.stream
    names = {stream.renaming.get(name, name) for name in stream_dataset.data_vars}
    names.update(stream.full_cell_variables)
    specs = stream.postprocess_specs()
    assert_postprocess_inputs(specs, names, f"stream {stream.name!r}")
    for spec in specs:
        names.update(spec.adds)
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
    add_provenance_args(parser)
    return parser


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    check_provenance_args(parser, args, pipeline_args)

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
    assert_output_store_absent(config.output.path)

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

    native_areacello = load_native_statics(config, wetmask)["areacello"]
    # Report only; no z* residual (no z_l field).
    spread = read_artifact_attrs(config.weights_url)["a_over_a_tilde_spread"]
    run_attrs = {
        NATIVE_AREA_CHECKS: {"a_over_a_tilde_spread": spread},
        **provenance_run_attrs(args.ace_commit, args.image_tag),
    }
    input_urls = sorted(
        {
            config.stream.store,
            config.wetmask.store,
            config.statics.store,
            config.weights_url,
        }
    )
    template = build_template(
        config,
        stream_dataset,
        statics,
        wetmask,
        output_time,
        input_urls,
        native_areacello,
        run_attrs,
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

    output_store = make_zarr_store(config.output.path, read_only=False)

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
                native_areacello=native_areacello,
            )
            | "write"
            >> WriteShardedZarr(
                output_store,
                template,
                time_chunk_size=config.output.time_chunk_size,
                time_shard_size=config.output.time_shard_size,
                split_chunks=split_chunks,
            )
        )
    logger.info("[write] pipeline complete: %s", config.output.path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # apache_beam may have already configured the root logger on import,
    # making basicConfig a no-op; raise the level explicitly.
    logging.getLogger().setLevel(logging.INFO)
    main()
