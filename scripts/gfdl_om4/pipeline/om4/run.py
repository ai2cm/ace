"""xarray_beam pipeline producing a training-reference ocean dataset.

Reads native-grid 0.25 degree tripolar OM4/CM4 output stores, applies
per-chunk transforms (C-grid to tracer-center interpolation, vector rotation,
conservative regridding to a Gaussian grid with weights in native MOM6 cell
areas, level splitting), and writes one templated, sharded zarr v3 store per
invocation, driven by a YAML config (see pipeline/om4/config.py and
configs/).

One stored value per field class (pipeline.native_area): 3D variables per
ocean volume (weight m_k Δz_k, Δz_k from native deptho), other 2D variables
per ocean area (weight m_0), ``full_cell_variables`` per full cell. Values
stay float64 until the ``OUTPUT_DTYPE`` cast, where the step-5 identities of
pipeline.native_area run (rule ``|lhs − rhs| ≤ rtol · S``).

Run locally on a subset with the DirectRunner (see the om4.mk smoke_test
target), or on Google Cloud Dataflow by passing the corresponding beam
pipeline options after the script's own arguments.

Masking conventions of the output store:

- Every time-varying field is NaN over land. This falls out of the
  normalized regrid: target cells with ocean fraction <=
  OCEAN_FRACTION_THRESHOLD (3D: zero ocean volume) are NaN.
- Per-level binary masks ``mask_0..N`` (and ``mask_2d``, equal to
  ``mask_0``) mark the regridded wetmask footprint: 1 where a target cell
  overlaps any ocean source area, else 0. The NaN pattern of every
  time-varying field equals the masks exactly at every timestep: training
  assumes NaN exactly where ``mask_k == 0`` (a finite target at a masked
  cell NaNs the loss; a NaN target at an unmasked cell NaNs metrics).
  To guarantee this, each chunk passes through _conform_to_wetmask before
  regridding, which raises unless the chunk's valid-data footprint equals
  the wetmask exactly, then masks the chunk to it.
- Rotated C-grid pairs are interpolated to tracer centers dropping invalid
  (land) faces from the average; a wet center whose faces on an axis are
  all land gets the no-normal-flow wall value 0.0 for that grid-relative
  component, filled before rotation. Streams with a ``face_mask_url``
  first invalidate faces the source carries as fake zeros over land (see
  pipeline/om4/face_masks.py). Either way the pair's footprint is the tracer
  wetmask, so NaN-equals-``mask_k`` holds for every output; details and
  rationale in _rotate_pairs.
- ``sea_surface_fraction`` is the regridded surface ocean fraction (0 over
  land, not NaN), usable to weight coastal cells.
- Variables listed in a stream's ``full_cell_variables`` (e.g.
  ``sea_ice_fraction``) are regridded with full-cell semantics — the value
  is per total cell area, land counted as zero — with land-NaN applied
  after; their wetmask-normalized (per-ocean-area) twins are written under
  renamed outputs (e.g. ``ocean_sea_ice_fraction``).
- Masks, ``idepth_*``, ``areacello``, ``sftof``, ``sea_surface_fraction``
  and ``h_prime_*`` are exempt from land-NaN by explicit list (see
  land_nan_exempt_names); they are 0 on land, ``idepth_*`` aside.
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
    Z_STAR_DEFINITION,
    assert_native_statics,
    check_extensive_identities,
    check_field_identities,
    check_static_identities,
    level_thickness,
    native_area_statics,
    regrid_full_cell,
    regrid_thickness_weighted,
    z_star_residual,
)
from ..ocean_emulators_port import (
    OCEAN_FRACTION_THRESHOLD,
    interpolate_to_cell_centers,
    regrid_normalized,
    rotate_vectors,
)
from ..postprocess import (
    DERIVATION_ATTR,
    SEA_ICE_FRACTION,
    ChunkContext,
    PostprocessConfig,
    add_provenance_args,
    assert_postprocess_inputs,
    check_provenance_args,
    provenance_attrs,
    provenance_run_attrs,
)
from ..weights import (
    get_regridder,
    open_areacello,
    open_source_grid,
    read_artifact_attrs,
)
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
from .face_masks import count_masked_candidate_faces, get_face_masks

logger = logging.getLogger(__name__)

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
        + ["areacello", "sftof", "sea_surface_fraction"]
        + [f"h_prime_{k}" for k in range(level_count)]
    )


# ---------------------------------------------------------------------------
# Source opening and load-bearing assertions
# ---------------------------------------------------------------------------


def open_stream(stream: StreamConfig, config: PipelineConfig) -> xr.Dataset:
    """Open a stream's variables lazily, time-subset, with fail-fast checks."""
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
        make_zarr_store(config.wetmask.store), chunks=None, decode_timedelta=False
    )
    if config.wetmask.variable not in ds.data_vars:
        raise AssertionError(
            f"wetmask variable {config.wetmask.variable!r} missing from "
            f"{config.wetmask.store}"
        )
    da = ds[config.wetmask.variable].isel({TIME_DIM: 0}).load()
    if LEVEL_DIM not in da.dims:
        raise AssertionError(
            f"wetmask variable {config.wetmask.variable!r} has no {LEVEL_DIM} dimension"
        )
    if da.sizes[LEVEL_DIM] != config.expected_level_count:
        raise AssertionError(
            f"wetmask variable has {da.sizes[LEVEL_DIM]} levels; expected "
            f"{config.expected_level_count}"
        )
    wetmask = da.notnull().reset_coords(drop=True)
    wetmask.attrs = {}
    return wetmask


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
                f"time coordinate of stream {name!r} differs from stream {names[0]!r}"
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


def _conform_to_wetmask(
    ds: xr.Dataset,
    wetmask: xr.DataArray,
    context: str,
) -> xr.Dataset:
    """Assert every tracer-grid variable's valid-data footprint equals the
    reference-time wetmask, and mask the dataset to it.

    Training requires the output NaN pattern to equal the static ``mask_k``
    exactly, so any instantaneous departure from the reference-time wetmask
    fails the run — there is no conforming repair. Staggered (C-grid)
    variables are not touched here; they conform to the wetmask during
    center interpolation in _rotate_pairs.
    """
    conformed = ds.copy()
    for name in ds.data_vars:
        da = ds[name]
        if not set(wetmask.dims).issubset(set(da.dims)):
            continue
        assert_footprint(da, wetmask, context)
        conformed[name] = da.where(wetmask).assign_attrs(da.attrs)
    return conformed


def _assert_no_masked_candidate_faces(
    u: xr.DataArray, v: xr.DataArray, wetmask: xr.DataArray, context: str
) -> None:
    """Assert no face matching the face-mask fingerprint (valid, exactly
    0.0, with a dry tracer neighbor) survives into the interpolation.

    Runs after the face masks are applied, so a nonzero count means the
    source carries land-born zero faces the artifact does not flag — a
    stale artifact or a changed source — which would silently depress
    coastal center velocities if averaged in.
    """
    wet = wetmask.transpose("yh", "xh").values
    for component, da, dims in (("u", u, ("yh", "xq")), ("v", v, ("yq", "xh"))):
        count = count_masked_candidate_faces(
            da.transpose(..., *dims).values, wet, component
        )
        if count:
            raise AssertionError(
                f"{context}: {count} valid exactly-0.0 {da.name!r} faces "
                "with a dry tracer neighbor survive the face masks; the "
                "face-mask artifact is stale relative to the source"
            )


def _rotate_pairs(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
    face_masks: xr.Dataset | None,
) -> xr.Dataset:
    """Interpolate each configured C-grid vector pair to tracer centers and
    rotate it to geographic components, preserving names and attrs.

    Faces the source masks as land drop out of the face->center average. An
    ocean center whose faces on an axis are *all* land is a wall for that
    axis — the model's resolved normal component there is identically zero
    by no-normal-flow — so that grid-relative component is set to 0.0. The
    fill happens *before* rotation: rotation mixes the components, so
    filling afterwards would also zero a valid along-wall component (e.g.
    the along-channel flow of a one-cell-wide channel). Every pair
    therefore keeps the reference wetmask footprint exactly.

    With ``face_masks`` (the stream's level-selected face-mask artifact),
    flagged faces — land points the source carries as valid zeros — are
    first treated as invalid, so the wall-zero fill applies at the same
    centers a properly-masked source would produce instead of the fake
    zeros being averaged into coastal centers.
    """
    pairs = [pair for pair in stream.rotated_pairs if pair[0] in ds.data_vars]
    if not pairs:
        return ds
    angle = _get_angle(weights_url)
    for u_name, v_name in pairs:
        u_attrs, v_attrs = dict(ds[u_name].attrs), dict(ds[v_name].attrs)
        u_in, v_in = ds[u_name], ds[v_name]
        if face_masks is not None:
            u_in = u_in.where(~face_masks["u_face_mask"])
            v_in = v_in.where(~face_masks["v_face_mask"])
            _assert_no_masked_candidate_faces(
                u_in, v_in, wetmask, f"stream {stream.name!r}"
            )
        u, v = interpolate_to_cell_centers(u_in, v_in, wetmask)
        u = u.fillna(0.0).where(wetmask)
        v = v.fillna(0.0).where(wetmask)
        u, v = rotate_vectors(u, v, angle)
        rotation_note = (
            "interpolated from C-grid points to tracer centers (normalizing "
            "by valid ocean neighbors; a grid-relative component with no "
            "valid face on its axis is 0 at the center, the no-normal-flow "
            "wall value) and rotated from grid-relative to geographic "
            "components"
        )
        if face_masks is not None:
            rotation_note = (
                "structurally-zero coastal C-grid faces (land carried as "
                "valid zeros by the source) treated as invalid; " + rotation_note
            )
        ds = ds.drop_vars([u_name, v_name])
        ds[u_name] = u.assign_attrs(u_attrs, **{DERIVATION_ATTR: rotation_note})
        ds[v_name] = v.assign_attrs(v_attrs, **{DERIVATION_ATTR: rotation_note})
    return ds


def _sea_ice_fraction_source(stream: StreamConfig) -> str | None:
    """The full-cell sea-ice fraction the stream's ``sea_ice`` transform
    reads, or None if the stream does not list ``sea_ice``."""
    for entry in stream.postprocess:
        selection = PostprocessConfig(entry) if isinstance(entry, str) else entry
        if selection.name == "sea_ice":
            name = selection.sources.get("sea_ice_fraction", SEA_ICE_FRACTION)
            if name not in stream.full_cell_variables:
                raise ValueError(
                    f"stream {stream.name!r}: sea_ice reads {name!r}, which is "
                    "not a full_cell_variable"
                )
            return name
    return None


# Output name a postprocess transform reads, for each checked output.
EXTENSIVE_SOURCES = {"hfds_total_area": "hfds"}


def _process_chunk(
    ds: xr.Dataset,
    stream: StreamConfig,
    wetmask: xr.DataArray,
    weights_url: str,
    target_grid_name: str,
    level_index: int | None,
    native_areacello: xr.DataArray,
    dz: xr.DataArray | None,
) -> xr.Dataset:
    """Transform one in-memory chunk: conform to the wetmask, rotate, check
    footprints, regrid in float64 with the native-area weights, check the
    step-5 identities, and (for 3D chunks) split the level into suffixed 2D
    variables.

    3D chunks (``level_index`` set) are regridded per ocean volume with the
    level's Δz (``dz``); 2D chunks per ocean area, and ``full_cell_variables``
    per full cell. Every variable's footprint is asserted equal to the
    reference-time wetmask and masked to it (see _conform_to_wetmask), so
    the output NaN pattern equals the ``mask_k`` statics at every timestep.
    Values stay float64 until the ``OUTPUT_DTYPE`` cast.
    """
    context = f"stream {stream.name!r}"
    if level_index is not None:
        context += f" level {level_index}"

    face_masks = None
    if stream.face_mask_url is not None:
        face_masks = get_face_masks(stream.face_mask_url).isel(
            {LEVEL_DIM: level_index if level_index is not None else 0}, drop=True
        )

    ds = ds.astype(np.float64)
    ds = _conform_to_wetmask(ds, wetmask, context)
    ds = _rotate_pairs(ds, stream, wetmask, weights_url, face_masks)
    for name in ds.data_vars:
        assert_footprint(ds[name], wetmask, context)

    regridder = get_regridder(weights_url, target_grid_name)
    area = open_areacello(weights_url, target_grid_name)
    if level_index is None:
        regridded, ocean_fraction = regrid_normalized(ds, regridder, wetmask)
        per_volume, per_ocean_area = xr.Dataset(), regridded
    else:
        if dz is None:
            raise ValueError(f"{context}: a 3D chunk needs Δz")
        regridded, ocean_fraction = regrid_thickness_weighted(
            ds, regridder, wetmask, dz
        )
        per_volume, per_ocean_area = regridded, xr.Dataset()

    full_cell = xr.Dataset()
    for name in stream.full_cell_variables:
        if name in ds.data_vars:
            full_cell[name] = regrid_full_cell(
                regridder, ds[name], wetmask, ocean_fraction
            )

    check_field_identities(
        ds,
        regridder,
        wetmask,
        area,
        ocean_fraction,
        native_areacello,
        context,
        per_volume=per_volume,
        per_ocean_area=per_ocean_area,
        full_cell=full_cell,
        dz=dz,
    )

    output = xr.Dataset()
    for name, da in regridded.data_vars.items():
        out_name = stream.renaming.get(name, name)
        derivation = da.attrs.get(DERIVATION_ATTR)
        if level_index is not None:
            depth = float(ds[LEVEL_DIM].item())
            level_note = (
                f"level {level_index} (depth {depth:g} m) of {name}, "
                "ocean-volume weighted (m_k Δz_k)"
            )
            derivation = f"{level_note}; {derivation}" if derivation else level_note
            da = da.squeeze(LEVEL_DIM, drop=True)
            out_name = f"{out_name}_{level_index}"
        output[out_name] = da.assign_attrs(
            provenance_attrs(stream.store, name, derivation)
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
        ice_fraction = _sea_ice_fraction_source(stream)
        native_terms = {}
        ice_volume = None
        if ice_fraction is not None and ice_fraction in ds.data_vars:
            native_terms["sea_ice_volume"] = ds["HI"] * ds[ice_fraction]
            ice_volume = regrid_full_cell(
                regridder, native_terms["sea_ice_volume"], wetmask, ocean_fraction
            )
        chunk_context = ChunkContext(
            ocean_fraction=ocean_fraction,
            areacello=area,
            ice_volume_per_cell_area=ice_volume,
            store=stream.store,
        )
        for spec in stream.postprocess_specs():
            if all(v in output.data_vars for v in spec.requires):
                output = spec.fn(output, chunk_context)

        source_names = {stream.renaming.get(n, n): n for n in ds.data_vars}
        for name, read in EXTENSIVE_SOURCES.items():
            if name in output.data_vars:
                native_terms[name] = ds[source_names[read]]
        check_extensive_identities(
            output,
            native_terms,
            wetmask,
            area,
            ocean_fraction,
            native_areacello,
            context,
        )

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
    native_deptho: xr.DataArray | None = None,
) -> tuple[xbeam.Key, xr.Dataset]:
    """Beam entry point: process one (time, level) chunk of a stream.

    ``native_areacello``, ``native_deptho``: the native statics on the
    wetmask's grid (see load_native_statics); Δz of a 3D chunk's level is
    formed here from ``native_deptho`` and the wetmask's ``z_l``."""
    assert stream is not None
    assert wetmask is not None
    assert weights_url is not None
    assert target_grid_name is not None
    assert native_areacello is not None
    assert native_deptho is not None
    level_index = key.offsets.get(LEVEL_DIM)
    dz = None
    if level_index is not None:
        interfaces = _interface_depths(wetmask[LEVEL_DIM].values)
        dz = level_thickness(
            native_deptho, interfaces[level_index : level_index + 2]
        ).isel({LEVEL_DIM: 0}, drop=True)
        wetmask = wetmask.isel({LEVEL_DIM: level_index}, drop=True)
    elif LEVEL_DIM in wetmask.dims:
        # 2D (surface) variables are checked and normalized against the
        # surface level of the wetmask.
        wetmask = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    output = _process_chunk(
        ds,
        stream,
        wetmask,
        weights_url,
        target_grid_name,
        level_index,
        native_areacello,
        dz,
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


# Read by fixed name from statics.store, outside statics.variables.
NATIVE_STATIC_NAMES = ["areacello", "sftof", "deptho"]


def load_native_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Native ``areacello`` (a_i, m²), ``sftof`` (%), ``deptho`` (H_i, m)
    from ``config.statics.store``, float64, on the wetmask's grid."""
    source = xr.open_zarr(
        make_zarr_store(config.statics.store), chunks=None, decode_timedelta=False
    )
    missing = set(NATIVE_STATIC_NAMES) - set(source.data_vars)
    if missing:
        raise AssertionError(
            f"native statics missing from {config.statics.store}: {sorted(missing)}"
        )
    native = source[NATIVE_STATIC_NAMES].load().astype(np.float64)
    xr.align(wetmask, native, join="exact")
    return native


def build_statics(config: PipelineConfig, wetmask: xr.DataArray) -> xr.Dataset:
    """Eagerly build all static output fields on the target grid.

    Areas, fractions, masks and ``h_prime_<k>`` come from the artifact's ŵ and
    A_J and the native ``areacello``, ``sftof``, ``deptho``
    (native_area.native_area_statics). Built and checked in float64, cast
    to ``OUTPUT_DTYPE`` last."""
    if OCEAN_FRACTION_THRESHOLD != 0:
        raise AssertionError(
            f"OCEAN_FRACTION_THRESHOLD = {OCEAN_FRACTION_THRESHOLD}; mask_Jk = "
            "1[V'_Jk > 0] equals today's mask only at 0"
        )
    regridder = get_regridder(config.weights_url, config.target_grid)
    source = xr.open_zarr(
        make_zarr_store(config.statics.store), chunks=None, decode_timedelta=False
    )
    missing = set(config.statics.variables) - set(source.data_vars)
    if missing:
        raise AssertionError(
            f"static variables missing from {config.statics.store}: {sorted(missing)}"
        )
    native = load_native_statics(config, wetmask)

    interfaces = _interface_depths(wetmask[LEVEL_DIM].values)
    assert_native_statics(
        wetmask, native["deptho"], interfaces, native["areacello"], native["sftof"]
    )
    dz = level_thickness(native["deptho"], interfaces)
    area_statics = native_area_statics(
        regridder,
        open_areacello(config.weights_url, config.target_grid),
        wetmask,
        dz,
        native["sftof"],
    )
    area_fields = area_statics.stored
    level_count = wetmask.sizes[LEVEL_DIM]
    statics = xr.Dataset()

    volume_note = (
        "V'_Jk = A_J · regrid(m_k Δz_k), the level-{k} ocean volume in native "
        f"cell areas (A_J, ŵ: {config.weights_url}; m_k: NaN pattern of "
        f"{config.wetmask.variable}; Δz_k from deptho)"
    )
    for k in range(level_count):
        statics[f"mask_{k}"] = area_fields[f"mask_{k}"].assign_attrs(
            long_name=f"ocean mask level-{k}",
            units="0 if land, 1 if ocean",
            **provenance_attrs(
                config.wetmask.store,
                config.wetmask.variable,
                f"1 where {volume_note.format(k=k)} is > 0, else 0",
            ),
        )
    statics["mask_2d"] = statics["mask_0"].copy()

    statics["sea_surface_fraction"] = area_fields["sea_surface_fraction"].assign_attrs(
        long_name="ocean fraction of cell area, in native cell areas",
        units="0-1",
        **provenance_attrs(
            config.statics.store,
            "sftof",
            "sftof / 100 = W'_J0 / A_J; usable to weight coastal cells",
        ),
    )
    statics["sftof"] = area_fields["sftof"].assign_attrs(
        long_name="ocean area percentage, regridded with native-area overlaps O",
        units="%",
        **provenance_attrs(
            config.statics.store,
            "sftof",
            "Σ_i O_Ji sftof_i / A_J = 100 W'_J0 / A_J over the full cell "
            f"(ŵ, A_J: {config.weights_url}); 0 on land and unmapped cells",
        ),
    )
    statics["areacello"] = area_fields["areacello"].assign_attrs(
        long_name="cell area covered by native ocean-model cells",
        units="m2",
        **provenance_attrs(
            config.weights_url,
            "areacello",
            "A_J = Σ_i O_Ji over native cells wet and dry, in native "
            f"(MOM6) cell areas from {config.statics.store}; 0 on unmapped cells",
        ),
    )
    for k in range(level_count):
        statics[f"h_prime_{k}"] = area_fields[f"h_prime_{k}"].assign_attrs(
            long_name=f"ocean volume per ocean area, level-{k}",
            units="m",
            **provenance_attrs(
                config.statics.store,
                "deptho",
                f"V'_Jk / W'_J0 where mask_{k} = 1, else 0; "
                f"{volume_note.format(k=k)}; W'_J0 = A_J · regrid(m_0)",
            ),
        )

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

    # Regriddable static source fields, per ocean area like the streams.
    fields = source[config.statics.variables].load().astype(np.float64)
    surface_wetmask = wetmask.isel({LEVEL_DIM: 0}, drop=True)
    regridded, _ = regrid_normalized(fields, regridder, surface_wetmask)
    for name, da in regridded.data_vars.items():
        statics[name] = da.assign_attrs(provenance_attrs(config.statics.store, name))
    deptho = (
        regridded["deptho"]
        if "deptho" in regridded
        else regrid_normalized(native[["deptho"]], regridder, surface_wetmask)[0][
            "deptho"
        ]
    )

    check_static_identities(
        area_statics, regridder, wetmask, native["areacello"], native["deptho"], deptho
    )

    # Enforce the land-NaN convention on everything not explicitly exempt.
    exempt = set(land_nan_exempt_names(level_count))
    land = statics["mask_2d"] == 0
    for name in statics.data_vars:
        if statics[name].ndim == 0:
            continue
        if name not in exempt:
            statics[name] = statics[name].where(~land)
        statics[name] = statics[name].astype(OUTPUT_DTYPE)
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
    native: xr.Dataset,
    run_attrs: dict,
) -> xr.Dataset:
    """Eagerly process the first timestep of every stream to build the
    output template, with statics stamped in for the driver to write.

    ``native``: load_native_statics, for the chunk side inputs;
    ``run_attrs``: root attrs added to the store."""
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
                native["areacello"],
                native["deptho"],
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
                native["areacello"],
                native["deptho"],
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
    template.attrs.update(run_attrs)
    return template


# Stored names the z* residual reads: η, and the X of cont_k.
Z_STAR_ETA = "SSH"
Z_STAR_FIELDS = ("thetao", "so")


def _stored_source(
    config: PipelineConfig, stream_datasets: dict[str, xr.Dataset], stored_name: str
) -> xr.DataArray:
    """The native variable a stream stores as ``stored_name`` (after
    ``renaming``)."""
    found = [
        ds[name]
        for stream in config.streams
        for ds in [stream_datasets[stream.name]]
        for name in ds.data_vars
        if stream.renaming.get(name, name) == stored_name
    ]
    if len(found) != 1:
        raise AssertionError(
            f"{len(found)} streams store {stored_name!r}; the z* residual needs one"
        )
    return found[0]


def native_area_checks(
    config: PipelineConfig,
    stream_datasets: dict[str, xr.Dataset],
    wetmask: xr.DataArray,
    native: xr.Dataset,
    output_time: xr.DataArray,
) -> dict:
    """Report-only root attr ``native_area_checks`` (no gate): the artifact's
    a/ã spread, and the z* residual at the first and last output times."""
    regridder = get_regridder(config.weights_url, config.target_grid)
    area = open_areacello(config.weights_url, config.target_grid)
    dz = level_thickness(native["deptho"], _interface_depths(wetmask[LEVEL_DIM].values))
    eta = _stored_source(config, stream_datasets, Z_STAR_ETA)
    fields = {
        name: _stored_source(config, stream_datasets, name) for name in Z_STAR_FIELDS
    }
    snapshots = {}
    for t in sorted({0, output_time.sizes[TIME_DIM] - 1}):
        snapshot = {TIME_DIM: t}
        snapshots[str(output_time.values[t])] = z_star_residual(
            regridder,
            area,
            wetmask,
            dz,
            native["areacello"],
            native["deptho"],
            eta.isel(snapshot, drop=True).load(),
            xr.Dataset(
                {name: x.isel(snapshot, drop=True) for name, x in fields.items()}
            ).load(),
        )
    spread = read_artifact_attrs(config.weights_url)["a_over_a_tilde_spread"]
    return {
        "a_over_a_tilde_spread": spread,
        "z_star_residual": {"definition": Z_STAR_DEFINITION, "snapshots": snapshots},
    }


def _expected_output_names(
    config: PipelineConfig, stream_datasets: dict[str, xr.Dataset]
) -> set[str]:
    names: set[str] = set()
    for stream in config.streams:
        ds = stream_datasets[stream.name]
        stream_names: set[str] = set()
        for name in ds.data_vars:
            out_name = stream.renaming.get(name, name)
            if LEVEL_DIM in ds[name].dims:
                stream_names.update(
                    f"{out_name}_{k}" for k in range(ds.sizes[LEVEL_DIM])
                )
            else:
                stream_names.add(out_name)
        stream_names.update(stream.full_cell_variables)
        specs = stream.postprocess_specs()
        assert_postprocess_inputs(specs, stream_names, f"stream {stream.name!r}")
        for spec in specs:
            stream_names.update(spec.adds)
        names.update(stream_names)
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

    config = load_config(args.config)
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
    native = load_native_statics(config, wetmask)
    logger.info("[%s] a/ã spread and z* residual (report only)", NATIVE_AREA_CHECKS)
    run_attrs = {
        NATIVE_AREA_CHECKS: native_area_checks(
            config, stream_datasets, wetmask, native, output_time
        ),
        **provenance_run_attrs(args.ace_commit, args.image_tag),
    }

    input_urls = sorted(
        {stream.store for stream in config.streams}
        | {config.statics.store, config.wetmask.store, config.weights_url}
    )
    template = build_template(
        config,
        stream_datasets,
        statics,
        wetmask,
        output_time,
        input_urls,
        native,
        run_attrs,
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

    output_store = make_zarr_store(config.output.path, read_only=False)

    streams_by_name = {stream.name: stream for stream in config.streams}
    logger.info("[pipeline] starting; writing to %s", config.output.path)
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        for name, ds in stream_datasets.items():
            stream = streams_by_name[name]
            branches = []
            ds_3d, ds_2d = _split_stream(ds)
            # Read at the source's own time-chunk width so each source chunk
            # is fetched once, not once per timestep it holds.
            if ds_3d.data_vars:
                width = source_time_chunk_size(ds_3d)
                branches.append((f"{name}_3d", ds_3d, {TIME_DIM: width, LEVEL_DIM: 1}))
            if ds_2d.data_vars:
                width = source_time_chunk_size(ds_2d)
                branches.append((f"{name}_2d", ds_2d, {TIME_DIM: width}))
            for label, branch_ds, chunks in branches:
                n_chunks = int(
                    np.prod(
                        [
                            math.ceil(branch_ds.sizes[dim] / size)
                            for dim, size in chunks.items()
                        ]
                    )
                )
                logger.info(
                    "[stream:%s] %d variables in %d chunks of %d timesteps",
                    label,
                    len(branch_ds.data_vars),
                    n_chunks,
                    chunks[TIME_DIM],
                )
                split_chunks = {
                    TIME_DIM: shard_aligned_chunk_size(
                        chunks[TIME_DIM], config.output.time_shard_size
                    )
                }
                if split_chunks[TIME_DIM] != chunks[TIME_DIM]:
                    logger.info(
                        "[stream:%s] splitting to %d before consolidating into "
                        "%d-wide shards; the %d-wide read does not align with "
                        "the shard boundary",
                        label,
                        split_chunks[TIME_DIM],
                        config.output.time_shard_size,
                        chunks[TIME_DIM],
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
                        native_areacello=native["areacello"],
                        native_deptho=native["deptho"],
                    )
                    | f"{label}_write"
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
