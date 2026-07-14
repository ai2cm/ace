"""YAML-driven configuration for the ocean dataset pipeline.

Each YAML file describes one output store: the source zarr stores and the
streams of variables to read from them, the transforms to apply (vector
rotation, level splitting), the target grid and weight artifact, and the
output layout. Transforms are code; configs only select and parameterize
them, so a new simulation or a new output store is a config change, not a
code change.
"""

import dataclasses

import dacite
import yaml

from .postprocess import POSTPROCESS


@dataclasses.dataclass
class StreamConfig:
    """One stream of time-varying variables read from a single source store.

    Attributes:
        name: label used in logging and beam stage names.
        store: URL of the source zarr store.
        variables: source variable names to process. 3D variables (with a
            level dimension) are split into per-level ``_0.._N`` outputs.
        rotated_pairs: pairs of (x-component, y-component) variable names to
            rotate from grid-relative to geographic (eastward/northward)
            components. C-grid components are interpolated to tracer centers
            first.
        renaming: mapping of source names to output names, applied before
            level splitting (a renamed 3D variable yields renamed per-level
            outputs).
        dim_renaming: mapping of source dimension names to the ocean-grid
            names the shared regridding machinery expects (tracer ``xh``/
            ``yh``, staggered ``xq``/``yq``). A dimension renamed to a
            staggered name with one more point than its tracer counterpart
            (symmetric staggering, both edges present) has its first point
            dropped to match the right/north-edge convention.
        time_subsample_stride: if set, keep every Nth timestep, aligned to
            the ends of N-step blocks of the source store's raw time grid
            (e.g. 20 subsamples 6-hourly data to 5-daily block ends). The
            cross-stream time-alignment assertion then guarantees the
            subsample lands exactly on the shared time coordinate.
        full_cell_variables: variables additionally regridded with full-cell
            semantics — NaN filled with 0 over the whole grid (land
            included) and conservatively regridded without ocean-fraction
            normalization — written under their source name, with NaN over
            land applied after. Each must also have a ``renaming`` entry so
            its wetmask-normalized twin doesn't collide.
        postprocess: named post-regrid transforms to apply per chunk, in
            order (see pipeline/postprocess.py).
        face_mask_url: URL prefix of a precomputed face-mask artifact (see
            pipeline/face_masks.py) for sources whose staggered velocities
            carry remap-born zeros over land. When set, the stream's rotated
            pairs have the flagged faces treated as invalid before center
            interpolation, so the wall-zero fill at centers with no valid
            face on an axis (see run._rotate_pairs) applies where a
            properly-masked source would put it instead of the fake zeros
            being averaged into coastal centers.
    """

    name: str
    store: str
    variables: list[str]
    rotated_pairs: list[list[str]] = dataclasses.field(default_factory=list)
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    dim_renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    time_subsample_stride: int | None = None
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
    postprocess: list[str] = dataclasses.field(default_factory=list)
    face_mask_url: str | None = None

    def __post_init__(self):
        for pair in self.rotated_pairs:
            if len(pair) != 2:
                raise ValueError(f"rotated_pairs entries must be [u, v]; got {pair}")
            for name in pair:
                if name not in self.variables:
                    raise ValueError(
                        f"rotated variable {name!r} not in stream {self.name!r} "
                        "variables"
                    )
        if self.time_subsample_stride is not None and self.time_subsample_stride < 1:
            raise ValueError(
                f"time_subsample_stride must be >= 1; got "
                f"{self.time_subsample_stride}"
            )
        for name in self.full_cell_variables:
            if name not in self.variables:
                raise ValueError(
                    f"full-cell variable {name!r} not in stream {self.name!r} "
                    "variables"
                )
            if name not in self.renaming:
                raise ValueError(
                    f"full-cell variable {name!r} needs a renaming entry in "
                    f"stream {self.name!r}: its full-cell output keeps the "
                    "source name, so the wetmask-normalized output must be "
                    "renamed to avoid a collision"
                )
        for name in self.postprocess:
            if name not in POSTPROCESS:
                raise ValueError(
                    f"unknown postprocess {name!r} in stream {self.name!r}; "
                    f"available: {sorted(POSTPROCESS)}"
                )
        if self.face_mask_url is not None and not self.rotated_pairs:
            raise ValueError(
                f"stream {self.name!r} sets face_mask_url but has no "
                "rotated_pairs for it to apply to"
            )


@dataclasses.dataclass
class WetmaskConfig:
    """Where the 3D ocean wetmask comes from.

    The wetmask is the NaN pattern of the reference variable's first
    timestep. Every processed variable is conformed to it (bottom slivers
    that dry/re-wet with sea level are filled from the level above or
    dropped; see run._conform_to_wetmask) and then asserted to match it
    exactly, so the output NaN pattern equals the ``mask_k`` statics at
    every timestep and a source whose footprint truly disagrees fails
    loudly instead of being silently clipped or zero-filled.

    Attributes:
        store: URL of the zarr store holding the reference variable.
        variable: name of a 3D (level, y, x) variable whose NaN pattern
            defines the wetmask.
    """

    store: str
    variable: str


@dataclasses.dataclass
class StaticsConfig:
    """Static (time-invariant) fields for the output store.

    Attributes:
        store: URL of the static source zarr store.
        variables: tracer-point fields to regrid onto the target grid.
    """

    store: str
    variables: list[str]


@dataclasses.dataclass
class OutputConfig:
    """Output store layout.

    Attributes:
        path: URL of the output zarr store.
        time_chunk_size: zarr chunk size along time.
        time_shard_size: zarr shard size along time; must be a multiple of
            ``time_chunk_size``.
    """

    path: str
    time_chunk_size: int = 1
    time_shard_size: int = 365

    def __post_init__(self):
        if self.time_shard_size % self.time_chunk_size != 0:
            raise ValueError(
                "time_shard_size must be a multiple of time_chunk_size; got "
                f"{self.time_shard_size} and {self.time_chunk_size}"
            )


@dataclasses.dataclass
class PipelineConfig:
    """Top-level configuration for one pipeline invocation (one output store).

    Attributes:
        streams: time-varying variable streams; all must share the same time
            coordinate.
        statics: static fields configuration.
        wetmask: source of the 3D ocean wetmask.
        target_grid: Gaussian target grid name (e.g. "F90").
        weights_url: URL prefix of the precomputed regridding weight artifact
            for the source grid x ``target_grid`` pair.
        output: output store layout.
        expected_level_count: number of vertical levels every 3D source
            variable must have.
        shift_timestamps_to_avg_interval_midpoint: if True, shift time labels
            backwards by half the timestep, the convention of legacy
            mean-state stores. Off for snapshot stores, whose raw timestamps
            are kept verbatim.
        start_time: optional inclusive time-range start (e.g. "0151-01-06"),
            applied to all streams.
        end_time: optional inclusive time-range end.
    """

    streams: list[StreamConfig]
    statics: StaticsConfig
    wetmask: WetmaskConfig
    target_grid: str
    weights_url: str
    output: OutputConfig
    expected_level_count: int = 19
    shift_timestamps_to_avg_interval_midpoint: bool = False
    start_time: str | None = None
    end_time: str | None = None

    def __post_init__(self):
        names = [stream.name for stream in self.streams]
        if len(set(names)) != len(names):
            raise ValueError(f"stream names must be unique; got {names}")


def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
