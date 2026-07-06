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
    """

    name: str
    store: str
    variables: list[str]
    rotated_pairs: list[list[str]] = dataclasses.field(default_factory=list)
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)

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


@dataclasses.dataclass
class WetmaskConfig:
    """Where the 3D ocean wetmask comes from.

    The wetmask is the NaN pattern of the reference variable's first
    timestep. Every processed variable's valid-data footprint is asserted
    against it, so a source whose footprint disagrees fails loudly instead
    of being silently clipped or zero-filled.

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
