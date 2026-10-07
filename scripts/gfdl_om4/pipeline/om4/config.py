"""YAML-driven configuration for the OM4 ocean dataset pipeline.

Each YAML file describes one output store: the source zarr stores and the
streams of variables to read from them, the transforms to apply (vector
rotation, level splitting), the target grid and weight artifact, and the
output layout. Transforms are code; configs only select and parameterize
them, so a new simulation or a new output store is a config change, not a
code change.
"""

import dataclasses

import dacite

from ..config_io import OutputConfig, WetmaskConfig, load_yaml
from ..postprocess import (
    Postprocess,
    PostprocessConfig,
    assert_postprocess_inputs,
    resolve_postprocess,
)
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
        time_block_mean: if set, average each N-step block of the source
            store's raw time grid (blocks anchored at its first step, a
            trailing partial block dropped), labeled by the block's last
            instant — the same instants ``time_subsample_stride: N`` keeps.
            Excludes ``time_subsample_stride``.
        full_cell_variables: variables additionally regridded with full-cell
            semantics — NaN filled with 0 over the whole grid (land
            included) and conservatively regridded without ocean-fraction
            normalization — written under their source name, with NaN over
            land applied after. Each must also have a ``renaming`` entry so
            its wetmask-normalized twin doesn't collide.
        full_cell_only: if True, every variable is written only with
            full-cell semantics, under its ``renaming`` entry if any, else its
            source name; no wetmask-normalized twin. ``full_cell_variables``
            must then list every variable.
        postprocess: post-regrid transforms to apply per chunk, in order
            (see pipeline/om4/postprocess.py): a registry name, or a
            :class:`~pipeline.postprocess.PostprocessConfig` naming the
            variables the transform reads (``kelvin_sst`` needs
            ``sources: {celsius_sst: <name>}``).
        face_mask_url: URL prefix of a precomputed face-mask artifact (see
            pipeline/om4/face_masks.py) for sources whose staggered velocities
            carry remap-born zeros over land. When set, the flagged faces
            of the stream's rotated pairs are treated as invalid before
            center interpolation (see run._rotate_pairs).
    """

    name: str
    store: str
    variables: list[str]
    rotated_pairs: list[list[str]] = dataclasses.field(default_factory=list)
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    dim_renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    time_subsample_stride: int | None = None
    time_block_mean: int | None = None
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
    full_cell_only: bool = False
    postprocess: list[str | PostprocessConfig] = dataclasses.field(default_factory=list)
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
        if self.time_block_mean is not None:
            if self.time_block_mean < 1:
                raise ValueError(
                    f"time_block_mean must be >= 1; got {self.time_block_mean}"
                )
            if self.time_subsample_stride is not None:
                raise ValueError(
                    f"stream {self.name!r} sets both time_subsample_stride and "
                    "time_block_mean; choose one"
                )
        if self.full_cell_only and set(self.full_cell_variables) != set(self.variables):
            raise ValueError(
                f"stream {self.name!r} is full_cell_only, so full_cell_variables "
                f"must list every variable; missing "
                f"{sorted(set(self.variables) - set(self.full_cell_variables))}"
            )
        for name in self.full_cell_variables:
            if name not in self.variables:
                raise ValueError(
                    f"full-cell variable {name!r} not in stream {self.name!r} "
                    "variables"
                )
            if name not in self.renaming and not self.full_cell_only:
                raise ValueError(
                    f"full-cell variable {name!r} needs a renaming entry in "
                    f"stream {self.name!r}: its full-cell output keeps the "
                    "source name, so the wetmask-normalized output must be "
                    "renamed to avoid a collision"
                )
        context = f"stream {self.name!r}"
        output_names = self.output_names(self.variables)
        assert_postprocess_inputs(
            self.postprocess_specs(),
            output_names,
            context,
            allow_level_suffix=True,
        )
        if self.face_mask_url is not None and not self.rotated_pairs:
            raise ValueError(
                f"stream {self.name!r} sets face_mask_url but has no "
                "rotated_pairs for it to apply to"
            )

    def full_cell_output_name(self, name: str) -> str:
        """Output name of ``name``'s full-cell regrid."""
        return self.renaming.get(name, name) if self.full_cell_only else name

    def output_names(self, names_2d, names_3d=(), level_count: int = 0) -> set[str]:
        """Regridded output names before postprocess additions."""
        names = {self.full_cell_output_name(name) for name in self.full_cell_variables}
        if self.full_cell_only:
            return names
        names.update(self.renaming.get(name, name) for name in names_2d)
        names.update(
            f"{self.renaming.get(name, name)}_{k}"
            for name in names_3d
            for k in range(level_count)
        )
        return names

    def postprocess_specs(self) -> list[Postprocess]:
        """The configured transforms, with their source names bound."""
        return resolve_postprocess(
            POSTPROCESS, self.postprocess, f"stream {self.name!r}"
        )


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
class PipelineConfig:
    """Top-level configuration for one pipeline invocation (one output store).

    Attributes:
        streams: time-varying variable streams; all must share the same time
            coordinate.
        statics: static fields configuration.
        wetmask: source of the 3D ocean wetmask: a (level, y, x) reference
            variable (see run.load_wetmask).
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
        if self.shift_timestamps_to_avg_interval_midpoint and any(
            stream.time_block_mean is not None for stream in self.streams
        ):
            raise ValueError(
                "shift_timestamps_to_avg_interval_midpoint is unsupported with "
                "time_block_mean streams, whose labels are block-end instants"
            )


def load_config(path: str) -> PipelineConfig:
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=load_yaml(path),
        config=dacite.Config(strict=True),
    )
