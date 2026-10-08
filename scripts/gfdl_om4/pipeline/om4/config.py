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
        full_cell_variables: variables additionally regridded with full-cell
            semantics — NaN filled with 0 over the whole grid (land
            included) and conservatively regridded without ocean-fraction
            normalization — written under their source name, with NaN over
            land applied after. Each must also have a ``renaming`` entry so
            its wetmask-normalized twin doesn't collide.
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
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
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
        context = f"stream {self.name!r}"
        output_names = {self.renaming.get(name, name) for name in self.variables}
        output_names.update(self.full_cell_variables)
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


def load_config(path: str) -> PipelineConfig:
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=load_yaml(path),
        config=dacite.Config(strict=True),
    )
