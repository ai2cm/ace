"""YAML-driven configuration for the OM4 ocean dataset pipeline.

Each YAML file describes one output store: the source zarr stores and the
streams of variables to read from them, the transforms to apply (vector
rotation, level splitting, time subsampling or coarsening), the target grid
and weight artifact, and the output layout. An ensemble's members differ
only in a name inside URLs, so a config may carry the ``{member}``
placeholder there and describe one store per member (see
pipeline/config_io.py). Transforms are code; configs only select and
parameterize them, so a new simulation or a new output store is a config
change, not a code change.
"""

import dataclasses

import dacite

from ..config_io import OutputConfig, StaticsConfig, WetmaskConfig, load_yaml
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
        time_block_mean_factor: if set, average every N consecutive timesteps,
            in N-step blocks aligned to the source store's raw time grid like
            ``time_subsample_stride`` (e.g. 5 coarsens daily means to 5-day
            means), labeling each block mean with its block's final
            timestamp. NaN anywhere in a block propagates to the mean, so a
            time-varying footprint still fails the wetmask assertion. N must
            evenly divide the source store's time axis (a trailing partial
            block raises rather than being dropped). Mutually exclusive with
            ``time_subsample_stride`` and with the pipeline's
            ``shift_timestamps_to_avg_interval_midpoint``.
        full_cell_variables: variables additionally regridded with full-cell
            semantics — zeroed over land (wetmask) and conservatively
            regridded without ocean-fraction normalization — written under
            their source name, with NaN over land applied after. Each must
            also have a ``renaming`` entry so its wetmask-normalized twin
            doesn't collide.
        postprocess: post-regrid transforms to apply per chunk, in order
            (see pipeline/om4/postprocess.py): a registry name, or a
            :class:`~pipeline.postprocess.PostprocessConfig` naming the
            variables the transform reads (``kelvin_sst`` needs
            ``sources: {celsius_sst: <name>}``).
        drop_outputs: output names (see :meth:`output_names`) removed from
            each chunk after all postprocess transforms, e.g. a source a
            transform replaces.
        ice_area_weighted: per-ice-area source variables (e.g. ``HI``,
            ``TS``) regridded weighted by ice area (rule R1, see
            pipeline/om4/ice_area.py) instead of by ocean area: 0 where the
            regridded weight is 0 over ocean, NaN over land. Subsampled or
            native-resolution streams only (no ``time_block_mean_factor``).
        ice_area_weight: source name of the ice-area fraction ``c`` the
            ``ice_area_weighted`` variables are weighted by. If not in
            ``variables``, it is read for weighting only and not written.
        ice_area_weight_store: URL of a store to read ``ice_area_weight``
            from instead of ``store``; its raw time axis must equal
            ``store``'s.
        ice_area_products: mapping of an ``ice_area_weighted`` variable ``X``
            to an output name for the ocean-area mean of ``c X`` (e.g.
            ``HI`` -> ice volume per ocean area).
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
    time_block_mean_factor: int | None = None
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
    postprocess: list[str | PostprocessConfig] = dataclasses.field(default_factory=list)
    face_mask_url: str | None = None
    drop_outputs: list[str] = dataclasses.field(default_factory=list)
    ice_area_weighted: list[str] = dataclasses.field(default_factory=list)
    ice_area_weight: str | None = None
    ice_area_weight_store: str | None = None
    ice_area_products: dict[str, str] = dataclasses.field(default_factory=dict)

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
        if self.time_block_mean_factor is not None:
            if self.time_block_mean_factor < 1:
                raise ValueError(
                    f"time_block_mean_factor must be >= 1; got "
                    f"{self.time_block_mean_factor}"
                )
            if self.time_subsample_stride is not None:
                raise ValueError(
                    f"stream {self.name!r} sets both time_subsample_stride and "
                    "time_block_mean_factor; choose one"
                )
            if "sea_ice" in self.postprocess_names():
                # sea_ice_volume from block means is mean_t(HI) * mean_t(c),
                # not mean_t(HI * c).
                raise ValueError(
                    f"stream {self.name!r} applies sea_ice postprocess to a "
                    "block-meaned stream (time_block_mean_factor); use "
                    "time_subsample_stride"
                )
            if self.ice_area_weighted:
                # Rule R2 (block means of c_t X_t) is not implemented; R1
                # here would weight by mean_t(c).
                raise ValueError(
                    f"stream {self.name!r} sets ice_area_weighted on a "
                    "block-meaned stream (time_block_mean_factor); use "
                    "time_subsample_stride"
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
        self._check_ice_area()
        context = f"stream {self.name!r}"
        output_names = {self.renaming.get(name, name) for name in self.variables}
        output_names.update(self.full_cell_variables)
        assert_postprocess_inputs(
            self.postprocess_specs(),
            output_names,
            context,
            allow_level_suffix=True,
        )
        unknown = set(self.drop_outputs) - self.output_names()
        if unknown:
            raise ValueError(
                f"drop_outputs {sorted(unknown)} not among stream {self.name!r} "
                f"outputs {sorted(self.output_names())}"
            )
        if self.face_mask_url is not None and not self.rotated_pairs:
            raise ValueError(
                f"stream {self.name!r} sets face_mask_url but has no "
                "rotated_pairs for it to apply to"
            )

    def _check_ice_area(self) -> None:
        context = f"stream {self.name!r}"
        for name in self.ice_area_weighted:
            if name not in self.variables:
                raise ValueError(
                    f"ice_area_weighted variable {name!r} not in {context} variables"
                )
            if name in self.full_cell_variables:
                raise ValueError(
                    f"ice_area_weighted variable {name!r} is a full_cell_variable "
                    f"in {context}"
                )
        weight = self.ice_area_weight
        if self.ice_area_weighted and weight is None:
            raise ValueError(
                f"{context} sets ice_area_weighted without ice_area_weight"
            )
        if weight is not None and weight in self.ice_area_weighted:
            raise ValueError(
                f"{context}: ice_area_weight {weight!r} is in ice_area_weighted"
            )
        if self.ice_area_weight_store is not None:
            if weight is None:
                raise ValueError(
                    f"{context} sets ice_area_weight_store without ice_area_weight"
                )
            if weight in self.variables:
                raise ValueError(
                    f"{context}: ice_area_weight {weight!r} is in variables, so it "
                    "is read from store, not ice_area_weight_store"
                )
        unweighted = set(self.ice_area_products) - set(self.ice_area_weighted)
        if unweighted:
            raise ValueError(
                f"{context}: ice_area_products keys {sorted(unweighted)} not in "
                "ice_area_weighted"
            )
        products = list(self.ice_area_products.values())
        others = self._base_output_names()
        weight_only = self.weight_only()
        if weight_only is not None:
            others.add(weight_only)
        collisions = {
            name for name in products if name in others or products.count(name) > 1
        }
        if collisions:
            raise ValueError(
                f"{context}: ice_area_products names {sorted(collisions)} collide "
                "with other names of the stream"
            )

    def _base_output_names(self) -> set[str]:
        names = {self.renaming.get(name, name) for name in self.variables}
        names.update(self.full_cell_variables)
        for spec in self.postprocess_specs():
            names.update(spec.adds)
        return names

    def output_names(self) -> set[str]:
        """Output base names (3D variables unsplit): renamed variables,
        full-cell variables, postprocess additions and ice-area products.
        A weight-only ``ice_area_weight`` is not written."""
        return self._base_output_names() | set(self.ice_area_products.values())

    def weight_only(self) -> str | None:
        """The ``ice_area_weight`` if it is read for weighting only."""
        if self.ice_area_weight is None or self.ice_area_weight in self.variables:
            return None
        return self.ice_area_weight

    def postprocess_names(self) -> list[str]:
        """Registry names of the configured transforms, in order."""
        return [
            entry if isinstance(entry, str) else entry.name
            for entry in self.postprocess
        ]

    def postprocess_specs(self) -> list[Postprocess]:
        """The configured transforms, with their source names bound."""
        return resolve_postprocess(
            POSTPROCESS, self.postprocess, f"stream {self.name!r}"
        )


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
            are kept verbatim. Not allowed when any stream sets
            ``time_block_mean_factor``.
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
        if self.shift_timestamps_to_avg_interval_midpoint:
            block_mean_streams = [
                stream.name
                for stream in self.streams
                if stream.time_block_mean_factor is not None
            ]
            if block_mean_streams:
                raise ValueError(
                    "shift_timestamps_to_avg_interval_midpoint cannot be "
                    "combined with time_block_mean_factor; set on streams "
                    f"{block_mean_streams}"
                )


def load_config(path: str, member: str | None = None) -> PipelineConfig:
    """Load a config, substituting ``member`` for the ``{member}`` placeholder
    (see pipeline.config_io.load_yaml for the substitution rules)."""
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=load_yaml(path, member),
        config=dacite.Config(strict=True),
    )
