"""YAML-driven configuration for the sea-surface dataset pipeline.

One config plus one ensemble-member name describes one output store: the
source store and the variables to read from it, the target grid and weight
artifact, and the output layout. The members of an ensemble differ only in a
name that appears inside URLs, so the config carries the MEMBER_PLACEHOLDER
token wherever that name belongs and :func:`load_config` substitutes it —
one reviewable config instead of a family of near-identical ones.

Transforms are code; configs only select and parameterize them, so a new
simulation or a new output store is a config change, not a code change.
"""

import dataclasses

import dacite

from ..config_io import MEMBER_PLACEHOLDER, OutputConfig, WetmaskConfig, load_yaml
from ..postprocess import (
    Postprocess,
    PostprocessConfig,
    assert_postprocess_inputs,
    resolve_postprocess,
)
from .postprocess import POSTPROCESS

__all__ = [
    "MEMBER_PLACEHOLDER",
    "OutputConfig",
    "PipelineConfig",
    "StreamConfig",
    "WetmaskConfig",
    "load_config",
]


@dataclasses.dataclass
class StreamConfig:
    """The stream of time-varying variables read from the source store.

    Attributes:
        name: label used in logging and beam stage names.
        store: URL of the source zarr store.
        variables: source variable names to process.
        renaming: mapping of source names to output names.
        dim_renaming: mapping of source dimension names to the ocean-grid
            tracer names the regridding machinery expects (``xh``/``yh``).
        full_cell_variables: variables additionally regridded with full-cell
            semantics — NaN filled with 0 over the whole grid (land
            included) and conservatively regridded without ocean-fraction
            normalization — written under their source name, with NaN over
            land applied after. Each must also have a ``renaming`` entry so
            its wetmask-normalized twin doesn't collide.
        postprocess: post-regrid transforms to apply per chunk, in order
            (see pipeline/randco2/postprocess.py): a registry name, or a
            :class:`~pipeline.postprocess.PostprocessConfig` naming the
            variables the transform reads (``kelvin_sst`` needs
            ``sources: {celsius_sst: <name>}``).
    """

    name: str
    store: str
    variables: list[str]
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    dim_renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
    postprocess: list[str | PostprocessConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self):
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
        )

    def postprocess_specs(self) -> list[Postprocess]:
        """The configured transforms, with their source names bound."""
        return resolve_postprocess(
            POSTPROCESS, self.postprocess, f"stream {self.name!r}"
        )


@dataclasses.dataclass
class PipelineConfig:
    """Top-level configuration for one pipeline invocation (one output store).

    Attributes:
        stream: the time-varying variable stream.
        wetmask: source of the 2D ocean wetmask.
        target_grid: Gaussian target grid name (e.g. "F90").
        weights_url: URL prefix of the precomputed regridding weight artifact
            for the source grid x ``target_grid`` pair.
        output: output store layout.
        start_time: optional inclusive time-range start (e.g. "0152-10-01").
        end_time: optional inclusive time-range end.
    """

    stream: StreamConfig
    wetmask: WetmaskConfig
    target_grid: str
    weights_url: str
    output: OutputConfig
    start_time: str | None = None
    end_time: str | None = None


def load_config(path: str, member: str | None = None) -> PipelineConfig:
    """Load a config, substituting ``member`` for MEMBER_PLACEHOLDER (see
    pipeline.config_io.load_yaml for the substitution rules)."""
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=load_yaml(path, member),
        config=dacite.Config(strict=True),
    )
