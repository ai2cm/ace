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
import yaml

from .postprocess import POSTPROCESS

# Token replaced by the ensemble-member name throughout a config's text.
MEMBER_PLACEHOLDER = "{member}"


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
        postprocess: named post-regrid transforms to apply per chunk, in
            order (see pipeline/postprocess.py).
    """

    name: str
    store: str
    variables: list[str]
    renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    dim_renaming: dict[str, str] = dataclasses.field(default_factory=dict)
    full_cell_variables: list[str] = dataclasses.field(default_factory=list)
    postprocess: list[str] = dataclasses.field(default_factory=list)

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
        for name in self.postprocess:
            if name not in POSTPROCESS:
                raise ValueError(
                    f"unknown postprocess {name!r} in stream {self.name!r}; "
                    f"available: {sorted(POSTPROCESS)}"
                )


@dataclasses.dataclass
class WetmaskConfig:
    """Where the 2D ocean wetmask comes from.

    The wetmask is the NaN pattern of the reference variable's first
    timestep. Every processed variable's valid-data footprint is asserted to
    equal it (see run._assert_footprint), so the output NaN pattern is the
    same at every timestep and a source whose footprint disagrees fails
    loudly instead of being silently zero-filled by the normalized regrid.

    Attributes:
        store: URL of the zarr store holding the reference variable.
        variable: name of a 2D (y, x) variable whose NaN pattern defines the
            wetmask.
    """

    store: str
    variable: str


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
    """Load a config, substituting ``member`` for MEMBER_PLACEHOLDER.

    Substitution happens on the raw text before parsing, so a placeholder may
    appear anywhere in a URL. A config carrying the placeholder requires a
    member; one without it rejects a member, since the name would then have
    no effect on the output store.
    """
    with open(path) as f:
        text = f.read()
    has_placeholder = MEMBER_PLACEHOLDER in text
    if has_placeholder and member is None:
        raise ValueError(
            f"{path} contains {MEMBER_PLACEHOLDER}; an ensemble-member name "
            "is required to resolve it"
        )
    if member is not None:
        if not has_placeholder:
            raise ValueError(
                f"member {member!r} was given but {path} contains no "
                f"{MEMBER_PLACEHOLDER} for it to fill, so it would not "
                "change the output store"
            )
        text = text.replace(MEMBER_PLACEHOLDER, member)
    return dacite.from_dict(
        data_class=PipelineConfig,
        data=yaml.safe_load(text),
        config=dacite.Config(strict=True),
    )
