"""Config pieces shared by every pipeline: YAML loading and the dataclasses
whose schema does not depend on the pipeline.

Each pipeline's own config module (e.g. pipeline/om4/config.py) defines its
``PipelineConfig`` and parses :func:`load_yaml`'s output into it.
"""

import dataclasses

import yaml

# Token replaced by an ensemble-member name throughout a config's text.
MEMBER_PLACEHOLDER = "{member}"


def load_yaml(path: str, member: str | None = None) -> dict:
    """Read a YAML config, substituting ``member`` for MEMBER_PLACEHOLDER.

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
    return yaml.safe_load(text)


@dataclasses.dataclass
class WetmaskConfig:
    """Where the ocean wetmask comes from.

    The wetmask is the NaN pattern of the reference variable's first
    timestep. Every processed variable's valid-data footprint must equal it
    exactly (see pipeline.zarr_io.assert_footprint), so the output NaN
    pattern is the same at every timestep. Whether the reference variable is
    3D (level, y, x) or 2D (y, x) is the pipeline's choice.

    Attributes:
        store: URL of the zarr store holding the reference variable.
        variable: name of the variable whose NaN pattern defines the wetmask.
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
