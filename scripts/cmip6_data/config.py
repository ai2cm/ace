"""Config dataclasses for the CMIP6 daily pilot scripts.

Two top-level configs:
- InventoryConfig: input to ``inventory.py`` — what to discover.
- ProcessConfig: input to ``process.py`` — what to process and how.

Both load from YAML via :meth:`from_file` using dacite.
"""

from dataclasses import dataclass, field
from typing import Optional

import dacite
import yaml

CORE_VARIABLES: list[str] = [
    # 3D state on plev8 (required).
    # ``ta`` is excluded because Pangeo daily coverage is essentially nil
    # (3 models); layer-mean T is derived from zg + hus in the processing
    # step instead and stored as ta_derived_layer_{0..6}.
    "ua",
    "va",
    "hus",
    "zg",
    # 2D state (required).
    # ``ps`` is excluded because no CMIP6 model publishes surface
    # pressure at daily cadence; ``psl`` + a topography mask substitutes.
    "tas",
    "huss",
    "psl",
    "pr",
]

OPTIONAL_VARIABLES: list[str] = [
    # TOA radiation
    "rsdt",
    "rsut",
    "rlut",
    # Surface radiation
    "rsds",
    "rsus",
    "rlds",
    "rlus",
    # Surface turbulent fluxes
    "hfss",
    "hfls",
    # Surface wind
    "sfcWind",
    "uas",
    "vas",
]

# Variables tracked by the inventory for visibility, but not required or
# used in processing. ``ta`` and ``ps`` are here so inventory summaries
# show their (sparse) coverage even though we substitute them with
# derived / proxy fields.
DIAGNOSTIC_VARIABLES: list[str] = ["ta", "ps"]

# Variables regridded conservatively; everything else is bilinear by default.
FLUX_LIKE_VARIABLES: frozenset[str] = frozenset(
    {
        "pr",
        "rsdt",
        "rsut",
        "rlut",
        "rsds",
        "rsus",
        "rlds",
        "rlus",
        "hfss",
        "hfls",
    }
)


@dataclass
class TimeWindow:
    start: str  # ISO date, e.g. "2000-01-01"
    end: str  # ISO date, inclusive


@dataclass
class TargetGrid:
    lat_step: float = 4.0
    lon_step: float = 4.0


@dataclass
class RegridConfig:
    default_state: str = "bilinear"
    default_flux: str = "conservative"
    per_variable: dict[str, str] = field(default_factory=dict)

    def method_for(self, variable: str) -> str:
        if variable in self.per_variable:
            return self.per_variable[variable]
        if variable in FLUX_LIKE_VARIABLES:
            return self.default_flux
        return self.default_state


@dataclass
class FillConfig:
    strategy: str = "persistence_down"
    emit_mask: bool = True


@dataclass
class ChunkingConfig:
    time: int = 32


@dataclass
class DefaultsConfig:
    core_variables: list[str] = field(default_factory=lambda: list(CORE_VARIABLES))
    optional_variables: list[str] = field(
        default_factory=lambda: list(OPTIONAL_VARIABLES)
    )
    time_subset: dict[str, TimeWindow] = field(default_factory=dict)
    target_grid: TargetGrid = field(default_factory=TargetGrid)
    regrid: RegridConfig = field(default_factory=RegridConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)


@dataclass
class Match:
    """Matcher for a subset of (source_id, experiment, variant_label) rows.

    ``source_id`` is required; other fields default to None, which matches
    every value for that field.
    """

    source_id: str
    experiment: Optional[str] = None
    variant_label: Optional[str] = None

    def matches(self, source_id: str, experiment: str, variant_label: str) -> bool:
        if self.source_id != source_id:
            return False
        if self.experiment is not None and self.experiment != experiment:
            return False
        if self.variant_label is not None and self.variant_label != variant_label:
            return False
        return True


@dataclass
class Override:
    match: Match
    time_subset: Optional[dict[str, TimeWindow]] = None


@dataclass
class Selection:
    source_ids: Optional[list[str]] = None
    experiments: list[str] = field(default_factory=lambda: ["historical", "ssp585"])
    # Max members per (source_id, experiment); None = no cap.
    max_members: Optional[int] = None


@dataclass
class ProcessConfig:
    inventory_path: str
    output_directory: str
    index_path: str
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    selection: Selection = field(default_factory=Selection)
    overrides: list[Override] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "ProcessConfig":
        return _load_yaml(cls, path)

    def resolve(
        self, source_id: str, experiment: str, variant_label: str
    ) -> "ResolvedDatasetConfig":
        """Apply overrides on top of defaults for one dataset."""
        time_subset = dict(self.defaults.time_subset)
        for override in self.overrides:
            if override.match.matches(source_id, experiment, variant_label):
                if override.time_subset is not None:
                    time_subset = dict(override.time_subset)
        return ResolvedDatasetConfig(
            source_id=source_id,
            experiment=experiment,
            variant_label=variant_label,
            core_variables=list(self.defaults.core_variables),
            optional_variables=list(self.defaults.optional_variables),
            time_subset=time_subset,
            target_grid=self.defaults.target_grid,
            regrid=self.defaults.regrid,
            fill=self.defaults.fill,
            chunking=self.defaults.chunking,
        )


@dataclass
class ResolvedDatasetConfig:
    """Defaults + overrides flattened for a single dataset. Not loaded
    from YAML — produced by ``ProcessConfig.resolve``.
    """

    source_id: str
    experiment: str
    variant_label: str
    core_variables: list[str]
    optional_variables: list[str]
    time_subset: dict[str, TimeWindow]
    target_grid: TargetGrid
    regrid: RegridConfig
    fill: FillConfig
    chunking: ChunkingConfig


@dataclass
class InventoryConfig:
    output_path: str  # fsspec URL to write the inventory table
    variables: list[str] = field(
        default_factory=lambda: (
            list(CORE_VARIABLES) + list(OPTIONAL_VARIABLES) + list(DIAGNOSTIC_VARIABLES)
        )
    )
    experiments: list[str] = field(default_factory=lambda: ["historical", "ssp585"])
    table_id: str = "day"

    @classmethod
    def from_file(cls, path: str) -> "InventoryConfig":
        return _load_yaml(cls, path)


def _load_yaml(cls, path: str):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return dacite.from_dict(
        data_class=cls, data=data, config=dacite.Config(cast=[tuple])
    )


__all__ = [
    "CORE_VARIABLES",
    "OPTIONAL_VARIABLES",
    "DIAGNOSTIC_VARIABLES",
    "FLUX_LIKE_VARIABLES",
    "TimeWindow",
    "TargetGrid",
    "RegridConfig",
    "FillConfig",
    "ChunkingConfig",
    "DefaultsConfig",
    "Match",
    "Override",
    "Selection",
    "ProcessConfig",
    "ResolvedDatasetConfig",
    "InventoryConfig",
]
