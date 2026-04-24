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

# Monthly-cadence forcings, interpolated to daily during processing.
# ``ts`` = surface temperature (SST over ocean, ice top temp over sea ice,
# land skin temp over land) — correct atmosphere-only lower boundary.
# ``siconc`` = sea-ice fraction (ocean-model grid; regridded to target).
FORCING_VARIABLES: list[str] = ["ts", "siconc"]

# Static per-model fields pulled from the CMIP6 ``fx`` table.
# ``sftlf`` = land fraction (land-sea mask).
# ``orog``  = surface altitude (orography).
STATIC_VARIABLES: list[str] = ["sftlf", "orog"]


def make_label(source_id: str, physics_index: int) -> str:
    """Compose the composite embedding label from (source_id, p).

    The embedding treats each ``(source_id, p)`` pair as a distinct
    "model" — realization (r), initialization (i), and forcing (f) are
    within-label variation. The physics index is always included, even
    for source_ids that publish only one ``p`` value, so the label
    maps 1:1 to CMIP6 metadata.
    """
    return f"{source_id}.p{physics_index}"


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
    # Gauss-Legendre grid spec; see grid.py / GAUSSIAN_GRID_N.
    # F<N> gives nlat=2N, nlon=4N. F22.5 is a ~4 deg grid (45 x 90).
    # Gauss-Legendre over regular lat-lon so spherical-harmonic
    # transforms on the processed data are exact.
    name: str = "F22.5"


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
    # Nearest-above-in-the-vertical fill for below-surface cells: each
    # column's below-surface levels inherit the lowest above-surface
    # level's value. (Not time persistence; just column-wise NN in plev.)
    strategy: str = "nearest_above"
    emit_mask: bool = True


@dataclass
class ChunkingConfig:
    # Inner zarr v3 chunk size along time. Per-timestep chunks match the
    # existing scripts/data_process convention and minimise read amplification
    # for training's short time windows.
    chunk_time: int = 1
    # Outer shard size along time. Groups inner chunks into a single GCS
    # object. None = unsharded (debug only). 365 = ~one shard per year.
    shard_time: Optional[int] = 365


@dataclass
class DefaultsConfig:
    core_variables: list[str] = field(default_factory=lambda: list(CORE_VARIABLES))
    optional_variables: list[str] = field(
        default_factory=lambda: list(OPTIONAL_VARIABLES)
    )
    forcing_variables: list[str] = field(
        default_factory=lambda: list(FORCING_VARIABLES)
    )
    static_variables: list[str] = field(default_factory=lambda: list(STATIC_VARIABLES))
    # Temporal interpolation for monthly forcings onto the daily time axis.
    forcing_interpolation: str = "linear"
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
    # Keep only this initialization_index (i). None = keep all i.
    require_i: Optional[int] = 1
    # Cap on realizations (r) within each label slice
    # (source_id, experiment, variant_p, variant_f). None = no cap.
    # When a label has multiple f values, the cap is applied per f, so a
    # model with 2 f values keeps up to 2 * max_members_per_f realizations.
    max_members_per_f: Optional[int] = 3


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
            forcing_variables=list(self.defaults.forcing_variables),
            static_variables=list(self.defaults.static_variables),
            forcing_interpolation=self.defaults.forcing_interpolation,
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
    forcing_variables: list[str]
    static_variables: list[str]
    forcing_interpolation: str
    time_subset: dict[str, TimeWindow]
    target_grid: TargetGrid
    regrid: RegridConfig
    fill: FillConfig
    chunking: ChunkingConfig


@dataclass
class CatalogQuery:
    """A single (table_id, variables) filter against the CMIP6 catalog."""

    table_id: str
    variables: list[str]
    # If False, this query ignores the top-level experiments filter — used
    # for ``fx`` (static) where the same data exists across experiments and
    # we just want it once per model.
    filter_by_experiment: bool = True


def _default_inventory_queries() -> list[CatalogQuery]:
    return [
        CatalogQuery(
            table_id="day",
            variables=(
                list(CORE_VARIABLES)
                + list(OPTIONAL_VARIABLES)
                + list(DIAGNOSTIC_VARIABLES)
            ),
        ),
        CatalogQuery(table_id="Amon", variables=["ts"]),
        CatalogQuery(table_id="SImon", variables=["siconc"]),
        CatalogQuery(
            table_id="fx",
            variables=list(STATIC_VARIABLES),
            filter_by_experiment=False,
        ),
    ]


@dataclass
class InventoryConfig:
    output_path: str  # fsspec URL to write the inventory table
    queries: list[CatalogQuery] = field(default_factory=_default_inventory_queries)
    experiments: list[str] = field(default_factory=lambda: ["historical", "ssp585"])

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
    "FORCING_VARIABLES",
    "STATIC_VARIABLES",
    "FLUX_LIKE_VARIABLES",
    "make_label",
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
    "CatalogQuery",
    "InventoryConfig",
]
