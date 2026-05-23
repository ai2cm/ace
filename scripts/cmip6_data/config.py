"""Config dataclasses for the CMIP6 daily pilot scripts.

Two top-level configs:
- InventoryConfig: input to ``inventory.py`` — what to discover.
- ProcessConfig: input to ``process.py`` — what to process and how.

Both load from YAML via :meth:`from_file` using dacite.

Variable naming convention. Variables whose CMIP6 source table is dataset-
dependent (i.e., the same physical quantity is published at different
cadences or from different tables by different models) get a
``{table}_{var}`` output name; variables with a single canonical source
keep the bare CMIP6 name. See ``SURFACE_AND_OCEAN_VARIABLES`` for the prefixed set.
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
    # TOA radiation. ``rsdt``/``rsut`` (and their clear-sky pair
    # ``rsutcs``, plus ``rlutcs``) aren't published on the standard
    # ``day`` table by *any* CMIP6 model — they live on ``CFday`` (see
    # ``CFDAY_VARIABLES``) and are folded into the day-cadence variable
    # list by the inventory + task-building code.
    "rsdt",
    "rsut",
    "rlut",
    "rsutcs",
    "rlutcs",
    # Surface radiation (all-sky + clear-sky; clear-sky are on CFday).
    "rsds",
    "rsus",
    "rlds",
    "rlus",
    "rsdscs",
    "rsuscs",
    "rldscs",
    # Surface turbulent fluxes
    "hfss",
    "hfls",
    # Surface wind
    "sfcWind",
    "uas",
    "vas",
]

# Day-cadence variables that CMIP6 publishes on the ``CFday`` table
# rather than ``day``. We treat them as ordinary day-cadence variables
# (no causal mapping needed — they're already daily means); the table
# distinction only matters at inventory + open time. ``rsdt`` is
# astronomically derivable but ``rsut`` is not (depends on cloud /
# surface albedo), so we ingest both from CFday rather than computing.
# Clear-sky pairs (``*cs``) are on CFday too — they enable cloud
# radiative-effect calculations downstream. No ``rluscs`` — surface
# upward LW is determined by surface T + emissivity, so all-sky and
# clear-sky values are identical and CMIP6 doesn't publish a separate
# clear-sky version.
CFDAY_VARIABLES: list[str] = [
    "rsdt",
    "rsut",
    "rsutcs",
    "rlutcs",
    "rsdscs",
    "rsuscs",
    "rldscs",
]


def cmip6_source_table(variable_id: str) -> str:
    """Return the CMIP6 table on which ``variable_id`` lives in our
    pipeline. Defaults to ``day``; overrides for the CFday-sourced
    radiation variables.
    """
    if variable_id in CFDAY_VARIABLES:
        return "CFday"
    return "day"


# Renames applied to the final output variables so that radiative-flux
# names match the convention used by upstream baseline datasets
# (SHIELD AMIP, ERA5, etc.). Pipelines that consume both should be able
# to point at the same name and find the same physical quantity. The
# rename is applied just before write; each renamed variable carries
# an ``original_name`` attribute with the bare CMIP6 source name so
# the provenance back to CMIP6 is preserved. Surface-and-ocean
# variables keep their source-prefixed names (``eday_ts``,
# ``simon_sea_ice_fraction``, etc.) because the prefix encodes useful
# cadence/source information that the baseline convention drops.
CMIP_TO_OUTPUT_RENAMES: dict[str, str] = {
    # Surface radiation: down/upward longwave/shortwave at the surface.
    "rlds": "DLWRFsfc",
    "rlus": "ULWRFsfc",
    "rsds": "DSWRFsfc",
    "rsus": "USWRFsfc",
    # TOA radiation: down/upward shortwave + upward longwave (OLR).
    "rsdt": "DSWRFtoa",
    "rsut": "USWRFtoa",
    "rlut": "ULWRFtoa",
    # Clear-sky radiation (TOA + surface). Naming convention inserts
    # ``C`` after the direction letter — matches the ERA5 build
    # pipeline. ``rsdt`` has no clear-sky variant because TOA incident
    # SW is purely astronomical.
    "rsutcs": "UCSWRFtoa",
    "rlutcs": "UCLWRFtoa",
    "rsdscs": "DCSWRFsfc",
    "rsuscs": "UCSWRFsfc",
    "rldscs": "DCLWRFsfc",
    # Turbulent fluxes (CMIP6 ``hfls``/``hfss`` are positive upward,
    # which matches the SHIELD/ERA5 convention after the ERA5 build
    # pipeline negates the ECMWF-native sign — see
    # ``scripts/era5/pipeline/xr-beam-pipeline.py``).
    "hfls": "LHTFLsfc",
    "hfss": "SHTFLsfc",
    # Surface precipitation rate (kg m-2 s-1 in both conventions).
    "pr": "PRATEsfc",
    # Near-surface atmosphere — CMIP6 CMOR fixes these at 2 m / 10 m,
    # matching the baseline naming.
    "tas": "TMP2m",
    "huss": "Q2m",
    "uas": "UGRD10m",
    "vas": "VGRD10m",
    # Geopotential height at 500 hPa. Other plev levels keep the
    # ``zg{level}`` form (the baseline ERA5 dataset publishes
    # ``h{level}`` for several levels; widen this later if we want
    # the rest aligned too).
    "zg500": "h500",
}

# Variables tracked by the inventory for visibility, but not required or
# used in processing. ``ta`` and ``ps`` are here so inventory summaries
# show their (sparse) coverage even though we substitute them with
# derived / proxy fields.
DIAGNOSTIC_VARIABLES: list[str] = ["ta", "ps"]

# Static per-model fields pulled from the CMIP6 ``fx`` table.
# ``sftlf`` = land fraction (land-sea mask).
# ``orog``  = surface altitude (orography).
STATIC_VARIABLES: list[str] = ["sftlf", "orog"]


# Variable "kinds" used to drive per-variable processing:
# - ``atmos_surface``: global field, no per-cell mask (e.g. Amon.ts, Eday.ts).
# - ``ocean_surface``: NaN over land — requires per-cell mask + horizontal
#   fill (e.g. Oday.tos, Omon.zos).
# - ``seaice_surface``: NaN over land and where the source publishes no ice
#   — requires per-cell mask + horizontal fill (e.g. SIday.siconc).
_SURFACE_AND_OCEAN_KINDS = ("atmos_surface", "ocean_surface", "seaice_surface")
_SURFACE_AND_OCEAN_CADENCES = ("daily", "monthly_causal")


@dataclass(frozen=True)
class SurfaceAndOceanVariable:
    """One source-prefixed surface-and-ocean variable.

    Variables whose CMIP6 source table varies across datasets (e.g. some
    models publish daily ``Eday.ts``, others only monthly ``Amon.ts``) get
    a source-prefixed output name so multiple cadences/tables can coexist
    in a single dataset. The training-side mask handles which ones are
    actually populated for a given model.

    Attributes:
        table_id: CMIP6 table the variable comes from (e.g. ``Eday``).
        var_id: CMIP6 variable name (e.g. ``ts``).
        output_name: Name in the output zarr (e.g. ``eday_ts``).
        cadence: ``daily`` (source matches the daily target axis) or
            ``monthly_causal`` (monthly source, mapped to daily via
            previous-month-mean — strictly causal, no future leakage).
        kind: ``atmos_surface``, ``ocean_surface``, or ``seaice_surface``.
            Drives mask emission and fill scheme; see ``_SURFACE_AND_OCEAN_KINDS``.
    """

    table_id: str
    var_id: str
    output_name: str
    cadence: str
    kind: str
    # Multiplicative scale applied to the source values before writing.
    # Set to 0.01 for sea-ice fractions (CMIP6 ``siconc`` is a percent;
    # we emit fractions on [0, 1] for consistency with land/ocean
    # fraction). Default 1.0 = no rescaling.
    unit_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _SURFACE_AND_OCEAN_KINDS:
            raise ValueError(f"kind={self.kind!r} not in {_SURFACE_AND_OCEAN_KINDS}")
        if self.cadence not in _SURFACE_AND_OCEAN_CADENCES:
            raise ValueError(
                f"cadence={self.cadence!r} not in {_SURFACE_AND_OCEAN_CADENCES}"
            )


# Surface-and-ocean variables — full set. Pipelines pull the
# subset that the source catalog actually publishes for each dataset; any
# variable missing from a dataset is simply absent in the output zarr
# (training handles via per-sample masking).
SURFACE_AND_OCEAN_VARIABLES: tuple[SurfaceAndOceanVariable, ...] = (
    # Surface temperature.
    # ``Amon.ts`` is the universal monthly fallback (model's own correct
    # SST + ice-top + skin composite). ``Eday.ts`` is the same quantity
    # daily — drop-in upgrade where published.
    SurfaceAndOceanVariable("Amon", "ts", "amon_ts", "monthly_causal", "atmos_surface"),
    SurfaceAndOceanVariable("Eday", "ts", "eday_ts", "daily", "atmos_surface"),
    # Sea-ice fraction & top temperature (monthly + daily). Both NaN over
    # land; daily is broadly published on ESGF (~25/37).
    SurfaceAndOceanVariable(
        "SImon",
        "siconc",
        "simon_sea_ice_fraction",
        "monthly_causal",
        "seaice_surface",
        unit_scale=0.01,  # CMIP6 ``siconc`` is %; emit [0, 1] fraction.
    ),
    SurfaceAndOceanVariable(
        "SImon", "sitemptop", "simon_sitemptop", "monthly_causal", "seaice_surface"
    ),
    SurfaceAndOceanVariable(
        "SIday",
        "siconc",
        "siday_sea_ice_fraction",
        "daily",
        "seaice_surface",
        unit_scale=0.01,
    ),
    SurfaceAndOceanVariable(
        "SIday", "sitemptop", "siday_sitemptop", "daily", "seaice_surface"
    ),
    SurfaceAndOceanVariable(
        "SIday", "sithick", "siday_sithick", "daily", "seaice_surface"
    ),
    # Daily ocean surface (Oday). NaN over land.
    SurfaceAndOceanVariable("Oday", "tos", "oday_tos", "daily", "ocean_surface"),
    SurfaceAndOceanVariable("Oday", "tossq", "oday_tossq", "daily", "ocean_surface"),
    SurfaceAndOceanVariable(
        "Oday", "omldamax", "oday_omldamax", "daily", "ocean_surface"
    ),
    SurfaceAndOceanVariable("Oday", "sos", "oday_sos", "daily", "ocean_surface"),
    # Monthly ocean diagnostics (Omon) — causal previous-month transform.
    # ``zos`` integrates full-column density; the others capture surface
    # forcing / mixed-layer / deep-ocean memory.
    SurfaceAndOceanVariable(
        "Omon", "zos", "omon_zos", "monthly_causal", "ocean_surface"
    ),
    SurfaceAndOceanVariable(
        "Omon", "hfds", "omon_hfds", "monthly_causal", "ocean_surface"
    ),
    SurfaceAndOceanVariable(
        "Omon", "mlotst", "omon_mlotst", "monthly_causal", "ocean_surface"
    ),
    SurfaceAndOceanVariable(
        "Omon", "tob", "omon_tob", "monthly_causal", "ocean_surface"
    ),
)


def _surface_and_ocean_by_output() -> dict[str, SurfaceAndOceanVariable]:
    return {h.output_name: h for h in SURFACE_AND_OCEAN_VARIABLES}


SURFACE_AND_OCEAN_BY_OUTPUT: dict[str, SurfaceAndOceanVariable] = (
    _surface_and_ocean_by_output()
)


# Output names of all surface-and-ocean variables — used as the default list
# of variables to attempt to ingest. Per-dataset, the pipeline opens only
# those whose source table is actually published.
SURFACE_AND_OCEAN_VARIABLE_NAMES: list[str] = [
    h.output_name for h in SURFACE_AND_OCEAN_VARIABLES
]


def make_label(source_id: str, physics_index: int) -> str:
    """Compose the composite embedding label from (source_id, p).

    The embedding treats each ``(source_id, p)`` pair as a distinct
    "model" — realization (r), initialization (i), and forcing (f) are
    within-label variation. The physics index is always included, even
    for source_ids that publish only one ``p`` value, so the label
    maps 1:1 to CMIP6 metadata.
    """
    return f"{source_id}.p{physics_index}"


# Variables regridded conservatively; everything else is bilinear by
# default. Fractions (siconc, sftlf, sftof) count as conservative too —
# we want the area-weighted mean across the coarser target cell, and
# bilinear of a {0, 1}-ish field tends to drift. Ocean-grid variables
# often fall back to bilinear in practice (see README's "Known
# regridding / data-pipeline limitations" — CESM2 SImon siconc rc=506),
# but the request encodes intent.
FLUX_LIKE_VARIABLES: frozenset[str] = frozenset(
    {
        # Radiative + turbulent fluxes
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
        # Surface ocean heat flux
        "hfds",
        # Fractions
        "siconc",
        "sftlf",
        "sftof",
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
    # Number of diffusion iterations for the horizontal ocean fill,
    # used to extrapolate ocean/sea-ice variables into land cells so the
    # stored data is NaN-free. The mask channel preserves the valid
    # extent; the fill keeps the boundary smooth for the network.
    ocean_fill_iterations: int = 50


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
    # Source-prefixed surface-and-ocean variables (surface T, sea-ice, ocean).
    # The pipeline opens only the subset whose source table is actually
    # published for each dataset. See ``SURFACE_AND_OCEAN_VARIABLES``.
    surface_and_ocean_variables: list[str] = field(
        default_factory=lambda: list(SURFACE_AND_OCEAN_VARIABLE_NAMES)
    )
    static_variables: list[str] = field(default_factory=lambda: list(STATIC_VARIABLES))
    # Maximum number of ``core_variables`` permitted to be missing from
    # the source ``day`` table without skipping the dataset. Missing
    # variables are simply absent from the output; training handles
    # missingness via ``allow_variable_masking`` on the training side
    # (per-sample masking). Defaults to 3 because dataset
    # generation is expensive and we don't want to re-run to relax this
    # later — coverage by threshold (37 eligible models at ``0``):
    # +10 at ``1``, +13 at ``2``, +15 at ``3``, +17 at ``4``. ``3``
    # captures everything except the E3SM family (which loses 5+ vars).
    # Set lower per-config to enforce stricter requirements.
    max_core_missing: int = 3
    time_subset: dict[str, TimeWindow] = field(default_factory=dict)
    target_grid: TargetGrid = field(default_factory=TargetGrid)
    regrid: RegridConfig = field(default_factory=RegridConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    # If False (default), duplicate timestamps in a source variable
    # cause the dataset to be skipped with a descriptive reason.
    # Enable per-dataset via an override after manually verifying that
    # the duplicates are a publishing artefact rather than a real
    # simulation boundary; the runtime still verifies data identity
    # before merging.
    allow_dedupe: bool = False


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
    allow_dedupe: Optional[bool] = None
    # Surface-and-ocean variables to skip for this dataset (e.g. ``simon_siconc``
    # for AWI-ESM-1-1-LR whose unstructured ocean grid OOMs xesmf).
    skip_surface_and_ocean_variables: Optional[list[str]] = None


@dataclass
class Selection:
    source_ids: Optional[list[str]] = None
    # Source IDs to exclude (applied after the ``source_ids`` filter).
    # Use to drop models whose published data has known quality issues
    # we don't want to ingest, e.g. INM-CM4-8 (anomalous ``zg`` at the
    # top of atmosphere — see README).
    exclude_source_ids: list[str] = field(default_factory=list)
    experiments: list[str] = field(
        default_factory=lambda: ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    )
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
    # index.csv (+ index.parquet when an engine is available) are written
    # at ``<output_directory>/index.*`` by process.py.
    output_directory: str
    # Optional shared location for the per-scenario external-forcings
    # zarrs (CO2, SO2, BC, LUH2 forest). When unset, defaults to
    # ``<output_directory>/external_forcings/``. Pointing several
    # output-directory versions at the same ``external_forcings_directory``
    # lets them reuse one staged copy of the forcings — the inputs are
    # version-independent until ``external_forcings.py`` changes, so
    # re-staging per version is wasteful.
    external_forcings_directory: Optional[str] = None
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
        return _resolve(
            self.defaults, self.overrides, source_id, experiment, variant_label
        )

    @property
    def resolved_external_forcings_directory(self) -> str:
        """``external_forcings_directory`` if set, otherwise the legacy
        ``<output_directory>/external_forcings`` default. Always ends
        without a trailing slash.
        """
        return (
            self.external_forcings_directory
            or f"{self.output_directory.rstrip('/')}/external_forcings"
        ).rstrip("/")


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
    surface_and_ocean_variables: list[str]
    static_variables: list[str]
    max_core_missing: int
    time_subset: dict[str, TimeWindow]
    target_grid: TargetGrid
    regrid: RegridConfig
    fill: FillConfig
    chunking: ChunkingConfig
    allow_dedupe: bool = False


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
    # Group SURFACE_AND_OCEAN_VARIABLES by source table_id so we issue one catalog
    # query per table. Preserve the order in SURFACE_AND_OCEAN_VARIABLES so the
    # inventory CSV is deterministic.
    surface_and_ocean_by_table: dict[str, list[str]] = {}
    for h in SURFACE_AND_OCEAN_VARIABLES:
        surface_and_ocean_by_table.setdefault(h.table_id, []).append(h.var_id)
    # Day-cadence variables that don't live on the standard ``day``
    # table — currently TOA shortwave (``rsdt``, ``rsut``) on
    # ``CFday``. The pipeline folds these into the day-cadence
    # variable set; the inventory just needs them in its catalog.
    return [
        CatalogQuery(
            table_id="day",
            variables=(
                list(CORE_VARIABLES)
                + [v for v in OPTIONAL_VARIABLES if v not in CFDAY_VARIABLES]
                + list(DIAGNOSTIC_VARIABLES)
            ),
        ),
        CatalogQuery(table_id="CFday", variables=list(CFDAY_VARIABLES)),
        *[
            CatalogQuery(table_id=tab, variables=variables)
            for tab, variables in surface_and_ocean_by_table.items()
        ],
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
    experiments: list[str] = field(
        default_factory=lambda: ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    )

    @classmethod
    def from_file(cls, path: str) -> "InventoryConfig":
        return _load_yaml(cls, path)


ESGF_DEFAULT_NODE = "https://esgf-data.dkrz.de/esg-search/search"


@dataclass
class ESGFConfig:
    """ESGF-specific settings for the ESGF processing pipeline."""

    search_node: str = ESGF_DEFAULT_NODE
    scratch_dir: str = "./scratch"


@dataclass
class ESGFProcessConfig:
    """Top-level config for the ESGF processing pipeline.

    Shares most fields with ``ProcessConfig`` but adds ESGF-specific
    settings and does not require ``inventory_path`` (the ESGF pipeline
    discovers datasets via the ESGF search API at runtime).
    """

    output_directory: str
    external_forcings_directory: Optional[str] = None  # see ProcessConfig.
    esgf: ESGFConfig = field(default_factory=ESGFConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    selection: Selection = field(default_factory=Selection)
    overrides: list[Override] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "ESGFProcessConfig":
        return _load_yaml(cls, path)

    def resolve(
        self, source_id: str, experiment: str, variant_label: str
    ) -> "ResolvedDatasetConfig":
        """Apply overrides on top of defaults for one dataset."""
        return _resolve(
            self.defaults, self.overrides, source_id, experiment, variant_label
        )

    @property
    def resolved_external_forcings_directory(self) -> str:
        return (
            self.external_forcings_directory
            or f"{self.output_directory.rstrip('/')}/external_forcings"
        ).rstrip("/")


def _resolve(
    defaults: DefaultsConfig,
    overrides: list[Override],
    source_id: str,
    experiment: str,
    variant_label: str,
) -> "ResolvedDatasetConfig":
    """Apply overrides on top of defaults for one dataset. Shared between
    ``ProcessConfig`` and ``ESGFProcessConfig``.
    """
    time_subset = dict(defaults.time_subset)
    allow_dedupe = defaults.allow_dedupe
    surface_and_ocean_variables = list(defaults.surface_and_ocean_variables)
    for override in overrides:
        if override.match.matches(source_id, experiment, variant_label):
            if override.time_subset is not None:
                time_subset = dict(override.time_subset)
            if override.allow_dedupe is not None:
                allow_dedupe = override.allow_dedupe
            if override.skip_surface_and_ocean_variables is not None:
                surface_and_ocean_variables = [
                    v
                    for v in surface_and_ocean_variables
                    if v not in override.skip_surface_and_ocean_variables
                ]
    return ResolvedDatasetConfig(
        source_id=source_id,
        experiment=experiment,
        variant_label=variant_label,
        core_variables=list(defaults.core_variables),
        optional_variables=list(defaults.optional_variables),
        surface_and_ocean_variables=surface_and_ocean_variables,
        static_variables=list(defaults.static_variables),
        max_core_missing=defaults.max_core_missing,
        time_subset=time_subset,
        target_grid=defaults.target_grid,
        regrid=defaults.regrid,
        fill=defaults.fill,
        chunking=defaults.chunking,
        allow_dedupe=allow_dedupe,
    )


@dataclass
class ESGFInventoryConfig:
    """Config for the ESGF inventory discovery script."""

    output_path: str
    search_node: str = ESGF_DEFAULT_NODE
    queries: list[CatalogQuery] = field(default_factory=_default_inventory_queries)
    experiments: list[str] = field(
        default_factory=lambda: ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    )

    @classmethod
    def from_file(cls, path: str) -> "ESGFInventoryConfig":
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
    "STATIC_VARIABLES",
    "FLUX_LIKE_VARIABLES",
    "SURFACE_AND_OCEAN_VARIABLES",
    "SURFACE_AND_OCEAN_VARIABLE_NAMES",
    "SURFACE_AND_OCEAN_BY_OUTPUT",
    "SurfaceAndOceanVariable",
    "ESGF_DEFAULT_NODE",
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
    "ESGFConfig",
    "ESGFProcessConfig",
    "ESGFInventoryConfig",
]
