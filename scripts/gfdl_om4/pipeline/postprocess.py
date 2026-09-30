"""Post-regrid transform contract, provenance helpers, and the transforms
more than one pipeline uses (``kelvin_sst``, the sea-ice fraction check).

A pipeline's named post-regrid transforms are registered in its own
subpackage (e.g. pipeline/om4/postprocess.py), in that module's
``POSTPROCESS`` dict: name -> factory. A factory's keyword arguments are the
variable names the transform reads, and it returns a :class:`Postprocess`.
A config selects a transform by name (factory defaults) or as
``{name: ..., sources: {<argument>: <variable>}}`` (see
:class:`PostprocessConfig`). Each transform operates on one regridded chunk
(output variable names, after renaming) and may adjust variables in place,
add derived ones, or assert a consistency relation. A :class:`Postprocess`
records the output variables the transform requires and the ones it adds, so
the driver can validate config selections and predict the output variable
set. A transform is skipped for chunks that don't carry its required
variables.
"""

import dataclasses
import functools
import inspect
from typing import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

# Provenance attribute names stamped on every output variable.
SOURCE_STORE_ATTR = "source_store"
SOURCE_VARIABLE_ATTR = "source_variable"
DERIVATION_ATTR = "derivation"


def provenance_attrs(store: str, variable: str, derivation: str | None = None) -> dict:
    attrs = {SOURCE_STORE_ATTR: store, SOURCE_VARIABLE_ATTR: variable}
    if derivation is not None:
        attrs[DERIVATION_ATTR] = derivation
    return attrs


def append_derivation(da: xr.DataArray, note: str) -> xr.DataArray:
    """Append ``note`` to the variable's derivation attribute, in place."""
    existing = da.attrs.get(DERIVATION_ATTR)
    da.attrs[DERIVATION_ATTR] = f"{existing}; {note}" if existing else note
    return da


@dataclasses.dataclass
class ChunkContext:
    """Per-chunk quantities available to postprocess transforms.

    Attributes:
        ocean_fraction: the regridded ocean fraction that normalized this
            chunk's regrid (the chunk's instantaneous surface ocean coverage
            on the target grid).
        store: source store URL of the stream, for provenance attrs.
        areacello: exact target-grid cell areas in m^2, for pipelines whose
            transforms need them; None otherwise.
    """

    ocean_fraction: xr.DataArray
    store: str
    areacello: xr.DataArray | None = None


@dataclasses.dataclass(frozen=True)
class Postprocess:
    """A registered transform with its variable contract.

    Attributes:
        fn: the transform, applied to a regridded chunk.
        requires: output variables that must be present in the chunk for the
            transform to apply; chunks lacking any of them are passed through
            unchanged.
        adds: output variables the transform adds.
    """

    fn: Callable[[xr.Dataset, ChunkContext], xr.Dataset]
    requires: tuple[str, ...]
    adds: tuple[str, ...]


PostprocessFactory = Callable[..., Postprocess]


@dataclasses.dataclass
class PostprocessConfig:
    """A transform selected in a stream's ``postprocess`` list.

    Attributes:
        name: key of the pipeline's POSTPROCESS registry.
        sources: keyword arguments of the registered factory, each naming an
            output variable the transform reads (e.g. ``celsius_sst: tos``).
            Arguments left out take the factory's default.
    """

    name: str
    sources: dict[str, str] = dataclasses.field(default_factory=dict)


def resolve_postprocess(
    registry: Mapping[str, PostprocessFactory],
    entries: Sequence["str | PostprocessConfig"],
    context: str,
) -> list[Postprocess]:
    """Build the configured transforms of one stream, in order.

    Raises:
        ValueError: an entry names an unknown transform, an unknown source
            argument, or leaves out a source argument without a default.
    """
    specs = []
    for entry in entries:
        selection = PostprocessConfig(entry) if isinstance(entry, str) else entry
        if selection.name not in registry:
            raise ValueError(
                f"unknown postprocess {selection.name!r} in {context}; "
                f"available: {sorted(registry)}"
            )
        factory = registry[selection.name]
        try:
            inspect.signature(factory).bind(**selection.sources)
        except TypeError as err:
            raise ValueError(
                f"postprocess {selection.name!r} in {context}: sources "
                f"{selection.sources} do not match "
                f"{inspect.signature(factory)}: {err}"
            ) from err
        specs.append(factory(**selection.sources))
    return specs


def assert_postprocess_inputs(
    specs: Sequence[Postprocess],
    produced: set[str],
    context: str,
    allow_level_suffix: bool = False,
) -> None:
    """Assert every transform's required variables are among ``produced``
    (the stream's output names) or added by an earlier transform.

    With ``allow_level_suffix``, ``<name>_<k>`` also counts as produced when
    ``<name>`` is: before the source is opened a 3D variable's per-level
    outputs are known only by their base name.
    """
    available = set(produced)
    for spec in specs:
        for name in spec.requires:
            base, _, suffix = name.rpartition("_")
            level_split = allow_level_suffix and suffix.isdigit() and base in available
            if name not in available and not level_split:
                raise ValueError(
                    f"postprocess requires {name!r}, which {context} does not "
                    f"produce; produced: {sorted(available)}"
                )
        available.update(spec.adds)


def kelvin_sst(
    ds: xr.Dataset, context: ChunkContext, *, celsius_sst: str
) -> xr.Dataset:
    """Add ``sst``: sea surface temperature in Kelvin, from ``celsius_sst``."""
    sst = ds[celsius_sst] + 273.15
    sst.attrs = {
        "long_name": "Sea surface temperature",
        "units": "K",
        **provenance_attrs(context.store, celsius_sst, f"{celsius_sst} + 273.15"),
    }
    ds["sst"] = sst
    return ds


def kelvin_sst_postprocess(*, celsius_sst: str) -> Postprocess:
    """``kelvin_sst`` reading the Celsius SST from output variable
    ``celsius_sst``. No default: the name differs between pipelines."""
    return Postprocess(
        functools.partial(kelvin_sst, celsius_sst=celsius_sst),
        requires=(celsius_sst,),
        adds=("sst",),
    )


# The full-cell sea-ice fraction and the ocean-relative one times the ocean
# fraction are the same quantity computed along two paths that should agree
# to float roundoff; a larger disagreement means the two variables were not
# regridded from the same source field.
MAX_SEA_ICE_FRACTION_MISMATCH = 1e-5

SEA_ICE_FRACTION = "sea_ice_fraction"
OCEAN_SEA_ICE_FRACTION = "ocean_sea_ice_fraction"


def check_sea_ice_fraction_consistency(
    ds: xr.Dataset,
    context: ChunkContext,
    *,
    sea_ice_fraction: str,
    ocean_sea_ice_fraction: str,
) -> None:
    """Assert ``sea_ice_fraction`` (ice area per total cell area) equals
    ``ocean_sea_ice_fraction`` (ice area per ocean area) times the cell's
    ocean fraction.

    The two come from one source field down two regridding paths — the
    full-cell path and the wetmask-normalized one — so they are redundant by
    construction, and a disagreement means they were not built from the same
    field.
    """
    frac = ds[sea_ice_fraction]
    reconstructed = ds[ocean_sea_ice_fraction] * context.ocean_fraction
    difference = np.abs((reconstructed - frac).values)
    if np.isnan(difference).all():
        raise AssertionError(
            f"{sea_ice_fraction} and {ocean_sea_ice_fraction} have no cell "
            "where both are defined; the chunk carries no ocean"
        )
    mismatch = float(np.nanmax(difference))
    if mismatch > MAX_SEA_ICE_FRACTION_MISMATCH:
        raise AssertionError(
            f"{sea_ice_fraction} disagrees with {ocean_sea_ice_fraction} x "
            f"ocean_fraction by up to {mismatch:g} "
            f"(limit {MAX_SEA_ICE_FRACTION_MISMATCH:g})"
        )


def sea_ice_fraction_consistency(
    ds: xr.Dataset,
    context: ChunkContext,
    *,
    sea_ice_fraction: str,
    ocean_sea_ice_fraction: str,
) -> xr.Dataset:
    """The fraction check alone, as a transform; adds nothing to the chunk."""
    check_sea_ice_fraction_consistency(
        ds,
        context,
        sea_ice_fraction=sea_ice_fraction,
        ocean_sea_ice_fraction=ocean_sea_ice_fraction,
    )
    return ds


def sea_ice_fraction_consistency_postprocess(
    *,
    sea_ice_fraction: str = SEA_ICE_FRACTION,
    ocean_sea_ice_fraction: str = OCEAN_SEA_ICE_FRACTION,
) -> Postprocess:
    return Postprocess(
        functools.partial(
            sea_ice_fraction_consistency,
            sea_ice_fraction=sea_ice_fraction,
            ocean_sea_ice_fraction=ocean_sea_ice_fraction,
        ),
        requires=(sea_ice_fraction, ocean_sea_ice_fraction),
        adds=(),
    )
