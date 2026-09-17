"""Named post-regrid transforms, selected in the YAML config.

Each transform operates on one regridded chunk (output variable names, after
renaming) and may adjust variables in place, add derived ones, or assert a
consistency relation. Transforms are registered in POSTPROCESS with the
output variables they require and the ones they add, so the driver can
validate config selections and predict the output variable set. A transform
is skipped for chunks that don't carry its required variables.
"""

import dataclasses
from typing import Callable

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


@dataclasses.dataclass
class ChunkContext:
    """Per-chunk quantities available to postprocess transforms.

    Attributes:
        ocean_fraction: the regridded ocean fraction that normalized this
            chunk's regrid (the chunk's surface ocean coverage on the target
            grid).
        store: source store URL of the stream, for provenance attrs.
    """

    ocean_fraction: xr.DataArray
    store: str


def kelvin_sst(ds: xr.Dataset, context: ChunkContext) -> xr.Dataset:
    """Add ``sst``: sea surface temperature in Kelvin, from ``SST``."""
    sst = ds["SST"] + 273.15
    sst.attrs = {
        "long_name": "Sea surface temperature",
        "units": "K",
        **provenance_attrs(context.store, "SST", "SST + 273.15"),
    }
    ds["sst"] = sst
    return ds


# The full-cell sea-ice fraction and the ocean-relative one times the ocean
# fraction are the same quantity computed along two paths that should agree
# to float roundoff; a larger disagreement means the two variables were not
# regridded from the same source field.
MAX_SEA_ICE_FRACTION_MISMATCH = 1e-5


def sea_ice_fraction_consistency(ds: xr.Dataset, context: ChunkContext) -> xr.Dataset:
    """Assert ``sea_ice_fraction`` (ice area per total cell area) equals
    ``ocean_sea_ice_fraction`` (ice area per ocean area) times the cell's
    ocean fraction.

    The two come from one source field down two regridding paths — the
    full-cell path and the wetmask-normalized one — so they are redundant by
    construction, and a disagreement means they were not built from the same
    field. Adds nothing to the chunk.
    """
    frac = ds["sea_ice_fraction"]
    reconstructed = ds["ocean_sea_ice_fraction"] * context.ocean_fraction
    difference = np.abs((reconstructed - frac).values)
    if np.isnan(difference).all():
        raise AssertionError(
            "sea_ice_fraction and ocean_sea_ice_fraction have no cell where "
            "both are defined; the chunk carries no ocean"
        )
    mismatch = float(np.nanmax(difference))
    if mismatch > MAX_SEA_ICE_FRACTION_MISMATCH:
        raise AssertionError(
            "sea_ice_fraction disagrees with ocean_sea_ice_fraction x "
            f"ocean_fraction by up to {mismatch:g} "
            f"(limit {MAX_SEA_ICE_FRACTION_MISMATCH:g})"
        )
    return ds


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


POSTPROCESS: dict[str, Postprocess] = {
    "kelvin_sst": Postprocess(kelvin_sst, requires=("SST",), adds=("sst",)),
    "sea_ice_fraction_consistency": Postprocess(
        sea_ice_fraction_consistency,
        requires=("sea_ice_fraction", "ocean_sea_ice_fraction"),
        adds=(),
    ),
}
