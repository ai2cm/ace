"""Named post-regrid transforms, selected per stream in the YAML config.

Each transform operates on one regridded chunk (output variable names, after
renaming) and may adjust variables in place or add derived ones. Transforms
are registered in POSTPROCESS with the output variables they require and the
ones they add, so the driver can validate config selections and predict the
output variable set. A transform is skipped for chunks that don't carry its
required variables (e.g. the 3D branch of a stream whose transform needs
surface fields).
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


def _append_derivation(da: xr.DataArray, note: str) -> xr.DataArray:
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
        areacello: exact target-grid cell areas in m^2.
        store: source store URL of the stream, for provenance attrs.
    """

    ocean_fraction: xr.DataArray
    areacello: xr.DataArray
    store: str


def kelvin_sst(ds: xr.Dataset, context: ChunkContext) -> xr.Dataset:
    """Add ``sst``: sea surface temperature in Kelvin, from ``tos``."""
    sst = ds["tos"] + 273.15
    sst.attrs = {
        "long_name": "Sea surface temperature",
        "units": "K",
        **provenance_attrs(context.store, "tos", "tos + 273.15"),
    }
    ds["sst"] = sst
    return ds


def hfds_total_area(ds: xr.Dataset, context: ChunkContext) -> xr.Dataset:
    """Add ``hfds_total_area``: heat flux into sea water per full cell area.

    The wetmask-normalized ``hfds`` is an average over ocean source area
    only; multiplying by the same ocean fraction that normalized it recovers
    the plain conservative regrid — the flux per total cell area.
    """
    out = ds["hfds"] * context.ocean_fraction
    out.attrs = {
        "long_name": "heat flux into sea water scaled by sea surface fraction",
        "units": ds["hfds"].attrs.get("units", "W/m2"),
        **provenance_attrs(
            context.store,
            "hfds",
            "hfds (an average over ocean source area) multiplied by the "
            "cell's ocean fraction, giving the flux per total cell area",
        ),
    }
    ds["hfds_total_area"] = out
    return ds


# The full-cell sea-ice fraction and the ocean-relative one times the ocean
# fraction are the same quantity computed along two paths that should agree
# to float roundoff; a larger disagreement means the two variables were not
# regridded from the same source field.
MAX_SEA_ICE_FRACTION_MISMATCH = 1e-5


def sea_ice(ds: xr.Dataset, context: ChunkContext) -> xr.Dataset:
    """Sea-ice conventions applied after regridding:

    - ``UI``/``VI`` and ``HI`` are zero where ``sea_ice_fraction`` is zero
      (NaN over land), so the fields are defined everywhere over ocean with
      no time-varying NaN pattern.
    - ``sea_ice_volume`` = ``HI`` x ``areacello`` x ``sea_ice_fraction``,
      in m^3.
    - Consistency check: ``sea_ice_fraction`` (ice area per total cell area)
      must equal ``ocean_sea_ice_fraction`` (ice area per ocean area) times
      the cell's ocean fraction.
    """
    frac = ds["sea_ice_fraction"]

    reconstructed = ds["ocean_sea_ice_fraction"] * context.ocean_fraction
    mismatch = float(np.nanmax(np.abs((reconstructed - frac).values)))
    if mismatch > MAX_SEA_ICE_FRACTION_MISMATCH:
        raise AssertionError(
            "sea_ice_fraction disagrees with ocean_sea_ice_fraction x "
            f"ocean_fraction by up to {mismatch:g} "
            f"(limit {MAX_SEA_ICE_FRACTION_MISMATCH:g})"
        )

    zero_note = "zero where sea_ice_fraction is zero, NaN over land"
    for name in ("UI", "VI", "HI"):
        zeroed = ds[name].where(frac > 0, 0.0).where(frac.notnull())
        zeroed.attrs = ds[name].attrs
        ds[name] = _append_derivation(zeroed, zero_note)

    volume = ds["HI"] * context.areacello * frac
    volume.attrs = {
        "long_name": "ice volume",
        "units": "m^3",
        **provenance_attrs(
            context.store,
            "HI",
            "HI x areacello x sea_ice_fraction (ice thickness times the "
            "ice-covered cell area)",
        ),
    }
    ds["sea_ice_volume"] = volume
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
    "kelvin_sst": Postprocess(kelvin_sst, requires=("tos",), adds=("sst",)),
    "hfds_total_area": Postprocess(
        hfds_total_area, requires=("hfds",), adds=("hfds_total_area",)
    ),
    "sea_ice": Postprocess(
        sea_ice,
        requires=(
            "sea_ice_fraction",
            "ocean_sea_ice_fraction",
            "HI",
            "UI",
            "VI",
        ),
        adds=("sea_ice_volume",),
    ),
}
