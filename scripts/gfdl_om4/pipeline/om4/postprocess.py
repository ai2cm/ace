"""OM4 named post-regrid transforms, selected per stream in the YAML config.

The transform contract (:class:`~pipeline.postprocess.Postprocess`,
:class:`~pipeline.postprocess.ChunkContext`), the provenance helpers and the
transforms shared with other pipelines (``kelvin_sst``, the sea-ice fraction
check) are in pipeline/postprocess.py. This module registers the OM4
transform factories in POSTPROCESS.
"""

import functools

import xarray as xr

from ..postprocess import (
    OCEAN_SEA_ICE_FRACTION,
    SEA_ICE_FRACTION,
    ChunkContext,
    Postprocess,
    PostprocessFactory,
    append_derivation,
    check_sea_ice_fraction_consistency,
    kelvin_sst_postprocess,
    provenance_attrs,
)


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


def hfds_total_area_postprocess() -> Postprocess:
    return Postprocess(hfds_total_area, requires=("hfds",), adds=("hfds_total_area",))


def sea_ice(
    ds: xr.Dataset,
    context: ChunkContext,
    *,
    sea_ice_fraction: str,
    ocean_sea_ice_fraction: str,
) -> xr.Dataset:
    """Sea-ice conventions applied after regridding:

    - Consistency check: ``sea_ice_fraction`` (ice area per total cell area)
      must equal ``ocean_sea_ice_fraction`` (ice area per ocean area) times
      the cell's ocean fraction
      (pipeline.postprocess.check_sea_ice_fraction_consistency).
    - ``UI``/``VI`` and ``HI`` are zero where ``sea_ice_fraction`` is zero
      (NaN over land), so the fields are defined everywhere over ocean with
      no time-varying NaN pattern.
    - ``sea_ice_volume`` = ``HI`` x ``areacello`` x ``sea_ice_fraction``,
      in m^3.
    """
    frac = ds[sea_ice_fraction]
    if context.areacello is None:
        raise ValueError("sea_ice needs ChunkContext.areacello")

    check_sea_ice_fraction_consistency(
        ds,
        context,
        sea_ice_fraction=sea_ice_fraction,
        ocean_sea_ice_fraction=ocean_sea_ice_fraction,
    )

    zero_note = f"zero where {sea_ice_fraction} is zero, NaN over land"
    for name in ("UI", "VI", "HI"):
        zeroed = ds[name].where(frac > 0, 0.0).where(frac.notnull())
        zeroed.attrs = ds[name].attrs
        ds[name] = append_derivation(zeroed, zero_note)

    volume = ds["HI"] * context.areacello * frac
    volume.attrs = {
        "long_name": "ice volume",
        "units": "m^3",
        **provenance_attrs(
            context.store,
            "HI",
            f"HI x areacello x {sea_ice_fraction} (ice thickness times the "
            "ice-covered cell area)",
        ),
    }
    ds["sea_ice_volume"] = volume
    return ds


def sea_ice_postprocess(
    *,
    sea_ice_fraction: str = SEA_ICE_FRACTION,
    ocean_sea_ice_fraction: str = OCEAN_SEA_ICE_FRACTION,
) -> Postprocess:
    return Postprocess(
        functools.partial(
            sea_ice,
            sea_ice_fraction=sea_ice_fraction,
            ocean_sea_ice_fraction=ocean_sea_ice_fraction,
        ),
        requires=(sea_ice_fraction, ocean_sea_ice_fraction, "HI", "UI", "VI"),
        adds=("sea_ice_volume",),
    )


POSTPROCESS: dict[str, PostprocessFactory] = {
    "kelvin_sst": kelvin_sst_postprocess,
    "hfds_total_area": hfds_total_area_postprocess,
    "sea_ice": sea_ice_postprocess,
}
