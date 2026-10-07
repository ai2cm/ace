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


# J/kg, the e_cand3 sea-ice budget's constants (L_f is SIS2's default).
LATENT_HEAT_VAPORIZATION = 2.5e6
LATENT_HEAT_FUSION = 3.34e5


def combination_total_area(
    ds: xr.Dataset,
    context: ChunkContext,
    *,
    name: str,
    coefficients: dict[str, float],
    long_name: str,
    units: str,
) -> xr.Dataset:
    """Add ``name`` = ocean_fraction x sum_v coefficients[v] x v, per total
    cell area, and drop its wetmask-normalized components."""
    combination = sum(c * ds[v] for v, c in coefficients.items())
    out = combination * context.ocean_fraction
    formula = " + ".join(f"({c:g}) x {v}" for v, c in coefficients.items())
    out.attrs = {
        "long_name": long_name,
        "units": units,
        **provenance_attrs(
            context.store,
            ", ".join(coefficients),
            f"[{formula}] (averages over ocean source area) multiplied by "
            "the cell's ocean fraction, giving the value per total cell area",
        ),
    }
    ds[name] = out
    return ds.drop_vars(list(coefficients))


def _combination_postprocess(
    name: str, coefficients: dict[str, float], long_name: str, units: str
) -> Postprocess:
    return Postprocess(
        functools.partial(
            combination_total_area,
            name=name,
            coefficients=coefficients,
            long_name=long_name,
            units=units,
        ),
        requires=tuple(coefficients),
        adds=(name,),
        removes=tuple(coefficients),
    )


def calving_residue_total_area_postprocess() -> Postprocess:
    return _combination_postprocess(
        "calving_residue_total_area",
        {
            "hflso": -1.0,
            "evs": LATENT_HEAT_VAPORIZATION,
            "prsn": -LATENT_HEAT_FUSION,
        },
        "-hflso + L_v evs - L_f prsn scaled by sea surface fraction",
        "W/m2",
    )


def frozen_mass_total_area_postprocess() -> Postprocess:
    return _combination_postprocess(
        "frozen_mass_total_area",
        {"simass": 1.0, "sisnmass": 1.0},
        "sea ice plus snow mass scaled by sea surface fraction",
        "kg/m2",
    )


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
    "calving_residue_total_area": calving_residue_total_area_postprocess,
    "frozen_mass_total_area": frozen_mass_total_area_postprocess,
    "sea_ice": sea_ice_postprocess,
}
