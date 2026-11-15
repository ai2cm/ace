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
    sea_ice_fraction_consistency_postprocess,
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


# J/kg, SIS2 LATENT_HEAT_VAPOR and LATENT_HEAT_FUSION
# (docs/cm4-params/SIS_parameter_doc.all in the ai2cm workspace).
LATENT_HEAT_VAPOR = 2.5e6
LATENT_HEAT_FUSION = 3.34e5
CALVING_RESIDUE = "calving_residue"


def calving_residue(
    ds: xr.Dataset, context: ChunkContext, *, hflso: str, evs: str, prsn: str
) -> xr.Dataset:
    """Add ``calving_residue`` = -hflso + L_v evs - L_f prsn, per ocean area.

    MOM6's ``hflso`` is L_v evap - L_f fprec - L_f frunoff, with ``evs`` =
    evap and ``prsn`` = fprec (mass into the ocean positive), so the residue
    is the latent heat of calving melt, L_f frunoff.
    """
    out = -ds[hflso] + LATENT_HEAT_VAPOR * ds[evs] - LATENT_HEAT_FUSION * ds[prsn]
    formula = f"-{hflso} + L_v {evs} - L_f {prsn}"
    source = ds[hflso].attrs
    out.attrs = {
        "long_name": "Surface Downward Latent Heat Flux residue due to calving",
        "units": source.get("units", "W/m2"),
        **{k: source[k] for k in ("cell_methods", "cell_measures") if k in source},
        **provenance_attrs(
            context.store,
            ", ".join((hflso, evs, prsn)),
            f"{formula}, L_v = {LATENT_HEAT_VAPOR:g} J/kg, "
            f"L_f = {LATENT_HEAT_FUSION:g} J/kg (SIS2), on the per-ocean-area "
            "regridded inputs",
        ),
    }
    ds[CALVING_RESIDUE] = out
    return ds


def calving_residue_postprocess(
    *, hflso: str = "hflso", evs: str = "evs", prsn: str = "prsn"
) -> Postprocess:
    return Postprocess(
        functools.partial(calving_residue, hflso=hflso, evs=evs, prsn=prsn),
        requires=(hflso, evs, prsn),
        adds=(CALVING_RESIDUE,),
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
    - ``sea_ice_volume`` = ``areacello`` x ``(HI c)_J``, in m^3: ``A_J``
      times the full-cell regrid of native ``HI`` x ``sea_ice_fraction``
      (ChunkContext.ice_volume_per_cell_area), so its sum over cells equals
      the native ice volume.
    """
    frac = ds[sea_ice_fraction]
    if context.areacello is None or context.ice_volume_per_cell_area is None:
        raise ValueError(
            "sea_ice needs ChunkContext.areacello and .ice_volume_per_cell_area"
        )

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

    volume = context.ice_volume_per_cell_area * context.areacello
    volume.attrs = {
        "long_name": "ice volume",
        "units": "m^3",
        **provenance_attrs(
            context.store,
            "HI",
            f"areacello x full-cell regrid of native HI x {sea_ice_fraction} "
            "(ice volume per cell area, zeroed over land)",
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
    "calving_residue": calving_residue_postprocess,
    "sea_ice": sea_ice_postprocess,
    "sea_ice_fraction_consistency": sea_ice_fraction_consistency_postprocess,
}
