"""Does the prescribed surface-flux correction reproduce the reference hfds?

`SurfaceEnergyFluxCorrectionConfig(method="prescribed")` replaces the emulator's
surface heat flux over open ocean with `net_flux`, computed by
`_compute_ocean_net_surface_energy_flux` from atmospheric forcing and SST. The
function's own comments record two omitted terms (calving latent heat, river
runoff), so the imposed flux may carry a bias that a perfect emulator would
inherit.

This evaluates the formula on *reference* forcing and *reference* SST and
compares it with the reference hfds. For an emulator whose prediction equalled
the reference everywhere, the corrected output would be

    net_flux * ssf * ocean_fraction + ref_hfds_total_area * (1 - ocean_fraction)

so its error against the reference is

    ocean_fraction * (net_flux * ssf - ref_hfds_total_area)

whose area-weighted global mean is the bias the correction formula injects on its
own. That number is directly comparable to the ~0.1 W/m2 surface-flux excess seen
in the 200-year piControl rollouts.

Run under `gantry` with the weka climate-default mount; see the record in the
research repo for the launch line.
"""

import argparse

import numpy as np
import torch
import xarray as xr

from fme.core.corrector.ocean import _compute_ocean_net_surface_energy_flux
from fme.core.ocean_data import OceanData

FORCING = [
    "DLWRFsfc", "DSWRFsfc", "ULWRFsfc", "USWRFsfc", "LHTFLsfc", "SHTFLsfc",
    "PRATEsfc", "total_frozen_precipitation_rate",
]
STATE = ["sst", "land_fraction", "ocean_sea_ice_fraction", "sea_surface_fraction",
         "hfds_total_area", "hfgeou"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr", default="/climate-default/2026-07-15-om4-picontrol-1deg-coupled-ocean.zarr")
    ap.add_argument("--start", default="0156-01-01")
    ap.add_argument("--stop", default="0176-01-01")
    args = ap.parse_args()

    ds = xr.open_zarr(args.zarr).sel(time=slice(args.start, args.stop))
    print(f"{args.zarr}\n  {ds.sizes}\n  {args.start} .. {args.stop}", flush=True)
    missing = [v for v in FORCING + STATE if v not in ds]
    if missing:
        raise SystemExit(f"missing variables: {missing}")

    ydim, xdim = [d for d in ds["hfds_total_area"].dims if d != "time"]
    lat = ds[ydim].values
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), ds.sizes[xdim]))
    weights = torch.tensor(w / w.sum(), dtype=torch.float64)   # sums to 1 over the map

    sums = {k: 0.0 for k in ("net_flux_ssf", "ref_hfds", "formula_bias", "of", "geo")}
    n = 0
    CHUNK = 200
    ntime = ds.sizes["time"]
    for i0 in range(0, ntime, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, ntime))
        block = ds.isel(time=sl).load()
        data = {v: torch.tensor(block[v].values, dtype=torch.float64) for v in FORCING + STATE}
        ocean = OceanData(data)
        net_flux = _compute_ocean_net_surface_energy_flux(data, ocean.sea_surface_temperature)
        ssf = data["sea_surface_fraction"]
        of = ocean.ocean_fraction
        imposed = net_flux * ssf                      # per total area, as the corrector uses it
        ref = data["hfds_total_area"]
        bias = of * (imposed - ref)                   # error a perfect emulator inherits
        for key, field in (("net_flux_ssf", imposed), ("ref_hfds", ref),
                           ("formula_bias", bias), ("of", of),
                           ("geo", data["hfgeou"])):
            sums[key] += float((field * weights).sum())
        n += block.sizes["time"]
        print(f"  {i0 + block.sizes['time']}/{ntime}", flush=True)

    print("\nArea-weighted global means over the period (W/m^2 unless noted)")
    print(f"  reference hfds_total_area                       {sums['ref_hfds'] / n:+.4f}")
    print(f"  formula net_flux * sea_surface_fraction         {sums['net_flux_ssf'] / n:+.4f}")
    print(f"  mean open-ocean fraction                        {sums['of'] / n:+.4f}  (fraction)")
    print(f"  geothermal flux hfgeou                          {sums['geo'] / n:+.4f}")
    print()
    print(f"  BIAS a perfect emulator inherits from the formula: {sums['formula_bias'] / n:+.4f} W/m^2")
    print(f"  (for comparison, the rollouts' surface-flux excess is ~+0.10 W/m^2)")


if __name__ == "__main__":
    main()
