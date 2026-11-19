"""Per-hemisphere latitude limits of the coupled-dataset sea-ice mask.

    sif_max_J = max over t of sif_J(t)            J: grid cell, t: full record
    zonal_max_k = max over lon of sif_max at row k   k: latitude row
    edge_h = row nearest the equator in hemisphere h with zonal_max_k > 0
    L_h = |lat| of the row next equatorward of edge_h (first ice-free row)
    mask: NaN on ocean where -L_south < lat < L_north (rows at L_h unmasked)

Usage:
    python compute_sea_ice_mask_limits.py \
        gs://vcm-ml-intermediate/2026-11-13-om4-picontrol-1deg-ocean-5daily.zarr \
        sea_ice_fraction
"""

import argparse

import numpy as np
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar

HEMISPHERES = ("north", "south")


def time_max(
    path: str, variable: str, time_chunk: int, num_workers: int
) -> xr.DataArray:
    da = xr.open_zarr(path, chunks={"time": time_chunk})[variable]
    with ProgressBar():
        return da.max("time").compute(num_workers=num_workers)


def zonal_max(sif_max: xr.DataArray) -> xr.DataArray:
    return sif_max.max("lon").fillna(0.0)  # all-land rows have no sea ice


def hemisphere_limit(zonal: xr.DataArray, hemisphere: str) -> dict:
    abs_lat = np.abs(zonal["lat"].values)
    in_hemisphere = zonal["lat"].values > 0
    if hemisphere == "south":
        in_hemisphere = ~in_hemisphere
    iced = in_hemisphere & (zonal.values > 0)
    if not iced.any():
        raise ValueError(f"No positive sea-ice fraction in the {hemisphere}.")
    i_edge = np.flatnonzero(iced)[np.argmin(abs_lat[iced])]
    equatorward = in_hemisphere & (abs_lat < abs_lat[i_edge])
    if not equatorward.any():
        raise ValueError(f"Sea ice reaches the {hemisphere} row nearest the equator.")
    return {
        "latitude_limit_deg": float(abs_lat[equatorward].max()),
        "edge_lat": float(zonal["lat"].values[i_edge]),
        "edge_zonal_max": float(zonal.values[i_edge]),
        "masked_rows": int(equatorward.sum()) - 1,
    }


def compute_limits(path: str, variable: str, time_chunk: int, num_workers: int) -> dict:
    zonal = zonal_max(time_max(path, variable, time_chunk, num_workers))
    hemispheres = {h: hemisphere_limit(zonal, h) for h in HEMISPHERES}
    return {
        "path": path,
        "variable": variable,
        "comparison": "max over full record > 0",
        "config": {
            "latitude_limits": {
                h: hemispheres[h]["latitude_limit_deg"] for h in HEMISPHERES
            }
        },
        "hemispheres": hemispheres,
    }


def config_block(result: dict) -> str:
    """coupled_sea_surface.sea_ice_mask YAML lines; repr keeps floats unrounded."""
    limits = result["config"]["latitude_limits"]
    lines = [
        "# coupled_sea_surface.sea_ice_mask: NaN on ocean where "
        "-south < lat < north, deg",
        "latitude_limits:",
    ] + [f"  {h}: {limits[h]!r}" for h in HEMISPHERES]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", help="Ocean zarr store.")
    parser.add_argument("variable", help="Sea-ice fraction variable name.")
    parser.add_argument("--time-chunk", type=int, default=73)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--output", help="Optional YAML file for the full result.")
    args = parser.parse_args()

    result = compute_limits(args.path, args.variable, args.time_chunk, args.num_workers)
    if args.output is not None:
        with open(args.output, "w") as f:
            yaml.safe_dump(result, f, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))
    print(config_block(result))


if __name__ == "__main__":
    main()
