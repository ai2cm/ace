# mypy: ignore-errors
# ruff: noqa: E501
"""Replace MOM6's net surface heat flux in a copy of the UFS 4deg training zarr by the flux rebuilt from the FV3 surface
components, the one the ocean corrector prescribes over open ocean and closes the column budget against (Troy 2026-10-27:
MOM6's hfds sits 3 W/m2 above the rebuild in every period while the replay's column budget closes against the rebuild to
0.2 W/m2; the ocean's hfds output should be trained to the flux its corrector uses).

  hfds[k]            = F_rebuild(components[k], sst[k-1])                 W/m2 per ocean area (NaN where the original is NaN)
  hfds_total_area[k] = F_rebuild(...) * sea_surface_fraction               W/m2 per total cell area
  F_rebuild = fme.core.corrector.ocean._compute_ocean_net_surface_energy_flux (DSW-USW+DLW-ULW-LH-SH - frozen*Lf
              + cp (P - LH/Lv)(SST_prev - 273.15)); no geothermal (that is a separate budget term).

The step-k flux uses the step-(k-1) SST because the corrector's prescribed flux at a step uses the input (previous) SST.
Also rewrites the hfds / hfds_total_area entries of the normalization statistics (centering, scaling-full-field,
scaling-residual, time-mean). Everything else in the store (fields, placeholders, unaccounted_heating) is untouched.

    python make_rebuilt_hfds_dataset.py <src zarr> <dst zarr> <stats src dir> <stats dst dir>
"""

import glob
import pathlib
import shutil
import sys
import time

import numpy as np
import torch
import xarray as xr
import zarr

from fme.core.corrector.ocean import _compute_ocean_net_surface_energy_flux

FORCINGS = [
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
    "total_frozen_precipitation_rate",
]
PERIODS = {
    "1994-2001": (1994, 2001),
    "2002-2011": (2002, 2011),
    "2012-2021": (2012, 2021),
    "2022-2023": (2022, 2023),
    "1994-2023": (1994, 2023),
}


def main(
    src: pathlib.Path,
    dst: pathlib.Path,
    stats_src: pathlib.Path,
    stats_dst: pathlib.Path,
) -> None:
    t0 = time.time()
    if not dst.exists():
        shutil.copytree(src, dst)
        print(f"copied zarr ({time.time() - t0:.0f}s)", flush=True)
    ds = xr.open_zarr(dst, consolidated=None, chunks=None)
    ssf = ds["sea_surface_fraction"].values.astype("f4")
    if ssf.ndim == 3:
        ssf = ssf[0]
    sst = ds["sst"].values.astype("f4")
    sst_prev = np.concatenate(
        [sst[:1], sst[:-1]], axis=0
    )  # step k uses the SST of step k-1 (k=0: its own)
    F = {n: torch.as_tensor(ds[n].values.astype("f4")) for n in FORCINGS}
    rebuild = (
        _compute_ocean_net_surface_energy_flux(F, torch.as_tensor(sst_prev))
        .numpy()
        .astype("f4")
    )
    old_h, old_ht = (
        ds["hfds"].values.astype("f4"),
        ds["hfds_total_area"].values.astype("f4"),
    )
    new_h = np.where(np.isfinite(old_h), rebuild, np.nan).astype("f4")
    new_ht = np.where(np.isfinite(old_ht), rebuild * ssf[None], np.nan).astype("f4")
    encoding = {
        v: {k: ds[v].encoding[k] for k in ("chunks", "dtype") if k in ds[v].encoding}
        for v in ("hfds", "hfds_total_area")
    }
    attrs_h = dict(
        ds["hfds"].attrs,
        rebuilt_from="FV3 surface components via fme.core.corrector.ocean._compute_ocean_net_surface_energy_flux with the previous step's SST (the corrector's prescribed open-ocean flux); replaces MOM6 hfds",
    )
    attrs_ht = dict(
        ds["hfds_total_area"].attrs,
        rebuilt_from=attrs_h["rebuilt_from"] + ", times sea_surface_fraction",
    )
    time_, lat, lon = ds["time"], ds["lat"], ds["lon"]
    yr = np.array([int(str(t)[:4]) for t in time_.values])
    ds.close()
    # verification: global means per total area by period, old vs new
    w = np.cos(np.deg2rad(lat.values))[:, None] * np.ones((1, lon.size), "f4")
    w = w / w.sum()

    def gm(x):
        return np.nansum(x * w[None], axis=(1, 2))

    g_old, g_new = gm(old_ht), gm(new_ht)
    for name, (a, b) in PERIODS.items():
        m = (yr >= a) & (yr <= b)
        print(
            f"  hfds_total_area global mean {name}: MOM6 {g_old[m].mean():+.2f} -> rebuilt {g_new[m].mean():+.2f} W/m2 (per total area)",
            flush=True,
        )
    # rewrite the two variables in place (drop the old arrays, append the new ones with the same chunks)
    for v in ("hfds", "hfds_total_area"):
        shutil.rmtree(dst / v)
    out = xr.Dataset(
        {
            "hfds": (("time", "lat", "lon"), new_h, attrs_h),
            "hfds_total_area": (("time", "lat", "lon"), new_ht, attrs_ht),
        },
        coords={"time": time_, "lat": lat, "lon": lon},
    )
    chunks = encoding["hfds"].get("chunks")
    if chunks:
        out = out.chunk({"time": chunks[0], "lat": chunks[1], "lon": chunks[2]})
    out.to_zarr(dst, mode="a", consolidated=False, encoding=encoding)
    if (src / ".zmetadata").exists():
        zarr.consolidate_metadata(str(dst))
    print(f"fields written ({time.time() - t0:.0f}s)", flush=True)
    # statistics entries
    stats_dst.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(stats_src / "*.nc")):
        name = pathlib.Path(f).name
        d = xr.open_dataset(f).load()
        for v, arr in (("hfds", new_h), ("hfds_total_area", new_ht)):
            if name == "centering.nc":
                d[v] = xr.DataArray(np.float32(np.nanmean(arr)))
            elif name.startswith("scaling-full-field"):
                d[v] = xr.DataArray(np.float32(np.nanstd(arr)))
            elif name.startswith("scaling-residual"):
                d[v] = xr.DataArray(np.float32(np.nanstd(arr[1:] - arr[:-1])))
            elif name.startswith("time-mean"):
                d[v] = (("lat", "lon"), np.nanmean(arr, 0).astype("f4"))
        d.to_netcdf(stats_dst / name)
        print(
            "  stats",
            name,
            {v: float(d[v]) for v in ("hfds", "hfds_total_area") if d[v].ndim == 0},
            flush=True,
        )
    # read-back check
    chk = xr.open_zarr(dst, consolidated=None, chunks=None)
    assert np.allclose(
        np.nan_to_num(chk["hfds_total_area"].values), np.nan_to_num(new_ht), atol=1e-3
    )
    assert set(chk.data_vars) >= {
        "hfds",
        "hfds_total_area",
        "unaccounted_heating",
        "sst",
        "thetao_0",
    }
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*(pathlib.Path(a) for a in sys.argv[1:5]))
