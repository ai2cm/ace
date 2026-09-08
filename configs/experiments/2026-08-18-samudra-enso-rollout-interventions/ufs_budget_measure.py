#!/usr/bin/env python3
"""Measure the UFS replay ocean's salt AND heat budget closure.

The replay is nudged toward analysis, so neither budget need close: the
residuals here measure the nudging increments the correctors would have to
accommodate (or the correctors get disabled). Salt: as in the CM4 version.
Heat: d/dt of column heat content (rho*cp*sum thetao_k dz_k, area mean)
vs the area-mean surface flux hfds integrated over the 5-day step.

Grounds the salt-content corrector: computes the global-mean column salt
content S(t) = area-mean of sum_k so_k * dz_k (partial bottom cells via
deptho) at every 5-day step, its per-step tendency statistics, and the
regression of that tendency on the change in global-mean sea ice volume
(the one genuine ocean salt exchange; slope estimates the effective ice
salinity). Everything else - precipitation, evaporation, runoff - moves
water, not salt, so if conservation holds the residual tendency should be
near zero and becomes the corrector's constant_unaccounted term.
"""

import argparse

import numpy as np
import xarray as xr
import zarr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="ocean 5-day zarr")
    ap.add_argument("--out", required=True, help="output netCDF")
    args = ap.parse_args()

    g = zarr.open_group(args.src, mode="r")
    idepth = np.array([float(g[f"idepth_{i}"][()]) for i in range(20)])
    deptho = g["deptho"][:].astype("float64")  # (lat, lon), NaN over land
    lat = g["lat"][:].astype("float64")
    coslat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, deptho.shape[1]))
    ocean_mask = np.isfinite(deptho)
    area_w = np.where(ocean_mask, coslat, 0.0)
    area_sum = area_w.sum()

    # per-level thickness with partial bottom cells
    dz = np.stack(
        [
            np.clip(
                np.minimum(deptho, idepth[k + 1]) - np.minimum(deptho, idepth[k]),
                0,
                None,
            )
            for k in range(19)
        ]
    )  # (19, lat, lon)
    dz = np.where(np.isfinite(dz), dz, 0.0)  # deptho is NaN over land

    n_time = g["so_0"].shape[0]
    chunk = 360
    S = np.zeros(n_time)
    for start in range(0, n_time, chunk):
        stop = min(start + chunk, n_time)
        acc = np.zeros(stop - start)
        for k in range(19):
            so = g[f"so_{k}"][start:stop].astype("float64")  # (t, lat, lon)
            col = np.where(np.isfinite(so), so, 0.0) * dz[k][None]
            acc += (col * area_w[None]).sum(axis=(1, 2)) / area_sum
        S[start:stop] = acc
        print(f"steps {start}-{stop} done", flush=True)

    # ---- heat content and hfds closure ----
    RHO_CP = 1029.0 * 3990.0
    H = np.zeros(n_time)
    F = np.zeros(n_time)
    for start in range(0, n_time, chunk):
        stop = min(start + chunk, n_time)
        acc = np.zeros(stop - start)
        for k in range(19):
            th = g[f"thetao_{k}"][start:stop].astype("float64")
            col = np.where(np.isfinite(th), th, 0.0) * dz[k][None]
            acc += (col * area_w[None]).sum(axis=(1, 2)) / area_sum
        H[start:stop] = RHO_CP * acc
        hf = g["hfds"][start:stop].astype("float64")
        hf = np.where(np.isfinite(hf), hf, 0.0)
        F[start:stop] = (hf * area_w[None]).sum(axis=(1, 2)) / area_sum
        print(f"heat steps {start}-{stop} done", flush=True)
    dt = 5 * 86400.0
    dH = np.diff(H)
    flux_int = 0.5 * (F[:-1] + F[1:]) * dt
    heat_resid = dH - flux_int
    print(
        f"HEAT: per-step dH mean {dH.mean():+.4e} J/m2 | flux integral mean "
        f"{flux_int.mean():+.4e} | residual mean {heat_resid.mean():+.4e} "
        f"std {heat_resid.std():.4e} (nudging+numerics)"
    )
    print(
        f"HEAT residual as equivalent W/m2: mean {heat_resid.mean()/dt:+.3f} "
        f"std {heat_resid.std()/dt:.3f}"
    )

    ice = g["sea_ice_volume"][:].astype("float64")
    ice = np.where(np.isfinite(ice), ice, 0.0)
    V = (ice * area_w[None]).sum(axis=(1, 2)) / area_sum  # (t,) global-mean m

    dS = np.diff(S)
    dV = np.diff(V)
    slope, intercept = np.polyfit(dV, dS, 1)
    resid = dS - (slope * dV + intercept)
    print(
        f"S mean {S.mean():.4f} psu*m | per-step dS: mean {dS.mean():+.3e} "
        f"std {dS.std():.3e}"
    )
    print(
        f"regression dS = {slope:+.3f} * dV + {intercept:+.3e} "
        f"(slope ~ -effective ice salinity, psu)"
    )
    print(
        f"residual tendency: mean {resid.mean():+.3e} "
        f"std {resid.std():.3e} psu*m / 5 days"
    )
    print(f"full-record net salt drift: {S[-1]-S[0]:+.4f} psu*m on mean {S.mean():.1f}")

    xr.Dataset(
        {
            "salt_content": ("time", S),
            "ice_volume": ("time", V),
            "heat_content": ("time", H),
            "hfds_global": ("time", F),
        },
        attrs={
            "dS_dV_slope_psu": float(slope),
            "dS_intercept": float(intercept),
            "residual_std": float(resid.std()),
        },
    ).to_netcdf(args.out)
    print("done")


if __name__ == "__main__":
    main()
