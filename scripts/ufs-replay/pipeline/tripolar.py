# ruff: noqa: E501, E731
"""Tripolar MOM6 (mx025) -> Gaussian regridding that reproduces the method NOAA's ufs2arco used for the
GCS mom6.zarr (ufs2arco.regrid.MOM6Regridder): conservative xesmf from the true cell corners, u/v first
moved to tracer points and rotated to east/north with MOM6's cos_rot/sin_rot."""

import os

import cf_xarray as cfxr
import numpy as np
import xarray as xr
import xesmf as xe


def mom6_static_from_hgrid(hgrid: xr.Dataset) -> xr.Dataset:
    x, y = hgrid["x"].values, hgrid["y"].values  # supergrid (2ny+1, 2nx+1)
    g = xr.Dataset()
    g["geolon"] = (("yh", "xh"), x[1::2, 1::2])
    g["geolat"] = (("yh", "xh"), y[1::2, 1::2])
    g["geolon_u"] = (("yh", "xq"), x[1::2, 2::2])
    g["geolat_u"] = (("yh", "xq"), y[1::2, 2::2])
    g["geolon_v"] = (("yq", "xh"), x[2::2, 1::2])
    g["geolat_v"] = (("yq", "xh"), y[2::2, 1::2])
    g["geolon_c"] = (("yq", "xq"), x[2::2, 2::2])
    g["geolat_c"] = (("yq", "xq"), y[2::2, 2::2])
    # MOM6 initialize_grid_rotation_angle (MOM_shared_initialization.F90), from the four corners of each T cell
    lonc, latc = x[::2, ::2], y[::2, ::2]  # (ny+1, nx+1) corners
    lt = g["geolon"].values
    wrap = lambda lon: (lon - (lt - 180.0)) % 360.0 + (lt - 180.0)
    sw, se, nw, ne = (
        wrap(lonc[:-1, :-1]),
        wrap(lonc[:-1, 1:]),
        wrap(lonc[1:, :-1]),
        wrap(lonc[1:, 1:]),
    )
    lsw, lse, lnw, lne = latc[:-1, :-1], latc[:-1, 1:], latc[1:, :-1], latc[1:, 1:]
    lon_scale = np.cos(np.deg2rad(0.25 * (lsw + lse + lnw + lne)))
    angle = np.arctan2(
        lon_scale * ((nw - se) + (ne - sw)), (lnw - lse) + (lne - lsw)
    )  # lonB(m,n): first index is i
    g["sin_rot"] = (("yh", "xh"), np.sin(angle))
    g["cos_rot"] = (("yh", "xh"), np.cos(angle))
    return g


def _grids_in(g):
    ny, nx = g["geolon"].shape
    lon_b = np.empty((ny + 1, nx + 1))
    lat_b = np.empty((ny + 1, nx + 1))
    lon_b[1:, 1:] = g["geolon_c"].values
    lat_b[1:, 1:] = g["geolat_c"].values
    lon_b[:, 0] = lon_b[:, -1]
    lat_b[:, 0] = lat_b[:, -1]
    dy = (lat_b[2, :] - lat_b[1, :]).mean()
    lat_b[0, 1:] = lat_b[1, 1:] - dy
    lon_b[0, 1:] = lon_b[1, 1:]
    lon_b[0, 0] = lon_b[1, 0]
    lat_b[0, 0] = lat_b[0, 1]
    mk = lambda lo, la: xr.Dataset(
        {
            "lon": (("y", "x"), g[lo].values),
            "lat": (("y", "x"), g[la].values),
            "lon_b": (("y_b", "x_b"), lon_b),
            "lat_b": (("y_b", "x_b"), lat_b),
        }
    )
    return (
        mk("geolon", "geolat"),
        mk("geolon_u", "geolat_u"),
        mk("geolon_v", "geolat_v"),
    )


def _grid_out(lats, lons):
    go = xr.Dataset()
    go["lon"] = xr.DataArray(lons, dims=["lon"])
    go["lat"] = xr.DataArray(lats, dims=["lat"])
    go = go.cf.add_bounds(["lat", "lon"])
    go = go.assign(
        {
            "lat_b": cfxr.bounds_to_vertices(go["lat_bounds"], "bounds", order=None),
            "lon_b": cfxr.bounds_to_vertices(go["lon_bounds"], "bounds", order=None),
        }
    )
    return go.drop_vars(["lat_bounds", "lon_bounds"])


class TripolarRegridder:
    def __init__(self, static, lats, lons, weights_dir, method="conservative"):
        self.g = static
        self.lats = lats
        self.lons = lons
        t, u, v = _grids_in(static)
        out = _grid_out(lats, lons)

        def rg(a, b, name):
            f = os.path.join(
                weights_dir, f"weights-mom6-{name}-{len(lats)}x{len(lons)}-{method}.nc"
            )
            return xe.Regridder(
                a,
                b,
                method,
                periodic=True,
                reuse_weights=os.path.exists(f),
                filename=f,
                unmapped_to_nan=True,
            )

        self.tt = rg(t, out, "t2g")
        self.ut = rg(u, t, "u2t")
        self.vt = rg(v, t, "v2t")

    def _yx(self, da):
        ydim = "yh" if "yh" in da.dims else "yq"
        xdim = "xh" if "xh" in da.dims else "xq"
        return da.rename({ydim: "y", xdim: "x"}).drop_vars(["y", "x"], errors="ignore")

    def scalar(self, da):
        return self.tt(self._yx(da)).assign_coords(lat=self.lats, lon=self.lons)

    def vector(self, u, v):
        ut, vt = self.ut(self._yx(u)), self.vt(self._yx(v))
        c = xr.DataArray(self.g["cos_rot"].values, dims=("y", "x"))
        s = xr.DataArray(self.g["sin_rot"].values, dims=("y", "x"))
        ue, vn = ut * c + vt * s, vt * c - ut * s
        return (
            self.tt(ue).assign_coords(lat=self.lats, lon=self.lons),
            self.tt(vn).assign_coords(lat=self.lats, lon=self.lons),
        )
