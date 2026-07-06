"""Regridding utilities ported from the ai2cm fork of ocean_emulators
(github.com/ai2cm/ocean_emulators), so this pipeline has no dependency on
that repository.

Deliberate departures from the ported code (and from the conventions of the
datasets produced by scripts/data_process/compute_ocean_dataset.py, which
called it):

- C-grid -> tracer-center velocity interpolation normalizes by the count of
  valid (ocean) neighbors instead of filling land with zeros, which biased
  coastal speeds low.
- Wetmask normalization of the conservative regrid is explicit
  (mask -> regrid field and mask -> divide) rather than xESMF's
  ``skipna=True, na_thres=1`` weight-application option, which is unavailable
  when weights are precomputed. The two are equivalent at
  ``OCEAN_FRACTION_THRESHOLD = 0``: a target cell stays ocean if it overlaps
  any sliver of ocean source area.
- The regridded ocean-fraction field is returned so it can be stored;
  compute_ocean_dataset.py computed it but never exposed it for weighting
  coastal cells.
"""

import numpy as np
import xarray as xr

# A regridded target cell is kept (treated as ocean) where its ocean
# fraction exceeds this threshold. Zero reproduces the mask convention of
# xESMF ``skipna=True, na_thres=1`` used for earlier ocean datasets: any
# overlap with ocean source area keeps the cell.
OCEAN_FRACTION_THRESHOLD = 0.0


def convert_supergrid(hgrid: xr.Dataset) -> xr.Dataset:
    """Extract tracer-cell geometry from a MOM6 supergrid (ocean_hgrid.nc).

    The supergrid samples the mesh at twice the model resolution: odd
    (1-based: even) supergrid points are tracer-cell centers and even points
    are cell corners. Returns a dataset with 2D ``lon``/``lat`` centers
    (dims ``yh``/``xh``), 2D ``lon_b``/``lat_b`` corners (dims
    ``yh_b``/``xh_b``, one larger in each dimension — the layout xESMF
    expects for conservative regridding), and the grid rotation ``angle``
    (degrees CCW from east) at cell centers.
    """
    center_indices = dict(nyp=slice(1, None, 2), nxp=slice(1, None, 2))
    corner_indices = dict(nyp=slice(0, None, 2), nxp=slice(0, None, 2))
    center_rename = {"nyp": "yh", "nxp": "xh"}
    corner_rename = {"nyp": "yh_b", "nxp": "xh_b"}
    return xr.Dataset(
        coords={
            "lon": hgrid.x.isel(center_indices).rename(center_rename),
            "lat": hgrid.y.isel(center_indices).rename(center_rename),
            "lon_b": hgrid.x.isel(corner_indices).rename(corner_rename),
            "lat_b": hgrid.y.isel(corner_indices).rename(corner_rename),
            "angle": hgrid.angle_dx.isel(center_indices).rename(center_rename),
        }
    )


def rotate_vectors(
    u: xr.DataArray, v: xr.DataArray, angle: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Rotate grid-relative vector components by ``angle`` (degrees, CCW) to
    geographic (eastward/northward) components.

    All three arrays must be colocated (tracer centers): interpolate C-grid
    components with :func:`interpolate_to_cell_centers` first.
    """
    if len(angle.dims) != 2:
        raise ValueError(f"Expected only two dimensions on `angle`. Got {angle.dims}")
    if not (
        set(angle.dims).issubset(set(u.dims)) and set(angle.dims).issubset(set(v.dims))
    ):
        raise ValueError("`u` and `v` need to be on the same grid position as `angle`.")
    theta = np.deg2rad(angle)
    u_rotated = u * np.cos(theta) - v * np.sin(theta)
    v_rotated = u * np.sin(theta) + v * np.cos(theta)
    return u_rotated, v_rotated


def _interpolate_right_to_center(
    da: xr.DataArray, dim: str, center_dim: str, periodic: bool
) -> xr.DataArray:
    """Average a field from staggered "right" positions (MOM6 ``xq``/``yq``,
    each point on the right/north edge of the like-indexed tracer cell) onto
    tracer centers, normalizing by the number of valid neighbors.

    Center i sits between staggered points i-1 and i. Land points (NaN)
    drop out of the average instead of being filled with zeros, so coastal
    values are means of ocean neighbors only. Cells with no valid neighbor
    come out NaN. Along a periodic dimension the first center wraps around;
    otherwise the edge center uses its single in-domain neighbor.
    """
    if da.sizes[dim] != da[dim].size:
        raise ValueError(f"Expected coordinate for staggered dim {dim}")
    left = da.roll({dim: 1}, roll_coords=False) if periodic else da.shift({dim: 1})
    total = left.fillna(0.0) + da.fillna(0.0)
    count = left.notnull().astype(da.dtype) + da.notnull().astype(da.dtype)
    interpolated = (total / count.where(count > 0)).drop_vars(dim)
    return interpolated.rename({dim: center_dim})


def interpolate_to_cell_centers(
    u: xr.DataArray,
    v: xr.DataArray,
    wetmask: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate C-grid vector components to tracer-cell centers.

    ``u`` must have staggered dim ``xq`` (periodic) and ``v`` staggered dim
    ``yq``; both are averaged onto ``xh``/``yh`` tracer centers with
    valid-neighbor normalization and then restricted to ``wetmask`` (the
    tracer-cell ocean mask, broadcastable against the interpolated fields).
    """
    u_centered = _interpolate_right_to_center(u, "xq", "xh", periodic=True)
    v_centered = _interpolate_right_to_center(v, "yq", "yh", periodic=False)
    return u_centered.where(wetmask), v_centered.where(wetmask)


def regrid_normalized(
    ds: xr.Dataset, regridder, wetmask: xr.DataArray
) -> tuple[xr.Dataset, xr.DataArray]:
    """Wetmask-normalized conservative regrid of every variable in ``ds``.

    Fields are zeroed over land, regridded with the raw conservative
    ``regridder``, and divided by the regridded ``wetmask`` (the target-cell
    ocean fraction), so each target value is an average over ocean source
    area only. Target cells with ocean fraction <= OCEAN_FRACTION_THRESHOLD
    are NaN.

    ``wetmask`` may be 2D or include a level dimension; it must be
    broadcastable against each variable. Returns the regridded dataset and
    the regridded ocean fraction.
    """
    ocean_fraction = regridder(wetmask.astype("float64"), keep_attrs=False).fillna(0.0)
    divisor = ocean_fraction.where(ocean_fraction > OCEAN_FRACTION_THRESHOLD)
    out = xr.Dataset(attrs=ds.attrs)
    for name, da in ds.data_vars.items():
        masked = da.where(wetmask).fillna(0.0)
        out[name] = (regridder(masked, keep_attrs=True) / divisor).astype(da.dtype)
        out[name].attrs = da.attrs
    ocean_fraction.attrs = {
        "long_name": "fraction of target cell overlapping ocean source cells",
        "units": "1",
    }
    return out, ocean_fraction
