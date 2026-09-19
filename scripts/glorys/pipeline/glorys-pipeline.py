"""xarray-beam pipeline: GLORYS12V1 ocean reanalysis + ERA5 forcing -> SamudrACE
training zarr (1 degree Gaussian grid, 19 CM4-matched levels, 5-day cadence).

Follows the runner/infrastructure pattern of ``scripts/ufs-replay/`` and
``scripts/era5/``. Read that pipeline's README first; only the differences
are documented here.

Sources
-------
Ocean: Copernicus Marine ``GLOBAL_MULTIYEAR_PHY_001_030`` daily means
(``cmems_mod_glo_phy_my_0.083deg_P1D-m``), served as an anonymously
readable zarr v2 store on CloudFerro S3: 1/12 degree, 50 z-levels,
1993-01-01 .. present (interim after 2021-06-30). Static geometry
(``e3t``, ``deptho``, 3-D ``mask``) comes from the product's static
datasets. Fluxes and stress are NOT published; the ten forcing fields
come from the 1-degree ERA5 store built by ``scripts/era5/`` (already on
the F90 grid), window-averaged over the 5-day interval that ENDS at each
output time (the same end-of-window labelling ``scripts/ufs-replay`` and
the SHiELD-family ``time_coarsen`` configs use).

Cadence
-------
GLORYS publishes daily MEANS, not snapshots. The pipeline reads every
``--time_stride``-th day (default 5) and treats each daily mean as the
state at that time, which is the closest analogue of the 5-day snapshot
convention of the CM4 and UFS training sets. Reading every day and
coarsening downstream would cost 5x the egress (~45 TB) for no gain.

Vertical
--------
GLORYS's 50 fixed z-levels do not nest inside the 19 CM4 layers, so the
integer ``[start, end)`` index groups of the UFS pipeline cannot be used.
A 19x50 overlap matrix (metres of each GLORYS cell inside each target
layer, from the static ``e3t``) provides fractional weights; below-bottom
cells are NaN in the source and drop out of the weighted mean. The
partial bottom cell is treated as a full cell (GLORYS publishes no
per-column thickness).

Horizontal
----------
Conservative regridding (xESMF) from the regular 1/12 degree grid to the
Gaussian F90 grid: a 12x coarsening, versus 4x for CM4/UFS. GLORYS stops
at 80S: the 10 southernmost F90 rows have no source cells and are written
as land (mask 0, land_fraction 1).

Variables
---------
Prognostic: thetao_k, so_k, uo_k, vo_k (k=0..18), sst (K, from the top
level), ssu, ssv, zos, ocean_sea_ice_fraction (siconc), HI (sithick),
sea_ice_volume (siconc * sithick), UI/VI (usi/vsi). Static: hfgeou, copied
from the CM4 field because GLORYS publishes none and the corrector's
scaled_temperature heat-content correction requires it. Forcing: DLWRFsfc,
DSWRFsfc, ULWRFsfc, USWRFsfc, LHTFLsfc, SHTFLsfc, PRATEsfc,
eastward/northward_surface_wind_stress,
total_frozen_precipitation_rate. There is no hfds/wfo/tauuo/tauvo: the
reanalysis does not publish the fluxes its ocean saw, and aliasing ERA5
stress as a "diagnostic output" would give the model a target equal to
its own input. Training configs must drop those four from ``out_names``.
"""

import argparse
import datetime
import functools
import logging
from typing import Sequence

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
import xesmf as xe
from apache_beam.options.pipeline_options import PipelineOptions
from fsspec.implementations.http import HTTPFileSystem
from obstore.store import from_url
from zarr.storage import FsspecStore, ObjectStore

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_GRID = "F90"
DEFAULT_TIME_STRIDE = 5  # days between output states
FORCING_TIME_STEP = 6  # hours between ERA5 store timesteps
VDIM = "elevation"  # CMEMS vertical dim: negative metres, deepest first

_S3 = "https://s3.waw3-1.cloudferro.com"
_PRODUCT = "GLOBAL_MULTIYEAR_PHY_001_030"
_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311"
_STATIC = "cmems_mod_glo_phy_my_0.083deg_static_202311"
URL_OCEAN = f"{_S3}/mdl-arco-time-025/arco/{_PRODUCT}/{_DATASET}/timeChunked.zarr"
URL_COORDS = (
    f"{_S3}/mdl-arco-time-026/arco/{_PRODUCT}/{_STATIC}--ext--coords/static.zarr"
)
URL_BATHY = f"{_S3}/mdl-arco-time-026/arco/{_PRODUCT}/{_STATIC}--ext--bathy/static.zarr"
URL_FORCING = "gs://vcm-ml-intermediate/2026-08-13-era5-1deg-8layer-1940-2025.zarr"
# GLORYS publishes no geothermal heat flux, but the ocean corrector's
# scaled_temperature heat-content correction requires it in every one of its
# three flux branches. Take the static CM4 field, which is already on the
# 1-degree output grid.
URL_HFGEOU = "gs://vcm-ml-intermediate/2026-09-02-cm4-1pctCO2-1deg-ocean-1daily.zarr"
# The fine-tune starts from a checkpoint pretrained on the UFS replay ocean, so
# every conv kernel encodes that land-sea geometry. Intersecting the two masks
# keeps the geometry fixed: cells that are ocean only in UFS have no GLORYS
# data, and cells that are ocean only in GLORYS would be a cold start in
# weights that never saw them. See --no_ufs_mask_intersection to disable.
URL_UFS_MASK = (
    "gs://vcm-ml-intermediate/"
    "2026-06-29-ufs-replay-ocean-1deg-19level-5day-1994-2023.zarr"
)
# River-mouth cells, masked out because runoff is not one of the emulator's
# inputs: GLORYS resolves discharge at 1/12 degree and the controlling variable
# is simply absent from the model, so these cells are unpredictable by
# construction rather than by lack of capacity. They also carry outsized
# leverage on normalization. Built by make_plume_mask.py; see its docstring.
URL_PLUME_MASK = (
    "gs://vcm-ml-intermediate/2026-09-17-GLORYS-trial-run/plume-mask-5psu.zarr"
)

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {"F22.5": 22.5, "F45": 45, "F90": 90, "F360": 360}

# 3-D ocean variables, vertically remapped and split into per-level 2-D fields
VARS_3D = ("thetao", "so", "uo", "vo")
# 2-D ocean / sea-ice variables read from the same store
VARS_2D = ("zos", "siconc", "sithick", "usi", "vsi")

# Target layer interfaces (metres), identical to the idepth_0..19 of the CM4
# and UFS-replay training sets (read from the UFS store 2026-09-08).
TARGET_INTERFACES = [
    0.0,
    2.69,
    9.88,
    22.88,
    41.33,
    61.27,
    108.28,
    163.57,
    245.57,
    371.76,
    566.81,
    775.17,
    1047.79,
    1389.24,
    1797.30,
    2430.21,
    3139.34,
    4093.56,
    5089.68,
    5902.06,
]
N_LEVELS = len(TARGET_INTERFACES) - 1

# ERA5 store name -> output name (only the two stress fields differ)
FORCING_VARS = {
    "DLWRFsfc": "DLWRFsfc",
    "DSWRFsfc": "DSWRFsfc",
    "ULWRFsfc": "ULWRFsfc",
    "USWRFsfc": "USWRFsfc",
    "LHTFLsfc": "LHTFLsfc",
    "SHTFLsfc": "SHTFLsfc",
    "PRATEsfc": "PRATEsfc",
    "eastward_surface_stress": "eastward_surface_wind_stress",
    "northward_surface_stress": "northward_surface_wind_stress",
    "total_frozen_precipitation_rate": "total_frozen_precipitation_rate",
}


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


class _CmemsHTTPFileSystem(HTTPFileSystem):
    """HTTP filesystem that treats 403 as 'missing'.

    CloudFerro S3 answers 403 (not 404) for object keys that do not exist,
    which is how zarr represents all-fill chunks (e.g. the deepest level
    of a 3-D field). zarr must see FileNotFoundError there to fill with
    NaN instead of raising. The same 403 is why ``zarr_format=2`` must be
    pinned: zarr-python 3 probes for ``zarr.json`` first.
    """

    def _raise_not_found_for_status(self, response, url):
        if response.status == 403:
            raise FileNotFoundError(url)
        super()._raise_not_found_for_status(response, url)


def open_cmems(url: str) -> xr.Dataset:
    fs = _CmemsHTTPFileSystem(asynchronous=True)
    store = FsspecStore(fs, path=url, read_only=True)
    return xr.open_zarr(
        store, consolidated=True, zarr_format=2, chunks=None, decode_timedelta=False
    )


def _make_zarr_store(url: str, read_only: bool = True):
    if url.startswith("gs://"):
        return ObjectStore(from_url(url), read_only=read_only)
    return url


def open_ocean(variables: Sequence[str], times: pd.DatetimeIndex) -> xr.Dataset:
    ds = open_cmems(URL_OCEAN)[list(variables)]
    ds = ds.sel(time=times)
    return ds.rename({"latitude": "lat", "longitude": "lon"})


def open_forcing(times: pd.DatetimeIndex) -> xr.Dataset:
    """ERA5 6-hourly forcing on the F90 grid, restricted to ``times``."""
    ds = xr.open_zarr(_make_zarr_store(URL_FORCING), chunks=None)[list(FORCING_VARS)]
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.sel(time=times)
    if ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds.rename(FORCING_VARS)


# ---------------------------------------------------------------------------
# Time coordinates
# ---------------------------------------------------------------------------


def output_times(start: datetime.datetime, end: datetime.datetime, stride: int):
    """Output state times: every ``stride``-th GLORYS day in [start, end]."""
    return pd.date_range(start, end, freq=f"{stride}D")


def forcing_window_times(out_times: pd.DatetimeIndex, stride: int) -> pd.DatetimeIndex:
    """ERA5 times covering the window ENDING at each output time.

    A GLORYS daily mean labelled 00Z on day D represents [D, D+1), i.e. a
    state centred on D+12h; the forcing between consecutive states k-1
    and k is the ERA5 interval [t_{k-1}+12h, t_k+12h). The ERA5 store is
    labelled at interval end, so the window's steps are the 6-hourly
    labels in (t_{k-1}+12h, t_k+12h].
    """
    step = pd.Timedelta(hours=FORCING_TIME_STEP)
    first = out_times[0] - pd.Timedelta(days=stride) + pd.Timedelta(hours=12) + step
    last = out_times[-1] + pd.Timedelta(hours=12)
    return pd.date_range(first, last, freq=step)


def steps_per_window(stride: int) -> int:
    return stride * 24 // FORCING_TIME_STEP


# ---------------------------------------------------------------------------
# Gaussian grid helpers (matching scripts/era5/ and scripts/ufs-replay/)
# ---------------------------------------------------------------------------


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes(n: float) -> np.ndarray:
    from numpy.polynomial.legendre import leggauss

    x, _ = leggauss(round(2 * n))
    return np.sort(np.degrees(np.arcsin(x)))


def _make_target_grid(output_grid: str) -> xr.Dataset:
    n = GAUSSIAN_GRID_N[output_grid]
    lat = _gaussian_latitudes(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], _cell_bounds(lat, -90, 90)),
            "lon_b": (["lon_b"], _cell_bounds(lon, 0, 360)),
        }
    )


def _make_source_grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Regular 1/12 degree GLORYS grid with cell bounds.

    Latitude spans -80..90 only: bounds are the true cell edges (-80 - dlat/2
    .. 90), NOT stretched to the poles, so target cells south of the domain
    get zero overlap (-> treated as land) instead of smeared data.
    """
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    return xr.Dataset(
        {
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (
                ["lat_b"],
                _cell_bounds(lat, lat[0] - dlat / 2, min(lat[-1] + dlat / 2, 90.0)),
            ),
            "lon_b": (
                ["lon_b"],
                _cell_bounds(lon, lon[0] - dlon / 2, lon[-1] + dlon / 2),
            ),
        }
    )


_REGRIDDER_CACHE: dict = {}


def _load_regrid_weights(weights_path: str):
    """Resolve ``--regrid_weights`` to something xESMF accepts.

    xESMF takes a local filesystem path or an in-memory Dataset, not a URL, so
    a ``gs://`` path has to be read here. Without this every worker recomputes
    the 8.8 M-cell conservative weights, which costs ~3.5 min and ~10 GB each;
    reading the 158 MB file instead is seconds.
    """
    if "://" not in weights_path:
        return weights_path
    import fsspec

    with fsspec.open(weights_path, "rb") as f:
        return xr.open_dataset(f).load()


def _get_regridder(output_grid: str, source_grid: xr.Dataset, weights_path: str | None):
    key = (output_grid, len(source_grid["lat"]), len(source_grid["lon"]))
    if key not in _REGRIDDER_CACHE:
        dst = _make_target_grid(output_grid)
        kwargs = {}
        if weights_path:
            kwargs = {
                "weights": _load_regrid_weights(weights_path),
                "reuse_weights": True,
            }
        _REGRIDDER_CACHE[key] = xe.Regridder(
            source_grid, dst, "conservative", periodic=True, **kwargs
        )
    return _REGRIDDER_CACHE[key]


def _regrid_dataset(
    ds: xr.Dataset,
    output_grid: str,
    source_grid: xr.Dataset,
    weights_path: str | None,
    *,
    skipna: bool = True,
    na_thres: float = 1.0,
) -> xr.Dataset:
    """Regrid each variable individually to keep memory bounded."""
    regridder = _get_regridder(output_grid, source_grid, weights_path)
    out = {}
    for name in ds.data_vars:
        out[name] = regridder(
            ds[name], keep_attrs=True, skipna=skipna, na_thres=na_thres
        )
    return xr.Dataset(out, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Vertical remap: 50 GLORYS z-levels -> 19 CM4 layers with fractional weights
# ---------------------------------------------------------------------------


def _source_interfaces(e3t: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return (order, interfaces): ``order`` sorts the stored ``elevation``
    axis shallow-first; ``interfaces`` are the 51 cell edges in metres."""
    elev = e3t[VDIM].values
    order = np.argsort(-elev)  # elevation is negative: -0.49 first
    # e3t is stored float32; accumulating 50 of them in float32 drifts the deep
    # interfaces by ~1e-4 m, which is coarser than the nesting check's tolerance.
    thick = e3t.values[order].astype(np.float64)
    return order, np.concatenate([[0.0], np.cumsum(thick)])


def overlap_matrix(e3t: xr.DataArray, target_interfaces: Sequence[float]) -> np.ndarray:
    """(N_LEVELS, 50) metres of each stored-order GLORYS cell inside each
    target layer. Rows are target layers (shallow first); columns follow the
    store's ``elevation`` order so the matrix can multiply raw arrays."""
    order, iface = _source_interfaces(e3t)
    t = np.asarray(target_interfaces)
    w = np.zeros((len(t) - 1, len(iface) - 1))
    for k in range(len(t) - 1):
        w[k] = np.clip(
            np.minimum(iface[1:], t[k + 1]) - np.maximum(iface[:-1], t[k]), 0, None
        )
    out = np.zeros_like(w)
    out[:, order] = w  # back to stored order
    return out


def remap_vertical(da: xr.DataArray, weights: np.ndarray, name: str) -> xr.Dataset:
    """Thickness-weighted remap of one 3-D field to per-level 2-D fields.

    NaN source cells (below the sea floor) contribute zero weight; a
    target layer with no valid source cell in a column is NaN.
    """
    vals = da.transpose(VDIM, "lat", "lon").values.astype(np.float32)
    valid = np.isfinite(vals)
    vals0 = np.where(valid, vals, 0.0)
    nz, ny, nx = vals.shape
    num = weights.astype(np.float32) @ vals0.reshape(nz, -1)
    den = weights.astype(np.float32) @ valid.reshape(nz, -1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan).reshape(N_LEVELS, ny, nx)
    long_name = da.attrs.get("long_name", name)
    return xr.Dataset(
        {
            f"{name}_{k}": xr.DataArray(
                out[k],
                dims=("lat", "lon"),
                coords={"lat": da["lat"], "lon": da["lon"]},
                attrs={
                    "long_name": f"{long_name} level-{k}",
                    "units": da.attrs.get("units", ""),
                },
            )
            for k in range(N_LEVELS)
        }
    )


# ---------------------------------------------------------------------------
# Nearest-neighbour fill for residual coastal NaN (2-D fields only)
# ---------------------------------------------------------------------------


def _nn_fill(field: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Fill NaN ocean cells of a 2-D field from the nearest valid cell."""
    from scipy.ndimage import distance_transform_edt

    need = np.isnan(field) & ocean
    if not need.any():
        return field
    valid = ~np.isnan(field)
    _, idx = distance_transform_edt(~valid, return_distances=True, return_indices=True)
    out = field.copy()
    out[need] = field[idx[0][need], idx[1][need]]
    return out


# ---------------------------------------------------------------------------
# Invariant fields (masks, fractions, deptho, idepth)
# ---------------------------------------------------------------------------


def _geothermal_heat_flux(mask_2d: xr.DataArray) -> xr.DataArray:
    """``hfgeou`` on the output grid, taken from the static CM4 field.

    GLORYS publishes no geothermal heat flux. The CM4 field is already on the
    1-degree grid, so this is a copy rather than a regrid -- but the two masks
    differ, so cells that are ocean in GLORYS and land in CM4 arrive NaN and
    are filled with the CM4 ocean mean. The flux is order 0.1 W/m**2 against a
    surface forcing of order 100, so the filled cells carry no weight; they
    exist so the corrector never sees a NaN.
    """
    cm4 = xr.open_zarr(_make_zarr_store(URL_HFGEOU))["hfgeou"].load()
    if cm4.sizes != mask_2d.sizes or not np.allclose(
        cm4["lat"].values, mask_2d["lat"].values
    ):
        raise ValueError(
            "the CM4 hfgeou grid does not match the output grid "
            f"({dict(cm4.sizes)} vs {dict(mask_2d.sizes)}); it can only be "
            "copied onto the 1-degree F90 grid it was written on"
        )
    cm4 = cm4.assign_coords(lat=mask_2d["lat"], lon=mask_2d["lon"])
    ocean_mean = float(cm4.where(np.isfinite(cm4)).mean())
    out = cm4.fillna(ocean_mean).where(mask_2d > 0).astype(np.float32)
    out.attrs = {
        "long_name": "Upward geothermal heat flux at sea floor",
        "units": "W m-2",
        "source": "CM4 static field (GLORYS publishes none)",
    }
    return out


def _ufs_masks(reference: xr.DataArray) -> dict[str, xr.DataArray]:
    """The UFS replay ocean masks on the output grid, for intersection.

    Raises if the grids differ: the masks are copied cell-for-cell, so a
    mismatch would silently intersect the wrong cells.
    """
    ufs = xr.open_zarr(_make_zarr_store(URL_UFS_MASK))
    names = ["mask_2d"] + [f"mask_{k}" for k in range(N_LEVELS)]
    if not np.allclose(ufs["lat"].values, reference["lat"].values) or not np.allclose(
        ufs["lon"].values, reference["lon"].values
    ):
        raise ValueError(
            "the UFS replay mask grid does not match the output grid; it can "
            "only be intersected on the 1-degree F90 grid it was written on"
        )
    out = {}
    for name in names:
        m = ufs[name].load().astype(np.float32)
        out[name] = m.assign_coords(lat=reference["lat"], lon=reference["lon"])
    return out


def _plume_mask(url: str, reference: xr.DataArray) -> xr.DataArray:
    """The river-plume mask on the output grid: 0 at cells to exclude.

    Raises if the grids differ or the store is missing, rather than silently
    training on cells the mask was meant to remove.
    """
    try:
        ds = xr.open_zarr(_make_zarr_store(url))
    except Exception as exc:
        raise ValueError(
            f"could not open the river-plume mask at {url}: {exc}. Build it "
            "with scripts/glorys/make_plume_mask.py, or pass --plume_mask '' "
            "to run without one"
        ) from exc
    if not np.allclose(ds["lat"].values, reference["lat"].values) or not np.allclose(
        ds["lon"].values, reference["lon"].values
    ):
        raise ValueError(
            "the river-plume mask grid does not match the output grid; rebuild "
            "it with make_plume_mask.py --output_grid matching this run"
        )
    mask = ds["plume_mask"].load().astype(np.float32)
    return mask.assign_coords(lat=reference["lat"], lon=reference["lon"])


def build_invariants(
    output_grid: str,
    weights: np.ndarray,
    weights_path: str | None,
    intersect_ufs_mask: bool = True,
    plume_mask_url: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the time-invariant output fields from the static datasets.

    Returns (invariant_ds, source_grid). Per-level masks are 1 wherever any
    GLORYS cell contributing to that layer is ocean in any source cell of
    the target footprint (max over levels, then conservative regrid > 0),
    matching the UFS convention. ``sea_surface_fraction`` is the
    conservatively regridded surface mask; F90 rows south of 80S get zero
    overlap and come out as land.
    """
    bathy = open_cmems(URL_BATHY).rename({"latitude": "lat", "longitude": "lon"})
    mask3 = bathy["mask"].load().astype(np.float32)  # (elevation, lat, lon)
    deptho = bathy["deptho"].load().astype(np.float32)
    src = _make_source_grid(mask3["lat"].values, mask3["lon"].values)

    contributes = (weights > 0).astype(np.float32)  # (19, 50)
    level_masks = {}
    for k in range(N_LEVELS):
        lm = (mask3 * contributes[k][:, None, None]).max(VDIM)
        level_masks[f"mask_{k}"] = lm
    native = xr.Dataset(level_masks)
    native["mask_2d"] = mask3.isel({VDIM: int(np.argmax(mask3[VDIM].values))})
    native["deptho"] = deptho.where(native["mask_2d"] > 0)

    frac = _regrid_dataset(
        native, output_grid, src, weights_path, skipna=True, na_thres=1.0
    )
    ufs = _ufs_masks(frac["mask_2d"]) if intersect_ufs_mask else None
    plume = _plume_mask(plume_mask_url, frac["mask_2d"]) if plume_mask_url else None
    inv = {}
    sea_fraction = frac["mask_2d"].fillna(0.0).clip(0, 1).astype(np.float32)
    if ufs is not None:
        sea_fraction = sea_fraction.where(ufs["mask_2d"] > 0, 0.0).astype(np.float32)
    if plume is not None:
        sea_fraction = sea_fraction.where(plume > 0, 0.0).astype(np.float32)
    for k in range(N_LEVELS):
        m = (frac[f"mask_{k}"].fillna(0.0) > 0).astype(np.float32)
        if ufs is not None:
            m = (m * (ufs[f"mask_{k}"] > 0).astype(np.float32)).astype(np.float32)
        if plume is not None:
            m = (m * (plume > 0).astype(np.float32)).astype(np.float32)
        m.attrs = {
            "long_name": f"ocean mask level-{k}",
            "units": "0 if land, 1 if ocean",
        }
        inv[f"mask_{k}"] = m
    mask_2d = (sea_fraction > 0).astype(np.float32)
    mask_2d.attrs = {"long_name": "ocean mask", "units": "0 if land, 1 if ocean"}
    inv["mask_2d"] = mask_2d
    sea_fraction.attrs = {"long_name": "sea surface fraction", "units": "fraction"}
    inv["sea_surface_fraction"] = sea_fraction
    land = (1.0 - sea_fraction).astype(np.float32)
    land.attrs = {"long_name": "land fraction", "units": "fraction"}
    inv["land_fraction"] = land
    dep = frac["deptho"].where(mask_2d > 0).astype(np.float32)
    dep.attrs = {"long_name": "Sea Floor Depth Below Geoid", "units": "m"}
    inv["deptho"] = dep
    inv["hfgeou"] = _geothermal_heat_flux(mask_2d)

    for i, d in enumerate(TARGET_INTERFACES):
        label = "Depth interface 0 (surface)" if i == 0 else f"Depth interface {i}"
        inv[f"idepth_{i}"] = xr.DataArray(
            float(d), attrs={"units": "meters", "long_name": label}
        )

    return xr.Dataset(inv).reset_coords(drop=True), src


# ---------------------------------------------------------------------------
# Per-chunk processing (Beam workers)
# ---------------------------------------------------------------------------


def _finalize(ds: xr.Dataset, time_value) -> xr.Dataset:
    keep = {"time", "lat", "lon"}
    ds = ds.drop_dims([d for d in ds.dims if d not in keep]).reset_coords(drop=True)
    for name in ds.data_vars:
        ds[name] = ds[name].astype(np.float32)
    if "time" not in ds.dims:
        ds = ds.expand_dims(time=[time_value])
    return ds


def _mask_and_fill(ds: xr.Dataset, invariant_ds: xr.Dataset) -> xr.Dataset:
    ocean = invariant_ds["mask_2d"].values > 0
    for name in list(ds.data_vars):
        level = name.rsplit("_", 1)[-1]
        if level.isdigit():
            ds[name] = ds[name].where(invariant_ds[f"mask_{level}"] > 0)
        else:
            filled = _nn_fill(ds[name].values, ocean)
            ds[name] = xr.DataArray(
                filled, dims=ds[name].dims, coords=ds[name].coords, attrs=ds[name].attrs
            )
            ds[name] = ds[name].where(invariant_ds["mask_2d"] > 0)
    return ds


def process_ocean_3d(
    key: xbeam.Key,
    chunk: xr.Dataset,
    *,
    output_grid: str,
    weights: np.ndarray,
    source_grid: xr.Dataset,
    invariant_ds: xr.Dataset,
    weights_path: str | None,
):
    """One (time, variable) element: remap 50->19 levels, regrid, mask.

    Elements are split per variable (``split_vars=True``) so a worker holds
    one 1.8 GB native field at a time rather than all four.
    """
    (name,) = chunk.data_vars
    logging.info("ocean 3-D %s key=%s", name, key)
    da = chunk[name].squeeze("time", drop=True).load()
    time_value = chunk["time"].values[0]
    ds = remap_vertical(da, weights, name)
    ds = _regrid_dataset(ds, output_grid, source_grid, weights_path)
    if name == "thetao":
        sst = ds["thetao_0"] + 273.15
        sst.attrs = {"long_name": "Sea surface temperature", "units": "K"}
        ds["sst"] = sst
    elif name == "uo":
        ds["ssu"] = ds["uo_0"].copy()
        ds["ssu"].attrs = {"long_name": "Sea surface x-velocity", "units": "m/s"}
    elif name == "vo":
        ds["ssv"] = ds["vo_0"].copy()
        ds["ssv"].attrs = {"long_name": "Sea surface y-velocity", "units": "m/s"}
    ds = _mask_and_fill(ds, invariant_ds)
    out = _finalize(ds, time_value)
    new_key = key.replace(
        offsets={"time": key.offsets["time"]}, vars=frozenset(out.data_vars)
    )
    return new_key, out


def process_ocean_2d(
    key: xbeam.Key,
    chunk: xr.Dataset,
    *,
    output_grid: str,
    source_grid: xr.Dataset,
    invariant_ds: xr.Dataset,
    weights_path: str | None,
):
    """One time element of the 2-D fields: zos, siconc, sithick -> zos,
    ocean_sea_ice_fraction, HI, sea_ice_volume."""
    logging.info("ocean 2-D key=%s", key)
    ds = chunk.squeeze("time", drop=True).load()
    time_value = chunk["time"].values[0]
    ds = _regrid_dataset(ds, output_grid, source_grid, weights_path)
    out = xr.Dataset()
    out["zos"] = ds["zos"]
    out["zos"].attrs = {"long_name": "Sea Surface Height", "units": "m"}
    sic = ds["siconc"].fillna(0.0).clip(0, 1)
    sic.attrs = {"long_name": "sea ice fraction over ocean", "units": "fraction"}
    out["ocean_sea_ice_fraction"] = sic
    hi = ds["sithick"].fillna(0.0).where(sic > 0, 0.0)
    hi.attrs = {
        "long_name": "Sea Ice Thickness (mean over ice-covered area)",
        "units": "m",
    }
    out["HI"] = hi
    siv = sic * hi
    siv.attrs = {"long_name": "Sea Ice Volume Per Area", "units": "m"}
    out["sea_ice_volume"] = siv
    # Sea ice velocities. The ocean corrector's sea_ice_fraction_correction
    # lists UI/VI in zero_where_ice_free_names, so they must be zero (not NaN)
    # wherever there is no ice, matching the HI/sea_ice_volume treatment above.
    for src, dst, direction in (("usi", "UI", "eastward"), ("vsi", "VI", "northward")):
        vel = ds[src].fillna(0.0).where(sic > 0, 0.0)
        vel.attrs = {
            "long_name": f"Sea ice {direction} velocity",
            "units": "m s-1",
        }
        out[dst] = vel
    out = _mask_and_fill(out, invariant_ds)
    out = _finalize(out, time_value)
    new_key = key.replace(
        offsets={"time": key.offsets["time"]}, vars=frozenset(out.data_vars)
    )
    return new_key, out


def process_forcing(
    key: xbeam.Key,
    chunk: xr.Dataset,
    *,
    steps: int,
    out_times: np.ndarray,
    target_lat,
    target_lon,
):
    """One window of ``steps`` 6-hourly ERA5 fields -> its 5-day mean,
    labelled with the output time whose window it closes."""
    logging.info("forcing key=%s", key)
    chunk = chunk.load()
    assert (
        chunk.sizes["time"] == steps
    ), f"partial forcing window at {key}: {chunk.sizes['time']}"
    k = key.offsets["time"] // steps
    ds = chunk.mean("time", keep_attrs=True)
    ds = ds.assign_coords(lat=target_lat, lon=target_lon)
    out = _finalize(ds, out_times[k])
    new_key = key.replace(offsets={"time": k}, vars=frozenset(out.data_vars))
    return new_key, out


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def _output_attrs(
    ds_3d: xr.Dataset, ds_2d: xr.Dataset, ds_forcing: xr.Dataset
) -> dict[str, dict[str, str]]:
    """``{name: attrs}`` for every time-varying output.

    xbeam writes the template's metadata, not the per-chunk metadata, so the
    attrs the processors attach to each chunk are discarded and the template
    is the only place they can be set. Derived from the source stores here so
    the two cannot drift; the processors' copies are redundant but harmless.
    """
    attrs: dict[str, dict[str, str]] = {}
    for v in VARS_3D:
        src = ds_3d[v].attrs
        long_name = src.get("long_name", v)
        units = src.get("units", "")
        for k in range(N_LEVELS):
            attrs[f"{v}_{k}"] = {
                "long_name": f"{long_name} level-{k}",
                "units": units,
            }
    attrs["sst"] = {"long_name": "Sea surface temperature", "units": "K"}
    attrs["ssu"] = {"long_name": "Sea surface x-velocity", "units": "m/s"}
    attrs["ssv"] = {"long_name": "Sea surface y-velocity", "units": "m/s"}
    attrs["zos"] = {"long_name": "Sea Surface Height", "units": "m"}
    attrs["ocean_sea_ice_fraction"] = {
        "long_name": "sea ice fraction over ocean",
        "units": "fraction",
    }
    attrs["HI"] = {
        "long_name": "Sea Ice Thickness (mean over ice-covered area)",
        "units": "m",
    }
    attrs["sea_ice_volume"] = {"long_name": "Sea Ice Volume Per Area", "units": "m"}
    attrs["UI"] = {"long_name": "Sea ice eastward velocity", "units": "m s-1"}
    attrs["VI"] = {"long_name": "Sea ice northward velocity", "units": "m s-1"}
    for src_name, out_name in FORCING_VARS.items():
        src = ds_forcing[out_name].attrs if out_name in ds_forcing else {}
        attrs[out_name] = {
            "long_name": src.get("long_name", out_name),
            "units": src.get("units", ""),
        }
    return attrs


def make_template(
    out_times: pd.DatetimeIndex,
    invariant_ds: xr.Dataset,
    attrs: dict[str, dict[str, str]] | None = None,
) -> xr.Dataset:
    """Analytic template: every time-varying output is (time, lat, lon) float32."""
    attrs = attrs or {}
    names = [f"{v}_{k}" for v in VARS_3D for k in range(N_LEVELS)]
    names += [
        "sst",
        "ssu",
        "ssv",
        "zos",
        "ocean_sea_ice_fraction",
        "HI",
        "sea_ice_volume",
        "UI",
        "VI",
    ]
    names += list(FORCING_VARS.values())
    lat, lon = invariant_ds["lat"], invariant_ds["lon"]
    zeros = xr.DataArray(
        np.zeros((len(lat), len(lon)), dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    base = xr.Dataset({n: zeros.copy() for n in names})
    for name in names:
        base[name].attrs = dict(attrs.get(name, {}))
    template = xbeam.make_template(base).expand_dims(
        dim={"time": out_times.values}, axis=0
    )
    for name in invariant_ds.data_vars:
        template[name] = invariant_ds[name]
    return template


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def _get_parser():
    p = argparse.ArgumentParser(
        description="GLORYS12 + ERA5 -> SamudrACE training zarr"
    )
    p.add_argument("output_path", type=str)
    p.add_argument(
        "start_date", type=str, help="first output state (YYYY-MM-DD), a GLORYS day"
    )
    p.add_argument("end_date", type=str, help="last output state (YYYY-MM-DD)")
    p.add_argument(
        "--output_grid", default=DEFAULT_OUTPUT_GRID, choices=sorted(GAUSSIAN_GRID_N)
    )
    p.add_argument(
        "--time_stride",
        type=int,
        default=DEFAULT_TIME_STRIDE,
        help="days between output states",
    )
    p.add_argument("--output_time_chunksize", type=int, default=1)
    p.add_argument("--output_time_shardsize", type=int, default=360)
    p.add_argument(
        "--regrid_weights",
        type=str,
        default=None,
        help="optional precomputed xESMF conservative weights file "
        "(1/12 deg -> output grid); generating them on every worker costs "
        "minutes and several GB",
    )
    p.add_argument(
        "--no_ufs_mask_intersection",
        action="store_true",
        help="keep GLORYS's own land-sea mask instead of intersecting it with "
        "the UFS replay mask the fine-tune checkpoint was pretrained on",
    )
    p.add_argument(
        "--plume_mask",
        default=URL_PLUME_MASK,
        help="river-plume mask store to exclude (built by make_plume_mask.py); "
        "pass an empty string to keep the river-mouth cells",
    )
    return p


def main():
    parser = _get_parser()
    args, pipeline_args = parser.parse_known_args()
    logging.info("Pipeline args: %s", pipeline_args)

    start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    assert args.output_time_shardsize % args.output_time_chunksize == 0

    out_times = output_times(start, end, args.time_stride)
    steps = steps_per_window(args.time_stride)
    forcing_times = forcing_window_times(out_times, args.time_stride)
    logging.info(
        "%d output states %s .. %s", len(out_times), out_times[0], out_times[-1]
    )

    coords = open_cmems(URL_COORDS)
    weights = overlap_matrix(coords["e3t"].load(), TARGET_INTERFACES)
    # Every GLORYS cell is fully assigned to the target column except the
    # deepest one, which extends to 5958 m, below the 5902 m target bottom.
    covered = weights.sum(0)
    e3t = coords["e3t"].values
    deepest = int(np.argmin(coords["e3t"][VDIM].values))
    assert np.all(covered <= e3t + 1e-6) and np.allclose(
        np.delete(covered, deepest), np.delete(e3t, deepest)
    ), "GLORYS cells must nest inside the target column (except the deepest)"
    invariant_ds, source_grid = build_invariants(
        args.output_grid,
        weights,
        args.regrid_weights,
        intersect_ufs_mask=not args.no_ufs_mask_intersection,
        plume_mask_url=args.plume_mask or None,
    )

    ds_3d = open_ocean(VARS_3D, out_times)
    ds_2d = open_ocean(VARS_2D, out_times)
    ds_forcing = open_forcing(forcing_times)
    n_missing = len(forcing_times) - ds_forcing.sizes["time"]
    assert (
        n_missing == 0
    ), f"ERA5 store lacks {n_missing} forcing steps; shorten end_date"

    template = make_template(
        out_times, invariant_ds, _output_attrs(ds_3d, ds_2d, ds_forcing)
    )
    output_chunks = {"time": args.output_time_chunksize}
    output_shards = {"time": args.output_time_shardsize}
    output_store = _make_zarr_store(args.output_path, read_only=False)

    common = dict(
        output_grid=args.output_grid,
        source_grid=source_grid,
        invariant_ds=invariant_ds,
        weights_path=args.regrid_weights,
    )
    with beam.Pipeline(options=PipelineOptions(pipeline_args)) as p:
        (
            p
            | "3d_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_3d, chunks={"time": 1}, split_vars=True)
            | "3d_Process"
            >> beam.MapTuple(
                functools.partial(process_ocean_3d, weights=weights, **common)
            )
            | "3d_Consolidate" >> xbeam.ConsolidateChunks(output_shards)
            | "3d_ToZarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )
        (
            p
            | "2d_DatasetToChunks" >> xbeam.DatasetToChunks(ds_2d, chunks={"time": 1})
            | "2d_Process"
            >> beam.MapTuple(functools.partial(process_ocean_2d, **common))
            | "2d_Consolidate" >> xbeam.ConsolidateChunks(output_shards)
            | "2d_ToZarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )
        (
            p
            | "forcing_DatasetToChunks"
            >> xbeam.DatasetToChunks(ds_forcing, chunks={"time": steps})
            | "forcing_Process"
            >> beam.MapTuple(
                functools.partial(
                    process_forcing,
                    steps=steps,
                    out_times=out_times.values,
                    target_lat=invariant_ds["lat"].values,
                    target_lon=invariant_ds["lon"].values,
                )
            )
            | "forcing_Consolidate" >> xbeam.ConsolidateChunks(output_shards)
            | "forcing_ToZarr"
            >> xbeam.ChunksToZarr(
                output_store,
                template,
                zarr_chunks=output_chunks,
                zarr_shards=output_shards,
                zarr_format=3,
            )
        )


if __name__ == "__main__":
    main()
