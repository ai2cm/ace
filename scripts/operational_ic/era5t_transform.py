"""ERA5T model-level fields from the cloud-optimized ARCO stores -> the
0.25-degree analysis-ready layout of ``ar/model-level-1h-0p25deg.zarr-v1``.

The analysis-ready model-level store carries final ERA5 only (about three
months behind). The cloud-optimized stores carry ERA5T at a few days' lag, but
in the IFS native representations: ``co/model-level-wind.zarr-v2`` holds
temperature, vorticity and divergence as T639 spherical-harmonic coefficients;
``co/model-level-moisture.zarr-v2`` holds specific humidity and the four
cloud/precipitation water species on the N320 reduced Gaussian grid. This
module reproduces what ARCO's own build does with MetView, in pure Python:

- scalar spectral -> grid: ``torch_harmonics.InverseRealSHT`` with
  ``norm="4pi"``, ``csphase=False``, no conjugation, coefficients x 4 pi
  (ECMWF packing: for m in 0..T, for n in m..T: (re, im));
- vorticity/divergence -> u, v: streamfunction psi = -a^2 vo / (n(n+1)),
  velocity potential chi = -a^2 d / (n(n+1)), ``InverseRealVectorSHT`` on
  (chi, -psi) / a, then the two components swapped and the second negated;
- N320 -> 0.25 degree: periodic linear interpolation in longitude along each
  of the 640 rows, then linear in latitude.

Validated against the analysis-ready store (2026-06-15 00Z, level 137; see
the GLORYS12 scoping report, figure 7): T r=0.99997 / RMSE 0.18 K, q
r=0.9997, u/v r=0.996/0.997 (RMSE 0.5 m/s, polar/orographic residual).
``verify_ic.py`` repeats the comparison at 4 degrees for a whole day.
"""

import functools
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from common import CACHE_DIR

log = logging.getLogger("operational_ic.era5t")

ARCO = "gcp-public-data-arco-era5/"
STORE_WIND = ARCO + "co/model-level-wind.zarr-v2"
STORE_MOISTURE = ARCO + "co/model-level-moisture.zarr-v2"
TRUNCATION = 639
NLAT, NLON = 721, 1440
R_EARTH = 6371229.0  # m, the IFS value
TIME_ORIGIN = pd.Timestamp("1900-01-01")  # the stores index time in hours since
N_LEVELS = 137
LEVEL_BATCH = 8  # levels per transform call; bounds the working memory

# Output variable names of the analysis-ready store <- source array names.
WIND_SCALARS = {"temperature": "t"}
MOISTURE = {
    "specific_humidity": "q",
    "specific_cloud_liquid_water_content": "clwc",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_rain_water_content": "crwc",
    "specific_snow_water_content": "cswc",
}


def time_index(t: pd.Timestamp) -> int:
    return int((t - TIME_ORIGIN) / pd.Timedelta(hours=1))


@functools.lru_cache(maxsize=None)
def _fs():
    import gcsfs

    return gcsfs.GCSFileSystem(token="anon")


@functools.lru_cache(maxsize=None)
def _group(store: str):
    import zarr

    return zarr.open_group(_fs().get_mapper(store), mode="r")


def valid_time_stops() -> dict:
    """``valid_time_stop`` (final ERA5) and ``valid_time_stop_era5t`` of the
    two cloud-optimized stores, from their attributes."""
    out = {}
    for name, store in (("wind", STORE_WIND), ("moisture", STORE_MOISTURE)):
        a = dict(_group(store).attrs)
        out[name] = {
            "final": a.get("valid_time_stop"),
            "era5t": a.get("valid_time_stop_era5t"),
            "last_updated": a.get("last_updated"),
        }
    return out


def is_populated(t: pd.Timestamp) -> bool:
    """Whether the chunks the transform needs at ``t`` hold data: the store
    attributes can run ahead of some variables (arco-era5 issue 128)."""
    i = time_index(t)
    for store, names in (
        (STORE_WIND, ("t", "vo", "d")),
        (STORE_MOISTURE, tuple(MOISTURE.values())),
    ):
        g = _group(store)
        for name in names:
            sample = g[name][i, N_LEVELS - 1, :]
            if not np.isfinite(sample).all():
                return False
    return True


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _packing_indices(t: int = TRUNCATION) -> tuple[np.ndarray, np.ndarray]:
    """(n, m) of each complex coefficient in ECMWF packing order."""
    ns, ms = [], []
    for m in range(t + 1):
        n = np.arange(m, t + 1)
        ns.append(n)
        ms.append(np.full(len(n), m))
    return np.concatenate(ns), np.concatenate(ms)


def unpack(spec: np.ndarray) -> np.ndarray:
    """(..., 2 * ncoef) packed real/imag pairs -> (..., T+1, T+1) complex
    coefficients, scaled by 4 pi for torch_harmonics' ``norm="4pi"``."""
    n, m = _packing_indices()
    spec = np.asarray(spec, dtype=np.float32)
    coef = np.zeros(spec.shape[:-1] + (TRUNCATION + 1, TRUNCATION + 1), np.complex64)
    coef[..., n, m] = spec[..., 0::2] + 1j * spec[..., 1::2]
    return coef * np.float32(4 * np.pi)


@functools.lru_cache(maxsize=None)
def _inverse_laplacian() -> np.ndarray:
    n = np.arange(TRUNCATION + 1, dtype=np.float64)
    inv = np.zeros((TRUNCATION + 1, 1))
    inv[1:, 0] = -(R_EARTH**2) / (n[1:] * (n[1:] + 1))
    return inv.astype(np.float32)


def _cached_transform(kind: str):
    """A torch_harmonics inverse transform (``scalar`` or ``vector``) on the
    721 x 1440 grid with its Legendre table in float32, cached on disk.

    Building the tables costs 12 s / 5 GB (scalar) and 35 s / 14 GB peak
    (vector) in float64, and the library casts them to the input dtype on
    every call. Built once per cache directory, loaded in seconds after.
    """
    import torch
    import torch_harmonics as th

    cls = th.InverseRealSHT if kind == "scalar" else th.InverseRealVectorSHT
    buffer_name = "pct" if kind == "scalar" else "dpct"
    kw = dict(
        lmax=TRUNCATION + 1,
        mmax=TRUNCATION + 1,
        grid="equiangular",
        norm="4pi",
        csphase=False,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"isht-{kind}-T{TRUNCATION}-{NLAT}x{NLON}-float32.pt"
    if path.exists():
        table = torch.load(path)
    else:
        log.info("building the %s inverse SHT table (once; cached at %s)", kind, path)
        full = cls(NLAT, NLON, **kw)
        table = getattr(full, buffer_name).to(torch.float32).clone()
        del full
        torch.save(table, path)
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.nlat, module.nlon = NLAT, NLON
    module.grid, module.norm, module.csphase = kw["grid"], kw["norm"], kw["csphase"]
    module.lmax, module.mmax = kw["lmax"], kw["mmax"]
    module.register_buffer(buffer_name, table, persistent=False)
    return module


def spectral_scalar(packed: np.ndarray) -> np.ndarray:
    """(levels, 2 ncoef) packed -> (levels, NLAT, NLON) float32."""
    import torch

    isht = _cached_transform("scalar")
    out = np.empty((packed.shape[0], NLAT, NLON), np.float32)
    for k in range(0, packed.shape[0], LEVEL_BATCH):
        coef = torch.from_numpy(unpack(packed[k : k + LEVEL_BATCH]))
        with torch.no_grad():
            out[k : k + LEVEL_BATCH] = isht(coef).numpy()
    return out


def spectral_winds(vo: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(levels, 2 ncoef) packed vorticity and divergence -> u, v."""
    import torch

    ivsht = _cached_transform("vector")
    inv = _inverse_laplacian()
    u = np.empty((vo.shape[0], NLAT, NLON), np.float32)
    v = np.empty_like(u)
    for k in range(0, vo.shape[0], LEVEL_BATCH):
        psi = unpack(vo[k : k + LEVEL_BATCH]) * inv
        chi = unpack(d[k : k + LEVEL_BATCH]) * inv
        coef = torch.from_numpy(np.stack([chi, -psi], axis=1))
        with torch.no_grad():
            uv = ivsht(coef).numpy() / np.float32(R_EARTH)
        # torch_harmonics' two components come back swapped relative to
        # (u, v), with the second sign-flipped (probe_arco_era5_transform.py)
        u[k : k + LEVEL_BATCH] = uv[:, 1]
        v[k : k + LEVEL_BATCH] = -uv[:, 0]
    return u, v


# ---------------------------------------------------------------------------
# Reduced Gaussian N320 -> regular 0.25 degree
# ---------------------------------------------------------------------------


def _n320_coordinates() -> tuple[np.ndarray, np.ndarray]:
    path = CACHE_DIR / "n320_coords.npz"
    if not path.exists():
        g = _group(STORE_MOISTURE)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, latitude=g["latitude"][:], longitude=g["longitude"][:])
    with np.load(path) as f:
        return f["latitude"], f["longitude"]


@functools.lru_cache(maxsize=None)
def n320_to_regular_matrix() -> sp.csr_matrix:
    """Sparse (NLAT*NLON, n_points) interpolation matrix: periodic linear in
    longitude along each N320 row, then linear in latitude (constant beyond
    the first/last row)."""
    lat_rg, lon_rg = _n320_coordinates()
    rows = np.unique(lat_rg)[::-1]  # descending, like the regular grid
    lat_t = np.linspace(90, -90, NLAT)
    lon_t = np.arange(NLON) * 0.25
    # longitude interpolation per row -> one sparse operator per row
    row_ops = []
    for r in rows:
        idx = np.where(lat_rg == r)[0]
        order = np.argsort(lon_rg[idx])
        idx, lo = idx[order], lon_rg[idx][order]
        lo_p = np.concatenate([lo, [lo[0] + 360.0]])
        idx_p = np.concatenate([idx, [idx[0]]])
        j = np.searchsorted(lo_p, lon_t, side="right") - 1
        j = np.clip(j, 0, len(lo_p) - 2)
        w1 = (lon_t - lo_p[j]) / (lo_p[j + 1] - lo_p[j])
        row_ops.append((idx_p[j], 1 - w1, idx_p[j + 1], w1))
    # latitude weights: target lat between rows k (north) and k+1 (south)
    k = np.searchsorted(-rows, -lat_t, side="right") - 1
    k = np.clip(k, 0, len(rows) - 2)
    wl = (lat_t - rows[k]) / (rows[k + 1] - rows[k])
    wl = np.clip(wl, 0.0, 1.0)
    data, cols, rowi = [], [], []
    out_index = np.arange(NLAT * NLON).reshape(NLAT, NLON)
    for jlat in range(NLAT):
        for krow, wlat in ((k[jlat], 1 - wl[jlat]), (k[jlat] + 1, wl[jlat])):
            if wlat == 0.0:
                continue
            i0, w0, i1, w1 = row_ops[krow]
            for src, w in ((i0, w0), (i1, w1)):
                data.append(w * wlat)
                cols.append(src)
                rowi.append(out_index[jlat])
    mat = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rowi), np.concatenate(cols))),
        shape=(NLAT * NLON, len(lat_rg)),
    )
    return mat.tocsr()


def n320_to_regular(field: np.ndarray) -> np.ndarray:
    """(levels, n_points) on N320 -> (levels, NLAT, NLON)."""
    mat = n320_to_regular_matrix()
    out = mat @ np.asarray(field, np.float32).T  # (NLAT*NLON, levels)
    return np.ascontiguousarray(out.T.reshape(field.shape[0], NLAT, NLON), np.float32)


# ---------------------------------------------------------------------------
# The analysis-ready layout
# ---------------------------------------------------------------------------


def model_level_dataset(t: pd.Timestamp) -> xr.Dataset:
    """The eight model-level 3-D fields ``scripts/era5`` reads, at ``t``, on
    the analysis-ready layout (time, hybrid, latitude, longitude; latitude
    90 -> -90), with the ``GRIB_pv`` hybrid coefficients the pipeline needs."""
    i = time_index(t)
    wind = _group(STORE_WIND)
    moist = _group(STORE_MOISTURE)
    log.info("ERA5T spectral fields at %s (t, vo, d: 3 x 137 levels)", t)
    t_spec = wind["t"][i, :, :]
    vo_spec = wind["vo"][i, :, :]
    d_spec = wind["d"][i, :, :]
    fields = {"temperature": spectral_scalar(t_spec)}
    del t_spec
    fields["u_component_of_wind"], fields["v_component_of_wind"] = spectral_winds(
        vo_spec, d_spec
    )
    del vo_spec, d_spec
    log.info("ERA5T N320 moisture fields at %s (5 x 137 levels)", t)
    for out_name, src in MOISTURE.items():
        fields[out_name] = n320_to_regular(moist[src][i, :, :])
    coords = {
        "time": [t.to_datetime64()],
        "hybrid": np.asarray(wind["hybrid"][:], dtype=np.float64),
        "latitude": np.linspace(90, -90, NLAT),
        "longitude": np.arange(NLON) * 0.25,
    }
    pv = wind["t"].attrs.get("GRIB_pv")
    if isinstance(pv, str):
        import json

        pv = json.loads(pv)
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                arr[None],
                dims=("time", "hybrid", "latitude", "longitude"),
                coords=coords,
            )
            for name, arr in fields.items()
        }
    )
    units = {
        "temperature": "K",
        "u_component_of_wind": "m s**-1",
        "v_component_of_wind": "m s**-1",
    }
    for name in ds.data_vars:
        ds[name].attrs = {
            "long_name": name.replace("_", " "),
            "units": units.get(name, "kg kg**-1"),
            "source": "ERA5T from the ARCO cloud-optimized stores, transformed by "
            "scripts/operational_ic/era5t_transform.py",
        }
    ds["temperature"].attrs["GRIB_pv"] = list(pv)
    return ds
