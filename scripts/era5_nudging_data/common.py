import logging
import os
import tempfile

import fsspec
import xarray as xr

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger(__name__)

PRESSURE_AND_SURFACE_LEVEL_STORE = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)
TIME_DIMENSION = "time"
SHIELD_CALENDAR = "julian"
LONGITUDE_PERIOD = 360.0

RAW_LATITUDE_DIMENSION = "latitude"
RAW_LONGITUDE_DIMENSION = "longitude"
LATITUDE_DIMENSION = "lat"
LONGITUDE_DIMENSION = "lon"

CITATIONS = """\
Citation for ARCO ERA5 dataset
------------------------------

Carver, Robert W, and Merose, Alex. (2023):
    ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset.
    22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1,
    https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842

Citation for ERA5 dataset
-------------------------

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A.,
    Muñoz‐Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D.,
    Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G.,
    Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G.,
    Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J.,
    Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S.,
    Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P.,
    Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F.,
    Guillaume, S., Thépaut, J-N. (2017): Complete ERA5: Fifth generation of
    ECMWF atmospheric reanalyses of the global climate. Copernicus Climate
    Change Service (C3S) Data Store (CDS). (Accessed on DD-MM-YYYY)
"""


def interpolate_zonal_gaps(
    da: xr.DataArray, dim: str = RAW_LONGITUDE_DIMENSION, method: str = "linear"
) -> xr.DataArray:
    left = da.assign_coords({dim: da[dim] - LONGITUDE_PERIOD})
    right = da.assign_coords({dim: da[dim] + LONGITUDE_PERIOD})
    padded = xr.concat([left, da, right], dim=dim)
    filled = padded.chunk({dim: -1}).interpolate_na(dim=dim, method=method)
    return filled.isel({dim: slice(da.sizes[dim], -da.sizes[dim])})


def interpolate_meridional_gaps(
    da: xr.DataArray, dim: str = RAW_LATITUDE_DIMENSION, method: str = "linear"
) -> xr.DataArray:
    return da.interpolate_na(dim=dim, method=method, fill_value="extrapolate")


def interpolate_gaps(da: xr.DataArray) -> xr.DataArray:
    da = interpolate_zonal_gaps(da)
    da = interpolate_meridional_gaps(da)
    return da


def fill_missing_sst_and_sea_ice_values(ds: xr.Dataset) -> xr.Dataset:
    """Fill NaN SST and sea ice values with an appropriate fill method.

    Interpolates the sea_surface_temperature zonally and then meridionally;
    fills missing sea_ice_cover values with zero.
    """
    sea_surface_temperature = interpolate_gaps(ds.sea_surface_temperature)
    sea_ice_cover = ds.sea_ice_cover.fillna(0.0)
    ds = ds.assign(sea_surface_temperature=sea_surface_temperature)
    ds = ds.assign(sea_ice_cover=sea_ice_cover)
    return ds


def to_remote_netcdf(ds: xr.Dataset, destination: str, **kwargs) -> None:
    """Write a dataset to netCDF and upload it to a (possibly remote) path.

    Writes via the netCDF4 engine to a local temporary file rather than
    directly to an in-memory buffer, since xarray only supports in-memory
    writes for the ``scipy`` and ``h5netcdf`` engines. This is worth the
    extra local write: unlike h5netcdf, the netCDF4 engine writes plain
    ``str`` attributes using the classic, fixed-length NC_CHAR type rather
    than the netCDF4-only, variable-length NC_STRING type. This is
    important, since SHiELD cannot read NC_STRING attributes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "data.nc")
        ds.to_netcdf(local_path, engine="netcdf4", **kwargs)
        with open(local_path, "rb") as local_file:
            with fsspec.open(destination, "wb") as remote_file:
                remote_file.write(local_file.read())
