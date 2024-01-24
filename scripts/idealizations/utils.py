from enum import Enum

import torch
import torch_harmonics as harmonics


class Grid(Enum):
    EQUIANGULAR = "equiangular"
    LEGENDRE_GAUSS = "legendre-gauss"


def validate_file_path(file_path: str, extension: str) -> None:
    """Validates the file path"""
    if not file_path.endswith(extension):
        raise ValueError("File path must be a netCDF file")


def roundtrip(
    ds, nlat: int, nlon: int, grid: Grid, ignore_vars=("grid_yt_bnds", "grid_xt_bnds")
):
    ds = ds.copy()
    sht = harmonics.RealSHT(nlat, nlon, grid=grid.value)
    isht = harmonics.InverseRealSHT(nlat, nlon, grid=grid.value)
    for var in ds.data_vars:
        if var in ignore_vars:
            continue
        field = torch.tensor(ds[var].to_numpy(), dtype=torch.float64)
        # TODO(gideond) assert shape of field before passing to sht.
        field = isht(sht(field)).numpy()
        ds[var].data = field
    return ds
