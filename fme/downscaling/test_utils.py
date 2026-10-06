import datetime
from pathlib import Path

import cftime
import numpy as np
import torch
import xarray as xr

from fme.downscaling.typing_ import FineResCoarseResPair


def cell_centered_coordinate(start: float, end: float, n: int) -> torch.Tensor:
    """Centers of ``n`` equal-width cells spanning ``[start, end]``.

    Shared primitive for the downscaling tests' coordinate construction
    (longitude rolling in particular relies on cell-centered grids).
    """
    bounds = torch.linspace(start, end, n + 1)
    return (bounds[:-1] + bounds[1:]) / 2


def _midpoints_from_count(start, end, n_mid):
    return cell_centered_coordinate(start, end, n_mid).numpy()


def create_test_data_on_disk(
    filename: Path, dim_sizes, variable_names, coords_override: dict[str, xr.DataArray]
) -> Path:
    data_vars = {}

    for name in variable_names:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes), attrs={"units": "m", "long_name": name}
        )
    coords = {
        dim_name: (
            xr.DataArray(
                np.arange(size, dtype=np.float32),
                dims=(dim_name,),
            )
            if dim_name not in coords_override
            else coords_override[dim_name]
        )
        for dim_name, size in dim_sizes.items()
    }
    # for lat, lon, overwrite with midpoints that are consistent with a
    # fine grid that fits inside coarse grid
    for c in ["lat", "lon"]:
        if c in coords:
            coords[c] = _midpoints_from_count(0, 8, dim_sizes[c])

    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    unlimited_dims = ["time"] if "time" in ds.dims else None

    ds.to_netcdf(filename, unlimited_dims=unlimited_dims, format="NETCDF4_CLASSIC")
    return filename


def data_paths_helper(
    tmp_path, rename: dict = {}, num_timesteps: int = 4
) -> FineResCoarseResPair[str]:
    dim_sizes = FineResCoarseResPair[dict[str, int]](
        fine={"time": num_timesteps, "lat": 16, "lon": 16},
        coarse={"time": num_timesteps, "lat": 8, "lon": 8},
    )

    variable_names = ["var0", "var1", "HGTsfc"]
    variable_names = [rename.get(v, v) for v in variable_names]
    fine_path = tmp_path / "fine"
    coarse_path = tmp_path / "coarse"
    fine_path.mkdir()
    coarse_path.mkdir()
    time_coord = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(num_timesteps)
    ]
    coords = {"time": time_coord}
    create_test_data_on_disk(
        fine_path / "data.nc", dim_sizes.fine, variable_names, coords
    )
    create_test_data_on_disk(
        coarse_path / "data.nc", dim_sizes.coarse, variable_names, coords
    )
    return FineResCoarseResPair[str](fine=fine_path, coarse=coarse_path)
