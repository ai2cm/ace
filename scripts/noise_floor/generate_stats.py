import argparse
import dataclasses
import shutil
import tempfile
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import dacite
import dask.diagnostics
import fsspec
import numpy as np
import xarray as xr
import yaml
from scipy.optimize import curve_fit

from fme.core import metrics


@dataclasses.dataclass
class DataConfig:
    paths: List[str]
    stats_path: str
    years_per_ensemble: int
    is_amip: bool
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def get_dataset(self) -> xr.Dataset:
        datasets = [xr.open_zarr(path) for path in self.paths]
        ds = xr.concat(datasets, "sample")
        ds = ds.sel(time=slice(self.start_date, self.end_date))
        return ds


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


def get_annual(ds: xr.Dataset):
    """
    Resamples a dataset into annual means.
    """
    samples = [ds.isel(sample=i) for i in range(len(ds.sample))]
    annual_samples = [s.groupby(s.time.dt.year).mean() for s in samples]
    return xr.concat(annual_samples, "sample")


def get_stats(
    annual: xr.Dataset,
    years_per_ensemble: int,
    ensemble_members: int,
    window_size: int,
    area: xr.DataArray,
    amip: bool,
):
    """
    Gets the mean and standard deviation of the pattern bias implied by windows
    of a given number of years, from a dataset of annual means.
    """
    n_windows_per_sample = years_per_ensemble // window_size
    rmses = []
    if not amip:
        mean = annual.mean(["sample", "year"])
        annual_bias = annual - mean
    else:
        annual_bias = annual  # bias calculated on each window separately
    for i_w in range(n_windows_per_sample):
        window = annual_bias.isel(
            year=range(i_w * window_size, (i_w + 1) * window_size)
        )
        if amip:
            window = window - window.mean(["sample", "year"])
        bias_maps = window.mean("year")
        rmse = (bias_maps**2).weighted(area).mean(dim=["grid_yt", "grid_xt"]) ** 0.5
        if amip:
            rmse *= (ensemble_members / (ensemble_members - 1.0)) ** 0.5
        else:
            n_windows = n_windows_per_sample * ensemble_members
            rmse *= (n_windows / (n_windows - 1)) ** 0.5
        rmses.append(rmse)
    rmses_da = xr.concat(rmses, "window")
    mean_rmses = rmses_da.mean()
    stdev_rmses = rmses_da.std()
    return mean_rmses, stdev_rmses


def combine_window_stats(
    window_sizes: Sequence[int],
    years_per_ensemble: int,
    ensemble_members: int,
    func,
    annual: xr.Dataset,
    area: xr.DataArray,
    amip: bool,
):
    """
    Gets the mean and standard deviation of the pattern bias implied by windows
    of multiple different year lengths, from a dataset of annual means.
    """
    window_sizes = list(window_sizes)
    means = []
    stdevs = []
    for window_size in window_sizes:
        m, s = func(
            annual, years_per_ensemble, ensemble_members, window_size, area, amip
        )
        means.append(m)
        stdevs.append(s)
    m = xr.concat(means, "window_size")
    m = m.assign_coords({"window_size": window_sizes})
    s = xr.concat(stdevs, "window_size")
    s = s.assign_coords({"window_size": window_sizes})
    return m, s


def fit(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Fits given data to a function of the form y = c / sqrt(x) + b.
    """

    def model(x, c, b):
        return c * 1 / np.sqrt(x) + b

    popt, _ = curve_fit(model, x, y)

    c, b = popt

    def fitted_model(x):
        return model(x, c, b)

    return fitted_model


def get_output_datasets(
    dataset: xr.Dataset,
    years_per_ensemble: int,
    amip: bool,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    lat = dataset["grid_yt"]
    area = xr.DataArray(
        metrics.spherical_area_weights(lat.values, dataset.sizes["grid_xt"]),
        dims=["grid_yt", "grid_xt"],
    )
    annual = get_annual(dataset)
    available_years = annual.sizes["year"]
    if available_years != years_per_ensemble:
        warnings.warn(
            f"There are {available_years} years of data available, "
            f"but received value {years_per_ensemble} for years_per_ensemble."
        )
    with dask.diagnostics.ProgressBar():
        with dask.config.set(scheduler="processes"):
            annual = annual.isel(year=range(0, years_per_ensemble)).load()
    m, s = combine_window_stats(
        [1, 2, 5, 10],
        years_per_ensemble,
        len(dataset.sample),
        get_stats,
        annual,
        area,
        amip,
    )
    m = m.load()
    s = s.load()
    return m, s, annual


def main(
    dataset: xr.Dataset,
    years_per_ensemble: int,
    amip: bool,
    stats_path: str,
):
    m, s, annual_means = get_output_datasets(
        dataset=dataset,
        years_per_ensemble=years_per_ensemble,
        amip=amip,
    )
    if stats_path.endswith("/"):
        stats_path = stats_path[:-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        m.to_netcdf(tmpdir + "/rmse_means.nc")
        s.to_netcdf(tmpdir + "/rmse_stdevs.nc")
        annual_means.to_netcdf(tmpdir + "/annual_means.nc")
        copy(tmpdir + "/rmse_means.nc", stats_path + "/rmse_means.nc")
        copy(tmpdir + "/rmse_stdevs.nc", stats_path + "/rmse_stdevs.nc")
        copy(tmpdir + "/annual_means.nc", stats_path + "/annual_means.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config", type=str)
    args = parser.parse_args()

    with open(args.data_config, "r") as f:
        data_config = dacite.from_dict(DataConfig, yaml.safe_load(f))

    main(
        dataset=data_config.get_dataset(),
        years_per_ensemble=data_config.years_per_ensemble,
        amip=data_config.is_amip,
        stats_path=data_config.stats_path,
    )
