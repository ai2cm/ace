import argparse
import dataclasses
from typing import Callable, List, Sequence

import dacite
import numpy as np
import xarray as xr
import yaml
from scipy.optimize import curve_fit

from fme.core import metrics
from fme.core.wandb import WandB


@dataclasses.dataclass
class DataConfig:
    paths: List[str]
    names: List[str]

    def get_dataset(self) -> xr.Dataset:
        datasets = [xr.open_zarr(path) for path in self.paths]
        return xr.concat(datasets, "sample")


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
    n_windows = years_per_ensemble // window_size
    rmses = []
    if not amip:
        mean = annual.mean(["sample", "year"])
        annual = annual - mean
    for i_w in range(n_windows):
        window = annual.isel(year=range(i_w * window_size, (i_w + 1) * window_size))
        if amip:
            window = window - window.mean(["sample", "year"])
        bias_maps = window.mean("year")
        rmse = (bias_maps**2).weighted(area).mean(dim=["grid_yt", "grid_xt"]) ** 0.5
        if amip:
            rmse *= (ensemble_members / (ensemble_members - 1.0)) ** 0.5
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


def main(
    dataset: xr.Dataset,
    varnames: list[str],
    window_years: int,
    batches: int,
    years_per_ensemble: int,
    amip: bool,
    project: str,
    entity: str,
    name: str,
):
    lat = dataset["grid_yt"]
    area = xr.DataArray(
        metrics.spherical_area_weights(lat.values, 90),
        dims=["grid_yt", "grid_xt"],
    )
    annual = get_annual(dataset).isel(year=range(0, years_per_ensemble)).load()
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
    noise_floors = {}
    stds = {}
    for varname in varnames:
        x = m[varname].window_size
        y = m[varname].values
        fitted_model = fit(x, y)
        noise_floors[varname] = fitted_model(window_years)
        x = s[varname].window_size
        y = s[varname].values
        fitted_model = fit(x, y)
        stds[varname] = fitted_model(window_years)

    wandb = WandB.get_instance()
    wandb.configure(log_to_wandb=True)
    config = {
        "window_years": window_years,
        "batches": batches,
        "years_per_ensemble": years_per_ensemble,
        "amip": amip,
    }

    wandb.init(
        name=name + "-plus-std",
        job_type="training",
        config=config,
        project=project,
        entity=entity,
        reinit=True,
    )
    for i in [0, batches]:
        for varname in noise_floors:
            wandb.log(
                {
                    f"inference/time_mean/rmse/{varname}": noise_floors[varname]
                    + stds[varname],
                },
                step=i,
                sleep=0.1,
            )

    wandb.init(
        name=name,
        job_type="training",
        config=config,
        project=project,
        entity=entity,
        reinit=True,
    )
    for i in [0, batches]:
        for varname in noise_floors:
            wandb.log(
                {
                    f"inference/time_mean/rmse/{varname}": noise_floors[varname],
                },
                step=i,
                sleep=0.1,
            )

    wandb.init(
        name=name + "-minus-std",
        job_type="training",
        config=config,
        project=project,
        entity=entity,
        reinit=True,
    )
    for i in [0, batches]:
        for varname in noise_floors:
            wandb.log(
                {
                    f"inference/time_mean/rmse/{varname}": noise_floors[varname]
                    - stds[varname],
                },
                step=i,
                sleep=0.1,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument("window_years", type=int)
    parser.add_argument("batches", type=int)
    parser.add_argument("--years-per-ensemble", type=int, default=10)
    parser.add_argument("--is-amip", action="store_true")
    parser.add_argument("--project", type=str, default="fourcastnet")
    parser.add_argument("--entity", type=str, default="ai2cm")
    args = parser.parse_args()

    with open(args.data_config, "r") as f:
        data_config = dacite.from_dict(
            DataConfig, yaml.safe_load(f), config=dacite.Config(strict=True)
        )

    main(
        dataset=data_config.get_dataset(),
        varnames=data_config.names,
        window_years=args.window_years,
        batches=args.batches,
        years_per_ensemble=args.years_per_ensemble,
        amip=args.is_amip,
        project=args.project,
        entity=args.entity,
        name=args.name,
    )
