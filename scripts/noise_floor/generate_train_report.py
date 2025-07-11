import argparse
import dataclasses
from typing import Callable, List, Tuple

import dacite
import fsspec
import numpy as np
import xarray as xr
import yaml
from scipy.optimize import curve_fit

from fme.core.wandb import WandB


@dataclasses.dataclass
class DataConfig:
    stats_path: str
    names: List[str]

    def get_datasets(self) -> Tuple[xr.Dataset, xr.Dataset]:
        if self.stats_path.endswith("/"):
            stats_path = self.stats_path
        else:
            stats_path = self.stats_path + "/"
        return (
            xr.open_dataset(fsspec.open(stats_path + "rmse_means.nc").open()),
            xr.open_dataset(fsspec.open(stats_path + "rmse_stdevs.nc").open()),
        )


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
    rmse_means: xr.Dataset,
    rmse_stdevs: xr.Dataset,
    varnames: list[str],
    window_years: int,
    batches: int,
    project: str,
    entity: str,
    name: str,
):
    rmse_means = rmse_means.load()
    rmse_stdevs = rmse_stdevs.load()
    noise_floors = {}
    stds = {}
    for varname in varnames:
        x = rmse_means[varname].window_size
        y = rmse_means[varname].values
        fitted_model = fit(x, y)
        noise_floors[varname] = fitted_model(window_years)
        x = rmse_stdevs[varname].window_size
        y = rmse_stdevs[varname].values
        fitted_model = fit(x, y)
        stds[varname] = fitted_model(window_years)

    wandb = WandB.get_instance()
    wandb.configure(log_to_wandb=True)
    config = {
        "window_years": window_years,
        "batches": batches,
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
    parser.add_argument("--project", type=str, default="fourcastnet")
    parser.add_argument("--entity", type=str, default="ai2cm")
    args = parser.parse_args()

    with open(args.data_config, "r") as f:
        data_config = dacite.from_dict(
            DataConfig,
            yaml.safe_load(f),
        )

    rmse_means, rmse_stdevs = data_config.get_datasets()
    main(
        rmse_means=rmse_means,
        rmse_stdevs=rmse_stdevs,
        varnames=data_config.names,
        window_years=args.window_years,
        batches=args.batches,
        project=args.project,
        entity=args.entity,
        name=args.name,
    )
