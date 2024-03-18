import datetime
import os
import unittest.mock
from typing import Dict, Set
from unittest.mock import MagicMock

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

from fme.core.testing.wandb import mock_wandb
from fme.downscaling.train import Trainer, main
from fme.downscaling.typing_ import HighResLowResPair

NUM_TIMESTEPS = 10


def test_trainer():
    """Tests the trainer class using mock objects."""
    trainer = Trainer(
        model=MagicMock(),
        optimization=MagicMock(),
        train_data=MagicMock(),
        validation_data=MagicMock(),
        num_epochs=2,
    )

    with unittest.mock.patch(
        "fme.downscaling.aggregators.Aggregator.get_wandb"
    ) as mock_get_wandb:
        with mock_wandb():
            trainer.train_one_epoch()
            trainer.valid_one_epoch()

    mock_get_wandb.assert_called()


def create_test_data_on_disk(
    filename, dim_sizes, variable_names, coords_override: Dict[str, xr.DataArray]
):
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

    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")

    return filename


def data_paths_helper(tmp_path):
    dim_sizes = HighResLowResPair[Dict[str, int]](
        highres={"time": NUM_TIMESTEPS, "lat": 32, "lon": 32},
        lowres={"time": NUM_TIMESTEPS, "lat": 16, "lon": 16},
    )
    variable_names = ["x", "y"]
    highres_path = tmp_path / "highres"
    lowres_path = tmp_path / "lowres"
    highres_path.mkdir()
    lowres_path.mkdir()
    time_coord = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(NUM_TIMESTEPS)
    ]
    coords = {"time": time_coord}
    create_test_data_on_disk(
        highres_path / "data.nc", dim_sizes.highres, variable_names, coords
    )
    create_test_data_on_disk(
        lowres_path / "data.nc", dim_sizes.lowres, variable_names, coords
    )
    return HighResLowResPair[str](highres=highres_path, lowres=lowres_path)


@pytest.fixture
def train_data_paths(tmp_path):
    path = tmp_path / "train"
    path.mkdir()
    return data_paths_helper(path)


@pytest.fixture
def validation_data_paths(tmp_path):
    path = tmp_path / "validation"
    path.mkdir()
    return data_paths_helper(path)


@pytest.fixture
def stats_paths(tmp_path):
    variable_names = ["x", "y"]
    mean_path = create_test_data_on_disk(
        tmp_path / "stats-mean.nc", {}, variable_names, {}
    )
    std_path = create_test_data_on_disk(
        tmp_path / "stats-std.nc", {}, variable_names, {}
    )
    return mean_path, std_path


@pytest.fixture
def trainer_config(train_data_paths, validation_data_paths, stats_paths, tmp_path):
    file_path = (
        f"{os.path.dirname(os.path.abspath(__file__))}/configs/test_train_config.yaml"
    )
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    experiment_dir = tmp_path / "output"
    experiment_dir.mkdir()
    config["train_data"]["path_highres"] = str(train_data_paths.highres)
    config["train_data"]["path_lowres"] = str(train_data_paths.lowres)
    config["validation_data"]["path_highres"] = str(validation_data_paths.highres)
    config["validation_data"]["path_lowres"] = str(validation_data_paths.lowres)
    config["experiment_dir"] = str(experiment_dir)

    for res in ("highres", "lowres"):
        for key, path in zip(("global_means_path", "global_stds_path"), stats_paths):
            config["model"]["normalization"][res][key] = str(path)

    outpath = tmp_path / "train-config.yaml"
    with open(outpath, "w") as file:
        yaml.dump(config, file)
    return outpath


def _get_aggregator_keys(prefix: str) -> Set[str]:
    keys = [f"{prefix}/loss"]
    for metric_name in (
        "rmse",
        "weighted_rmse",
        "ssim",
        "psnr",
        "snapshot/image-error",
        "snapshot/image-full-field",
        "time_mean_map/error",
        "time_mean_map/full-field",
    ):
        for var_name in ("x", "y"):
            keys.append(f"{prefix}/{metric_name}/{var_name}")

    for instrinsic_name in ("histogram", "spectrum"):
        for var_name in ("x", "y"):
            for data_type in ("target", "prediction"):
                keys.append(f"{prefix}/{instrinsic_name}/{var_name}_{data_type}")
    return set(keys)


def test_train_main(trainer_config):
    """Check that training loop records the appropriate logs."""
    with mock_wandb() as wandb:
        main(trainer_config)
        logs = wandb.get_logs()

    with open(trainer_config, "r") as f:
        trainer_config_dict = yaml.safe_load(f)

    num_gradient_descent_updates_per_epoch = (
        NUM_TIMESTEPS // trainer_config_dict["train_data"]["batch_size"]
    )

    assert (
        len(logs)
        == 1  # validation is run once before training
        + trainer_config_dict["num_epochs"] * num_gradient_descent_updates_per_epoch
    )

    validation_keys = _get_aggregator_keys("validation")
    all_epoch_keys = validation_keys.union(_get_aggregator_keys("train"))
    batch_loss_key = "train/batch_loss"
    assert validation_keys == set(logs[0].keys())
    for step in range(1, len(logs)):
        keys = set(logs[step].keys())
        assert (
            batch_loss_key in keys
        ), "batch_loss should be logged at each training step."
        if step % num_gradient_descent_updates_per_epoch == 0:
            assert (
                all_epoch_keys & keys == all_epoch_keys
            ), "All train and validation metrics should be logged each epoch."
            assert (
                len(keys) == len(all_epoch_keys) + 1
            ), "batch_loss should also be logged each epoch."
        else:
            assert len(keys) == 1, "Within an epoch, only batch_loss should be logged."
