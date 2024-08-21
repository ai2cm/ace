import datetime
import os
import pathlib
import subprocess
import unittest.mock
from typing import Dict
from unittest.mock import MagicMock

import cftime
import dacite
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.core.optimization import NullOptimization
from fme.core.testing.wandb import mock_wandb
from fme.downscaling.train import Trainer, TrainerConfig, main, restore_checkpoint
from fme.downscaling.typing_ import FineResCoarseResPair

NUM_TIMESTEPS = 4


def test_trainer(tmp_path):
    """Tests the trainer class using mock objects."""
    mock_config = MagicMock()
    mock_config.experiment_dir = str(tmp_path / "experiment_dir")
    mock_config.checkpoint_dir = str(tmp_path / "checkpoint_dir")

    trainer = Trainer(
        model=MagicMock(),
        optimization=MagicMock(),
        train_data=MagicMock(),
        validation_data=MagicMock(),
        config=mock_config,
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
    dim_sizes = FineResCoarseResPair[Dict[str, int]](
        fine={"time": NUM_TIMESTEPS, "lat": 32, "lon": 32},
        coarse={"time": NUM_TIMESTEPS, "lat": 16, "lon": 16},
    )
    variable_names = ["x", "y", "HGTsfc"]
    fine_path = tmp_path / "fine"
    coarse_path = tmp_path / "coarse"
    fine_path.mkdir()
    coarse_path.mkdir()
    time_coord = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(NUM_TIMESTEPS)
    ]
    coords = {"time": time_coord}
    create_test_data_on_disk(
        fine_path / "data.nc", dim_sizes.fine, variable_names, coords
    )
    create_test_data_on_disk(
        coarse_path / "data.nc", dim_sizes.coarse, variable_names, coords
    )
    return FineResCoarseResPair[str](fine=fine_path, coarse=coarse_path)


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
    config["train_data"]["fine"] = [{"data_path": str(train_data_paths.fine)}]
    config["train_data"]["coarse"] = [{"data_path": str(train_data_paths.coarse)}]
    config["validation_data"]["fine"] = [{"data_path": str(validation_data_paths.fine)}]
    config["validation_data"]["coarse"] = [
        {"data_path": str(validation_data_paths.coarse)}
    ]

    config["experiment_dir"] = str(experiment_dir)
    config["save_checkpoints"] = True

    for res in ("fine", "coarse"):
        for key, path in zip(("global_means_path", "global_stds_path"), stats_paths):
            config["model"]["normalization"][res][key] = str(path)

    outpath = tmp_path / "train-config.yaml"
    with open(outpath, "w") as file:
        yaml.dump(config, file)
    return outpath


def test_train_main(trainer_config, very_fast_only: bool):
    """Check that training loop records the appropriate logs."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

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
        + trainer_config_dict["max_epochs"] * num_gradient_descent_updates_per_epoch
    )
    for step in range(len(logs)):
        keys = set(logs[step].keys())
        # if validation takes place during this step, check that at least some
        # number of metrics were logged (e.g. 5).
        assert len(keys) > 5 or keys == set(["train/batch_loss"])


def test_restore_checkpoint(trainer_config, tmp_path):
    with open(trainer_config, "r") as f:
        trainer_config_dict = yaml.safe_load(f)

    config = dacite.from_dict(data_class=TrainerConfig, data=trainer_config_dict)
    trainer1 = config.build()
    trainer2 = config.build()

    # Random initialization should result in two different sets of parameters
    # for the same configuration.
    assert any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(
            trainer1.model.modules.parameters(), trainer2.model.modules.parameters()
        )
    )

    tmp_path.mkdir(exist_ok=True)
    trainer1.save_all_checkpoints(float("-inf"))
    restore_checkpoint(trainer2)
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            trainer1.model.modules.parameters(), trainer2.model.modules.parameters()
        )
    )


def test_train_eval_modes(trainer_config, very_fast_only: bool):
    """Checks that at training time the model is stochastic (due to dropout) but
    that at validation time, it is deterministic."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    with open(trainer_config, "r") as f:
        trainer_config_dict = yaml.safe_load(f)
        trainer_config_dict["model"]["module"] = {
            "type": "unet_regression_song",
            "config": {"model_channels": 4},
        }

    config = dacite.from_dict(data_class=TrainerConfig, data=trainer_config_dict)
    trainer = config.build()

    trainer.train_one_epoch()
    assert trainer.model.module.training

    null_optimization = NullOptimization()

    batch = next(iter(trainer.train_data.loader))
    batch.fine = {k: v.to(torch.float64) for (k, v) in batch.fine.items()}
    batch.coarse = {k: v.to(torch.float32) for (k, v) in batch.coarse.items()}
    outputs1 = trainer.model.train_on_batch(batch, null_optimization)
    outputs2 = trainer.model.train_on_batch(batch, null_optimization)
    assert torch.any(outputs1.prediction["x"] != outputs2.prediction["x"])

    trainer.valid_one_epoch()
    assert not trainer.model.module.training
    outputs1 = trainer.model.train_on_batch(batch, null_optimization)
    outputs2 = trainer.model.train_on_batch(batch, null_optimization)
    assert torch.all(outputs1.prediction["x"] == outputs2.prediction["x"])


def test_resume(trainer_config, tmp_path, very_fast_only: bool):
    """Make sure the training is resumed from a checkpoint when restarted."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    with open(trainer_config, "r") as f:
        trainer_config_dict = yaml.safe_load(f)

    trainer_config_segment_one_dict = dict(trainer_config_dict)
    trainer_config_segment_one_dict["max_epochs"] = 2
    trainer_config_segment_one_dict["segment_epochs"] = 1

    trainer_config_segment_two_dict = dict(trainer_config_dict)
    trainer_config_segment_two_dict["max_epochs"] = 2
    trainer_config_segment_two_dict["segment_epochs"] = None

    trainer_config_segment_one = tmp_path / "config-segment-one.yaml"
    with open(trainer_config_segment_one, "w") as f:
        yaml.dump(trainer_config_segment_one_dict, f)

    trainer_config_segment_two = tmp_path / "config-segment-two.yaml"
    with open(trainer_config_segment_two, "w") as f:
        yaml.dump(trainer_config_segment_two_dict, f)

    mock = unittest.mock.MagicMock(side_effect=restore_checkpoint)
    with unittest.mock.patch("fme.downscaling.train.restore_checkpoint", new=mock):
        with mock_wandb() as wandb:
            main(trainer_config_segment_one)
            assert 1 == len(
                [log["epoch"] for log in wandb.get_logs().values() if "epoch" in log]
            )
            mock.assert_not_called()

    with unittest.mock.patch("fme.downscaling.train.restore_checkpoint", new=mock):
        with mock_wandb() as wandb:
            main(trainer_config_segment_two)
            assert 2 == len(
                [log["epoch"] for log in wandb.get_logs().values() if "epoch" in log]
            )
            mock.assert_called()


def test_resume_two_workers(trainer_config, skip_slow: bool):
    """Make sure the training is resumed from a checkpoint when restarted, using
    torchrun with NPROC_PER_NODE set to 2."""
    if skip_slow:
        # script is slow as everything is re-imported when it runs
        pytest.skip("Skipping slow tests")

    with open(trainer_config, "r") as f:
        trainer_config_dict = yaml.safe_load(f)
        trainer_config_dict["logging"]["log_to_wandb"] = False
        trainer_config_dict["max_epochs"] = 1
    with open(trainer_config, "w") as f:
        yaml.dump(trainer_config_dict, f)

    repo_path = pathlib.PurePath(__file__).parent.parent.parent.parent
    train_script_path = repo_path / "fme" / "fme" / "downscaling" / "train.py"
    command = [
        "torchrun",
        "--nproc_per_node",
        "2",
        train_script_path,
        trainer_config,
    ]
    initial_process = subprocess.run(command)
    initial_process.check_returncode()
    resume_process = subprocess.run(command)
    resume_process.check_returncode()
