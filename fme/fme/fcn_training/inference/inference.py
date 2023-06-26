import dataclasses
import os
import time
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.dicts import to_flat_dict
from fme.fcn_training.utils.data_loader_multifiles import get_data_loader
from fme.fcn_training.utils.data_loader_params import DataLoaderParams
import argparse

import netCDF4

import torch
import logging
from fme.fcn_training.utils import logging_utils

from fme.core import SingleModuleStepper
from fme.core.wandb import WandB

import fme
from fme.fcn_training.train_config import LoggingConfig
import dacite
import yaml

wandb = WandB.get_instance()


def load_stepper(checkpoint_file: str) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(
        checkpoint["stepper"], load_optimizer=False
    )
    return stepper


@dataclasses.dataclass
class InferenceConfig:
    """
    Attributes:
        experiment_dir: Directory to save results to
        n_forward_steps: Number of steps to run the model forward for
        checkpoint_path: Path to stepper checkpoint to load
        logging: configuration for logging
        validation_data: configuration for validation data
        log_video: Whether to log videos of the predictions
        save_prediction_files: Whether to save the predictions as a netcdf file
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    validation_data: DataLoaderParams
    log_video: bool = True
    save_prediction_files: bool = True

    def __post_init__(self):
        if self.save_prediction_files and not self.log_video:
            raise ValueError(
                "save_prediction_files is only valid when log_video is True, "
                "otherwise there are no predictions to save"
            )
        if self.validation_data.n_samples != 1:
            # TODO: we need to add a "raw" aggregator which stores the raw predictions,
            # then we can stop treating the "mean" aggregator as a raw aggregator
            # and remove this restriction
            # Without this exception the code would still run, but the output files
            # would be storing an ensemble mean prediction labelled as though it's a
            # single initial condition.
            raise NotImplementedError(
                "Currently only supports inference on a single sample"
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self):
        self.logging.configure_wandb(
            config=to_flat_dict(dataclasses.asdict(self)), resume=False
        )

    def load_stepper(self) -> SingleModuleStepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path)


def main(
    yaml_config: str,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
        with open(os.path.join(data["experiment_dir"], "config.yaml"), "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    config.configure_logging(log_filename="inference_out.log")
    config.configure_wandb()

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging_utils.log_beaker_url()

    stepper = config.load_stepper()
    logging.info("Loading inference data")
    data_requirements = stepper.get_data_requirements(
        n_forward_steps=config.n_forward_steps
    )
    valid_data_loader, _ = get_data_loader(
        config.validation_data,
        requirements=data_requirements,
        train=False,
    )

    aggregator = InferenceAggregator(
        record_step_20=config.n_forward_steps >= 20, log_video=config.log_video
    )
    with torch.no_grad():
        for data in valid_data_loader:
            stepper.run_on_batch(
                data,
                train=False,
                n_forward_steps=config.n_forward_steps,
                aggregator=aggregator,
            )
    step_logs = aggregator.get_inference_logs(label="inference")
    for log in step_logs:
        wandb.log(log)
        # wandb.log cannot be called more than "a few times per second"
        time.sleep(0.3)
    if config.save_prediction_files:
        ds = aggregator.get_dataset()
        # mean and video values are obvious from the rest of the name,
        # so remove the prefix
        remove_prefix = {}
        for name in ds.data_vars:
            if name.startswith("mean_"):
                remove_prefix[name] = name[5:]
            if name.startswith("video_"):
                remove_prefix[name] = name[6:]
        ds = ds.rename(remove_prefix)
        # propagate units from validation data
        validation_ds = netCDF4.MFDataset(
            os.path.join(config.validation_data.data_path, "*.nc")
        )
        for out_name in ds:
            for name in validation_ds.variables:
                if (
                    name in out_name
                    and "units" in validation_ds.variables[name].ncattrs()
                ):
                    ds[out_name].attrs["units"] = validation_ds[name].units
                if (
                    name in out_name
                    and "long_name" in validation_ds.variables[name].ncattrs()
                ):
                    ds[out_name].attrs["long_name"] = validation_ds[name].long_name
        for out_name in ds.data_vars.keys():
            if len(ds[out_name].shape) > 4:
                # should only have source, time, x, y
                raise NotImplementedError(
                    f"an initial condition dimension may have been added already, "
                    "edit this code to handle this case"
                )  # can remove the expand_dims hack below if this is the case
            ds[out_name] = ds[out_name].expand_dims(dim="initial_condition", axis=1)
        filename = os.path.join(
            config.experiment_dir,
            "autoregressive_predictions.nc",
        )
        ds.to_netcdf(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
