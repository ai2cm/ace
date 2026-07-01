import dataclasses
import logging
import os
from collections.abc import Sequence

import dacite
import torch

import fme
from fme.core.cli import prepare_config, prepare_directory
from fme.core.distributed import Distributed
from fme.core.generics.trainer import Trainer
from fme.coupled.train.train_config import (
    CoupledAggregatorBuilder,
    TrainBuilders,
    TrainConfig,
    build_trainer,
    get_inference_callback,
    get_validate_stepper_callback,
    get_validation_callback,
)

__all__ = [
    "CoupledAggregatorBuilder",
    "Trainer",
    "build_trainer",
    "get_inference_callback",
    "get_validate_stepper_callback",
    "get_validation_callback",
    "main",
    "run_train",
    "run_train_from_config",
]


def run_train(builders: TrainBuilders, config: TrainConfig):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config_data = dataclasses.asdict(config)
    config.logging.configure_logging(
        config.experiment_dir,
        log_filename="out.log",
        config=config_data,
        resumable=True,
    )
    if config.resume_results is not None:
        logging.info(
            f"Resuming training from results in {config.resume_results.existing_dir}"
        )
        config.resume_results.verify_wandb_resumption(config.experiment_dir)
    trainer = build_trainer(builders, config)
    trainer.train()
    logging.info(f"DONE ---- rank {dist.rank}")


def run_train_from_config(config: TrainConfig):
    run_train(TrainBuilders(config), config)


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    data = prepare_config(yaml_config, override=override_dotlist)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    train_config.set_random_seed()
    train_config.resume_results = prepare_directory(
        train_config.experiment_dir, data, train_config.resume_results
    )
    run_train_from_config(train_config)
