import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Any

import dacite
import torch

import fme
from fme.core.cli import prepare_config, prepare_directory
from fme.core.distributed import Distributed
from fme.coupled.train.train_config import TrainConfig


def run_train(config: TrainConfig):
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
    trainer = config.build_trainer()
    trainer.train()
    logging.info(f"DONE ---- rank {dist.rank}")


def _handle_deprecated_config_keys(config_data: dict[str, Any]) -> dict[str, Any]:
    config_copy = config_data.copy()
    if "validation_loader" in config_data:
        loader = config_copy.pop("validation_loader")
        config_copy["validation"] = {"loader": loader}
    return config_copy


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    data = prepare_config(yaml_config, override=override_dotlist)
    data = _handle_deprecated_config_keys(data)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    train_config.set_random_seed()
    train_config = dataclasses.replace(
        train_config,
        resume_results=prepare_directory(
            train_config.experiment_dir, data, train_config.resume_results
        ),
    )
    run_train(train_config)
