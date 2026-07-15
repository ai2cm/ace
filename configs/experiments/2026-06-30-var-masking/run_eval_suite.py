"""Run all evaluator entries in a generated var-masking eval suite."""

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import dacite
import torch

from fme.ace.inference.evaluator import (
    InferenceEvaluatorConfig,
    run_evaluator_from_config,
)
from fme.core.cli import prepare_config, prepare_directory
from fme.core.distributed import Distributed
from fme.core.generics.inference import get_record_to_wandb
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer


@dataclass
class EvalSuiteEntry:
    name: str
    config: InferenceEvaluatorConfig
    config_data: dict[str, Any]


@dataclass
class EvalSuiteConfig:
    experiment_dir: str
    logging: LoggingConfig
    inferences: list[EvalSuiteEntry]
    config_data: dict[str, Any]


def _load_suite(path: str) -> EvalSuiteConfig:
    config_data = prepare_config(path)
    entries = []
    for entry_data in config_data["inferences"]:
        evaluator_config = dacite.from_dict(
            data_class=InferenceEvaluatorConfig,
            data=entry_data["config"],
            config=dacite.Config(strict=True),
        )
        entries.append(
            EvalSuiteEntry(
                name=entry_data["name"],
                config=evaluator_config,
                config_data=entry_data["config"],
            )
        )
    logging_config = dacite.from_dict(
        data_class=LoggingConfig,
        data=config_data["logging"],
        config=dacite.Config(strict=True),
    )
    return EvalSuiteConfig(
        experiment_dir=config_data["experiment_dir"],
        logging=logging_config,
        inferences=entries,
        config_data=config_data,
    )


def run_eval_suite(path: str, validate_only: bool = False) -> None:
    suite = _load_suite(path)
    if validate_only:
        return

    prepare_directory(suite.experiment_dir, suite.config_data)
    suite.logging.configure_logging(
        experiment_dir=suite.experiment_dir,
        log_filename="eval_suite_out.log",
        config=suite.config_data,
        resumable=False,
    )

    logger = get_record_to_wandb()
    for entry in suite.inferences:
        logging.info(f"Running evaluator suite entry {entry.name!r}.")
        prepare_directory(entry.config.experiment_dir, entry.config_data)
        with GlobalTimer(), torch.no_grad():
            run_evaluator_from_config(
                config=entry.config,
                log_label=entry.name,
                configure_logging=False,
                logger=logger,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_config", type=str, help="Path to eval suite YAML.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the suite config without running inference.",
    )
    args = parser.parse_args()
    with Distributed.context():
        run_eval_suite(args.yaml_config, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
