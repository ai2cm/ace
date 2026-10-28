"""Run every evaluator entry of a generated eval suite under one wandb run.

Ported from the 2026-06-26 run_eval_suite.py. That version relied on
`run_evaluator_from_config(config, log_label=..., configure_logging=...,
logger=...)`, an fme change that exists only on exp/alexeyfm. This branch's
evaluator takes the config alone, configures logging itself and logs through
`get_record_to_wandb(label="inference")`. To keep fme unchanged, the suite
patches two things for the duration of each entry, both local to this process:

- the entry config's `configure_logging` is a no-op: logging (and the single
  wandb run) is configured once for the suite;
- `fme.ace.inference.evaluator.get_record_to_wandb` returns one shared
  WandBStepLogger whose label is the entry name, so each entry's metrics are
  logged as `{entry}/...` and the step counter keeps rising across entries
  (wandb drops logs at a step lower than one already logged).

The evaluator's timer durations are logged unlabeled (`label=""`), so their
keys repeat across entries; each step's values are kept in the history but the
run summary holds the last entry's.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import dacite
import torch

import fme.ace.inference.evaluator as evaluator_module
from fme.ace.inference.evaluator import (
    InferenceEvaluatorConfig,
    run_evaluator_from_config,
)
from fme.core.cli import prepare_config, prepare_directory
from fme.core.distributed import Distributed
from fme.core.generics.inference import WandBStepLogger
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


def load_suite(path: str) -> EvalSuiteConfig:
    """Load and strictly validate a suite: every entry as an
    InferenceEvaluatorConfig, the logging block as a LoggingConfig.
    """
    config_data = prepare_config(path)
    entries = []
    names = set()
    for entry_data in config_data["inferences"]:
        name = entry_data["name"]
        if name in names:
            raise ValueError(f"duplicate suite entry name {name!r}")
        names.add(name)
        evaluator_config = dacite.from_dict(
            data_class=InferenceEvaluatorConfig,
            data=entry_data["config"],
            config=dacite.Config(strict=True),
        )
        entries.append(
            EvalSuiteEntry(
                name=name, config=evaluator_config, config_data=entry_data["config"]
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
    suite = load_suite(path)
    if validate_only:
        return

    prepare_directory(suite.experiment_dir, suite.config_data)
    suite.logging.configure_logging(
        experiment_dir=suite.experiment_dir,
        log_filename="eval_suite_out.log",
        config=suite.config_data,
        resumable=False,
    )

    shared_logger = WandBStepLogger(label="")
    original_get_record = evaluator_module.get_record_to_wandb
    evaluator_module.get_record_to_wandb = lambda label="": shared_logger
    try:
        for entry in suite.inferences:
            logging.info(f"Running evaluator suite entry {entry.name!r}.")
            shared_logger._label = entry.name
            entry.config.configure_logging = lambda log_filename: None  # type: ignore[method-assign]
            prepare_directory(entry.config.experiment_dir, entry.config_data)
            with GlobalTimer(), torch.no_grad():
                run_evaluator_from_config(entry.config)
    finally:
        evaluator_module.get_record_to_wandb = original_get_record


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
