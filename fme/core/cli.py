import argparse
import dataclasses
import os
import shutil
from collections.abc import Sequence

import fsspec
import yaml

from .config import update_dict_with_dotlist


@dataclasses.dataclass
class ResumeResultsConfig:
    """Configuration for resuming a previously stopped or finished job.

    Typically only useful for training jobs which have already finished (e.g., to train
    for a larger value of max_epochs than originally configured) or which were stopped
    (e.g., to resume training on different hardware or to change data loader settings
    such as number of data workers).

    WARNING: We typically don't guarantee backwards compatibility for training, so this
    may not work well when resuming old experiments.

    Arguments:
        existing_dir: Directory with existing results to resume from.
        resume_wandb: If true, log to the same WandB job as given in the
            wandb_job_id file in existing_dir, if any.
    """

    existing_dir: str
    resume_wandb: bool = False


def prepare_config(path: str, override: Sequence[str] | None = None) -> dict:
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    data = update_dict_with_dotlist(data, override)
    return data


def prepare_directory(
    path: str, config_data: dict, resume_results: ResumeResultsConfig | None = None
) -> ResumeResultsConfig | None:
    """Create experiment directory and dump config_data to it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    if resume_results is not None and not os.path.isdir(
        os.path.join(path, "training_checkpoints")
    ):
        if not os.path.isdir(resume_results.existing_dir):
            raise ValueError(
                f"The directory {resume_results.existing_dir} does not exist."
            )
        # recursively copy all files in resume_results_path to path
        shutil.copytree(resume_results.existing_dir, path, dirs_exist_ok=True)
        wandb_run_id_path = os.path.join(path, "wandb_run_id")
        if not resume_results.resume_wandb and os.path.exists(wandb_run_id_path):
            os.remove(wandb_run_id_path)
    else:
        # either not given or ignored because we already resumed once before
        resume_results = None
    with fsspec.open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    return resume_results


def get_parser():
    """Standard arg parser for ACE entrypoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--override",
        nargs="*",
        help="A dotlist of key=value pairs to override the config. "
        "For example, --override a.b=1 c=2, where a dot indicates nesting.",
    )
    return parser
