import argparse
import os
import shutil
from collections.abc import Sequence

import yaml

from .config import update_dict_with_dotlist


def prepare_config(path: str, override: Sequence[str] | None = None) -> dict:
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    data = update_dict_with_dotlist(data, override)
    return data


def prepare_directory(
    path: str, config_data: dict, resume_results_path: str | None = None
) -> str | None:
    """Create experiment directory and dump config_data to it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    if resume_results_path is not None and not os.path.isdir(
        os.path.join(path, "training_checkpoints")
    ):
        if not os.path.isdir(resume_results_path):
            raise ValueError(f"The directory {resume_results_path} does not exist.")
        # recursively copy all files in resume_results_path to path
        shutil.copytree(resume_results_path, path, dirs_exist_ok=True)
    else:
        # either not given or ignored because we already resumed once before
        resume_results_path = None
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    return resume_results_path


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
