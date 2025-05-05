import argparse
import os
from collections.abc import Sequence

import yaml

from .config import update_dict_with_dotlist


def prepare_config(path: str, override: Sequence[str] | None = None) -> dict:
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    data = update_dict_with_dotlist(data, override)
    return data


def prepare_directory(path: str, config_data: dict):
    """Create experiment directory and dump config_data to it."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


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
