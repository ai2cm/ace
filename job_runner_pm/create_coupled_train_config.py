#!/usr/bin/env python
"""Create a coupled training config from pretrained component configs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

TRAINING_STEPPER_FIELDS = {
    "loss",
    "optimize_last_step_only",
    "n_ensemble",
    "parameter_init",
    "train_n_forward_steps",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _get_nested(data: dict[str, Any], keys: list[str]) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _require_mapping(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            raise ValueError(f"Expected mapping before {'.'.join(keys)}")
        node = node.setdefault(key, {})
    if not isinstance(node, dict):
        raise ValueError(f"Expected {'.'.join(keys)} to be a mapping")
    return node


def _merge_missing(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_missing(target[key], value)


def _replace_string_values(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_string_values(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_string_values(item, old, new)
            for key, item in value.items()
        }
    return value


def _component_stepper(config: dict[str, Any], label: str) -> dict[str, Any]:
    stepper = copy.deepcopy(config.get("stepper"))
    if not isinstance(stepper, dict):
        raise ValueError(f"{label} config is missing a top-level stepper mapping")
    for key in TRAINING_STEPPER_FIELDS:
        stepper.pop(key, None)
    return stepper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atmos-config", required=True, type=Path)
    parser.add_argument("--ocean-config", required=True, type=Path)
    parser.add_argument("--template-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    atmos = _replace_string_values(
        _load_yaml(args.atmos_config), "statsdata", "atmos_stats"
    )
    ocean = _replace_string_values(
        _load_yaml(args.ocean_config), "statsdata", "ocean_stats"
    )
    output = _load_yaml(args.template_config)

    ocean_stepper = _component_stepper(ocean, "ocean")
    atmos_stepper = _component_stepper(atmos, "atmosphere")

    ocean_target = _require_mapping(output, ["stepper", "ocean", "stepper"])
    atmos_target = _require_mapping(output, ["stepper", "atmosphere", "stepper"])
    _merge_missing(ocean_target, ocean_stepper)
    _merge_missing(atmos_target, atmos_stepper)

    sea_ice_fraction_name = _get_nested(
        ocean,
        [
            "stepper",
            "step",
            "config",
            "corrector",
            "config",
            "sea_ice_fraction_correction",
            "sea_ice_fraction_name",
        ],
    )
    if sea_ice_fraction_name is None:
        raise ValueError(
            "Failed to extract ocean sea_ice_fraction_name from "
            "stepper.step.config.corrector.config.sea_ice_fraction_correction"
        )
    _require_mapping(output, ["stepper", "ocean_fraction_prediction"])[
        "sea_ice_fraction_name"
    ] = sea_ice_fraction_name

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(args.output_config, output)


if __name__ == "__main__":
    main()
