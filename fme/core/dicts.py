from collections.abc import Mapping
from typing import Any


def to_flat_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """
    Converts any nested dictionaries to a flat version with
    the nested keys joined with a '.', e.g., {a: {b: 1}} ->
    {a.b: 1}.
    """
    new_flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub_d = to_flat_dict(v)
            for sk, sv in sub_d.items():
                new_flat[".".join([k, sk])] = sv
        else:
            new_flat[k] = v

    return new_flat


def to_nested_dict(d: Mapping[str, Any]) -> dict[str, Any]:
    """
    Converts a flat dictionary with '.' joined keys back into
    a nested dictionary, e.g., {a.b: 1} -> {a: {b: 1}}.
    """
    new_config: dict[str, Any] = {}

    for k, v in d.items():
        if "." in k:
            sub_keys = k.split(".")
            sub_d = new_config
            for sk in sub_keys[:-1]:
                sub_d = sub_d.setdefault(sk, {})
            sub_d[sub_keys[-1]] = v
        else:
            new_config[k] = v

    return new_config


def add_names(
    left: Mapping[str, Any], right: Mapping[str, Any], names: list[str]
) -> dict[str, Any]:
    """Add the 'names' from left dict to the right dict and return result.

    Args:
        left: These values will be added to the right dict.
        right: The dict to add the values to.
        names: The names of the keys to add.

    Returns:
        A new dict with the same keys as 'right', but with the values from 'left' added
        for specified names.
    """
    return {k: left[k] + right[k] if k in names else right[k] for k in right}
