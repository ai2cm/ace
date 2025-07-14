import re
from collections.abc import Callable

from torch import nn


def wildcard_match(pattern: str, name: str) -> bool:
    """
    Check if a name matches a wildcard pattern.

    A wildcard pattern can include "*" to match any number of characters.
    """
    # use regex
    pattern = pattern.replace(".", r"\.")
    pattern = pattern.replace("*", ".*")
    pattern = f"^{pattern}$"
    return bool(re.match(pattern, name))


def apply_by_wildcard(
    model: nn.Module,
    func: Callable[[nn.Module, str], None],
    include: list[str],
    exclude: list[str],
):
    missing_parameters = []
    for name in model.state_dict().keys():
        if any(wildcard_match(pattern, name) for pattern in include):
            if any(wildcard_match(pattern, name) for pattern in exclude):
                raise ValueError(
                    f"Parameter {name} is included in both include "
                    f"{include} and exclude {exclude}"
                )
            func(model, name)
        elif not any(wildcard_match(pattern, name) for pattern in exclude):
            missing_parameters.append(name)
    if len(missing_parameters) > 0:
        raise ValueError(
            f"Model has parameters {missing_parameters} which are not "
            f"specified in either include {include} "
            f"or exclude {exclude}"
        )
    return model
