import logging
import re
from collections.abc import Callable

from torch import nn


def wildcard_match(pattern: str, name: str) -> bool:
    """
    Check if a name matches a wildcard pattern.

    A wildcard pattern can include "*" to match any number of characters.
    """
    # use regex
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = f"^{regex_pattern}$"
    return bool(re.match(regex_pattern, name))


def _get_matching_pattern(patterns: list[str], name: str) -> str | None:
    """
    Get the first pattern that matches a name.
    """
    for pattern in patterns:
        # use regex
        if wildcard_match(pattern, name):
            return pattern
    return None


def _drop_from_set(set: set[str], item: str):
    if item in set:
        set.remove(item)


class UnusedRuleError(ValueError):
    pass


def apply_by_include(
    model: nn.Module,
    func: Callable[[nn.Module, str], None],
    include: list[str],
):
    """
    Apply a function to parameters in a model by wildcard rules.

    All parameters not included are excluded. Raises if an include rule
    is unmatched.

    Parameters:
        model: The model to apply the function to.
        func: The function to apply to the parameters.
        include: A list of wildcard patterns to include.
    """
    remaining_includes = set(include)
    _drop_from_set(remaining_includes, "*")
    applied_names = set()
    for name in model.state_dict().keys():
        matching_include = _get_matching_pattern(include, name)
        if matching_include is not None:
            _drop_from_set(remaining_includes, matching_include)
            func(model, name)
            applied_names.add(name)
    if len(remaining_includes) > 0:
        raise UnusedRuleError(
            f"Model has include rules that do not match any parameters, "
            f"include rules: {remaining_includes}, "
            f"parameters: {list(model.state_dict().keys())}"
        )
    logging.info(f"Applied function to parameters: {applied_names}")
    logging.info(
        f"Skipped parameters: {set(model.state_dict().keys()) - applied_names}"
    )
    return model


def apply_by_exclude(
    model: nn.Module,
    func: Callable[[nn.Module, str], None],
    exclude: list[str],
):
    """
    Apply a function to parameters in a model by wildcard rules.

    All parameters not excluded are included. Raises if an exclude rule
    is unmatched.

    Parameters:
        model: The model to apply the function to.
        func: The function to apply to the parameters.
        exclude: A list of wildcard patterns to exclude.
    """
    remaining_excludes = set(exclude)
    _drop_from_set(remaining_excludes, "*")
    applied_names = set()
    for name in model.state_dict().keys():
        matching_exclude = _get_matching_pattern(exclude, name)
        if matching_exclude is not None:
            _drop_from_set(remaining_excludes, matching_exclude)
        else:
            func(model, name)
            applied_names.add(name)
    if len(remaining_excludes) > 0:
        raise UnusedRuleError(
            f"Model has exclude rules that do not match any parameters, "
            f"exclude rules: {remaining_excludes}, "
            f"parameters: {list(model.state_dict().keys())}"
        )
    logging.info(f"Applied function to parameters: {applied_names}")
    logging.info(
        f"Skipped parameters: {set(model.state_dict().keys()) - applied_names}"
    )
    return model
