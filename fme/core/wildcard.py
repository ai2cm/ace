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


class MissingParameterError(ValueError):
    pass


class ConflictingRuleError(ValueError):
    pass


def apply_by_wildcard(
    model: nn.Module,
    func: Callable[[nn.Module, str], None],
    include: list[str],
    exclude: list[str],
    raise_if_unused: bool = False,
):
    """
    Apply a function to parameters in a model by wildcard rules.

    Parameters:
        model: The model to apply the function to.
        func: The function to apply to the parameters.
        include: A list of wildcard patterns to include.
        exclude: A list of wildcard patterns to exclude.
        raise_if_unused: Whether to raise an error if there are include or
            exclude rules that do not match any parameters. The "*" pattern
            is exempted and will not be checked, as it is often used to
            match "all parameters" in an empty list.
    """
    missing_parameters = []
    remaining_includes = set(include)
    remaining_excludes = set(exclude)
    _drop_from_set(remaining_includes, "*")
    _drop_from_set(remaining_excludes, "*")
    for name in model.state_dict().keys():
        matching_include = _get_matching_pattern(include, name)
        if matching_include is not None:
            _drop_from_set(remaining_includes, matching_include)
            matching_exclude = _get_matching_pattern(exclude, name)
            if matching_exclude is not None:
                raise ConflictingRuleError(
                    f"Parameter {name} is matched by both include rule "
                    f"{matching_include} and exclude rule {matching_exclude}"
                )
            func(model, name)
        else:
            matching_exclude = _get_matching_pattern(exclude, name)
            if matching_exclude is not None:
                _drop_from_set(remaining_excludes, matching_exclude)
            else:
                missing_parameters.append(name)
    if len(missing_parameters) > 0:
        raise MissingParameterError(
            f"Model has parameters {missing_parameters} which are not "
            f"specified in either include {include} "
            f"or exclude {exclude}"
        )
    if raise_if_unused and (len(remaining_includes) > 0 or len(remaining_excludes) > 0):
        raise UnusedRuleError(
            f"Model has include and/or exclude rules that do not match any parameters, "
            f"include rules: {remaining_includes}, exclude rules: {remaining_excludes}"
        )
    return model
