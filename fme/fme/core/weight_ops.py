from typing import Any, List, Mapping, Optional

import torch
from torch import nn

from .wildcard import wildcard_match


def strip_leading_module(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Remove the leading "module." from the keys of a state dict.

    This is necessary because SingleModuleStepper wraps the module in either
    a DistributedDataParallel layer or DummyWrapper layer, which adds a leading
    "module." to the keys of the state dict.
    """
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def overwrite_weights(
    from_state: Mapping[str, Any],
    to_module: torch.nn.Module,
    exclude_parameters: Optional[List[str]] = None,
):
    """
    Overwrite the weights in to_module with the weights in from_module.

    When an axis is larger in to_module than in from_module, only the initial
    slice is overwritten. For example, if the from module has a parameter `a`
    of shape [10, 10], and the to module has a parameter `a` of shape [20, 10],
    then only the first 10 rows of `a` will be overwritten.

    If an axis is larger in from_module than in to_module, an exception is raised.

    Args:
        from_state: module state dict containing weights to be copied
        to_module: module whose weights will be overwritten
        exclude_parameters: list of parameter names to exclude from the loaded
            weights. Wildcards can be used, e.g. "decoder.*.weight".
    """
    if exclude_parameters is None:
        exclude_parameters = []
    from_names = set(from_state.keys())
    to_names = set(to_module.state_dict().keys())
    if not from_names.issubset(to_names):
        missing_parameters = from_names - to_names
        raise ValueError(
            f"Dest module is missing parameters {missing_parameters}, "
            "which is not allowed"
        )
    for name in from_names:
        if any(wildcard_match(pattern, name) for pattern in exclude_parameters):
            continue
        from_param = from_state[name]
        try:
            overwrite_weight_initial_slice(to_module, name, from_param)
        except AttributeError:  # if state is not a parameter
            pass


def overwrite_weight_initial_slice(module, name, from_param):
    """
    Overwrite the initial slice of a parameter in module with from_param.

    When an axis is larger in to_module than in from_module, only the initial
    slice is overwritten. For example, if the from module has a parameter `a`
    of shape [10, 10], and the to module has a parameter `a` of shape [20, 10],
    then only the first 10 rows of `a` will be overwritten.

    If an axis is larger in from_module than in to_module, an exception is raised.

    Args:
        module: module whose parameter will be overwritten
        name: name of the parameter to be overwritten
        from_param: parameter to overwrite with
    """
    to_param = module.get_parameter(name)
    if len(from_param.shape) != len(to_param.shape):
        raise ValueError(
            f"Dest parameter {name} has "
            f"{len(to_param.shape.shape)} "
            "dimensions which needs to be equal to the loaded "
            f"parameter dimension {len(from_param.shape)}"
        )
    for from_size, to_size in zip(from_param.shape, to_param.shape):
        if from_size > to_size:
            raise ValueError(
                f"Dest parameter has size {to_size} along one of its "
                "dimensions which needs to be greater than loaded "
                f"parameter size {from_size}"
            )
    slices = tuple(slice(0, size) for size in from_param.shape)
    with torch.no_grad():
        new_param_data = to_param.data.clone()
        new_param_data[slices] = from_param.data
        _set_nested_parameter(module, name, new_param_data)


def _set_nested_parameter(module, param_name, new_param):
    *path, name = param_name.split(".")
    for p in path:
        module = getattr(module, p)
    if not isinstance(new_param, nn.Parameter):
        new_param = nn.Parameter(new_param)
    setattr(module, name, new_param)
