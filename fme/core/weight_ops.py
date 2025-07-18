import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .wildcard import apply_by_wildcard, wildcard_match


@dataclasses.dataclass
class CopyWeightsConfig:
    """
    Configuration for copying weights from a base model to a target model.

    Used during training to overwrite weights after every batch of data,
    to have the effect of "freezing" the overwritten weights. When the
    target parameters have longer dimensions than the base model, only
    the initial slice is overwritten.

    This is used to achieve an effect of freezing model parameters that
    can freeze a subset of each weight that comes from a smaller base weight.
    This is less efficient than true parameter freezing, but layer
    freezing is all-or-nothing for each parameter.

    All parameters must be covered by either the include or exclude list,
    but not both.

    Parameters:
        include: list of wildcard patterns to overwrite
        exclude: list of wildcard patterns to exclude from overwriting
    """

    include: list[str] = dataclasses.field(default_factory=list)
    exclude: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        for pattern in self.include:
            if any(wildcard_match(pattern, exclude) for exclude in self.exclude):
                raise ValueError(
                    f"Parameter {pattern} is included in both include "
                    f"{self.include} and exclude {self.exclude}"
                )
        for pattern in self.exclude:
            if any(wildcard_match(pattern, include) for include in self.include):
                raise ValueError(
                    f"Parameter {pattern} is included in both include "
                    f"{self.include} and exclude {self.exclude}"
                )

    @torch.no_grad()
    def apply(self, weights: list[Mapping[str, Any]], modules: list[nn.Module]):
        """
        Apply base weights to modules according to the include/exclude lists
        of this instance.

        In order to "freeze" the weights during training, this must be called after
        each time the weights are updated in the training loop.

        Args:
            weights: list of base weights to apply
            modules: list of modules to apply the weights to
        """
        if len(modules) > 1:
            # We can support multiple modules by having this configuration take a list
            # of include/exclude for each module. Not implemented right now because it
            # is not needed, and would make the configuration more confusing for the
            # single-module case (especially when it's only ever single-module).
            raise NotImplementedError("only one module currently supported")
        if len(modules) != len(weights):
            raise ValueError("number of modules and weights must match")
        for module, weight in zip(modules, weights):

            def func(module, name):
                overwrite_weight_initial_slice(module, name, weight[name])

            apply_by_wildcard(module, func, self.include, self.exclude)
        return module


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
    exclude_parameters: list[str] | None = None,
):
    """
    Overwrite the weights in to_module with the weights in from_state.

    When an axis is larger in to_module than in from_state, only the initial
    slice is overwritten. For example, if the from module has a parameter `a`
    of shape [10, 10], and the to module has a parameter `a` of shape [20, 10],
    then only the first 10 rows of `a` will be overwritten.

    If an axis is larger in from_state than in to_module, an exception is raised.

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

    When an axis is larger in the module's param than in from_param,
    only the initial slice is overwritten. For example, if the from module
    has a parameter `a` of shape [10, 10], and the to module has a parameter
    `a` of shape [20, 10], then only the first 10 rows of `a` will be overwritten.

    If an axis is larger in from_param, an exception is raised.

    Args:
        module: module whose parameter will be overwritten
        name: name of the parameter to be overwritten
        from_param: parameter to overwrite with
    """
    try:
        to_param = module.get_parameter(name)
    except AttributeError:
        if name == "device_buffer" or name == "module.device_buffer":
            return  # ignore device buffer, used for GPU operations
        raise
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
    getattr(module, name)[:] = new_param
