import dataclasses
from typing import Any, List, Mapping, Optional

import torch
from torch import nn

from fme.core.device import get_device
from fme.core.wildcard import apply_by_wildcard, wildcard_match
from fme.fcn_training.registry.registry import ModuleSelector


@dataclasses.dataclass
class FrozenParameterConfig:
    """
    Configuration for freezing parameters in a model.

    Parameter names can include wildcards, e.g. "encoder.*" will select
    all parameters in the encoder, while "encoder.*.bias" will select all
    bias parameters in the encoder. All parameters must be specified
    in either the include or exclude list, or
    an exception will be raised.

    An exception is raised if a parameter is included by both lists.

    Attributes:
        include: list of parameter names to freeze (set requires_grad = False)
        exclude: list of parameter names to ignore
    """

    include: List[str] = dataclasses.field(default_factory=list)
    exclude: List[str] = dataclasses.field(default_factory=list)

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

    def apply(self, model: nn.Module):
        apply_by_wildcard(model, _freeze_weight, self.include, self.exclude)


def _freeze_weight(module: nn.Module, name: str):
    try:
        module.get_parameter(name).requires_grad = False
    except AttributeError:  # non-parameter state
        pass


@dataclasses.dataclass
class ParameterInitializationConfig:
    """
    A class which applies custom initialization to module parameters.

    Assumes the module weights have already been randomly initialized.

    Supports overwriting the weights of the built model with weights from a
    pre-trained model. If the built model has larger weights than the
    pre-trained model, only the initial slice of the weights is overwritten.

    Attributes:
        weight_path: path to a SingleModuleStepper checkpoint
            containing weights to load
        exclude_parameters: list of parameter names to exclude from the loaded
            weights. Used for example to keep the random initialization for
            final layer(s) of a model, and only overwrite the weights for
            earlier layers. Takes values like "decoder.2.weight".
        frozen_parameters: configuration for freezing parameters in the built model
    """

    weights_path: Optional[str] = None
    exclude_parameters: List[str] = dataclasses.field(default_factory=list)
    frozen_parameters: FrozenParameterConfig = dataclasses.field(
        default_factory=lambda: FrozenParameterConfig(exclude=["*"])
    )

    def apply(self, module: nn.Module, init_weights: bool) -> nn.Module:
        """
        Apply the weight initialization to a module.

        Args:
            module: a nn.Module to initialize
            init_weights: whether to initialize the weight values

        Returns:
            a nn.Module with initialization applied
        """
        if init_weights and self.weights_path is not None:
            return _overwrite_weights_from_stepper_path(
                module, self.weights_path, exclude_parameters=self.exclude_parameters
            )
        self.frozen_parameters.apply(module)
        return module


def _strip_leading_module(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
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


def set_nested_parameter(module, param_name, new_param):
    *path, name = param_name.split(".")
    for p in path:
        module = getattr(module, p)
    if not isinstance(new_param, nn.Parameter):
        new_param = nn.Parameter(new_param)
    setattr(module, name, new_param)


def _overwrite_weights_from_stepper_path(
    module: nn.Module, weights_path: str, exclude_parameters: Optional[List[str]] = None
):
    """
    Overwrite the weights in module with the weights in the SingleModuleStepper
    checkpoint at weights_path.
    """
    checkpoint = torch.load(weights_path, map_location=get_device())
    loaded_builder = ModuleSelector.from_state(
        checkpoint["stepper"]["config"]["builder"]
    )
    if "data_shapes" in checkpoint["stepper"]:
        # included for backwards compatibility
        data_shapes = checkpoint["stepper"]["data_shapes"]
        loaded_img_shape = data_shapes[list(data_shapes.keys())[0]][-2:]
    else:
        loaded_img_shape = checkpoint["stepper"]["img_shape"]
    loaded_model = loaded_builder.build(
        n_in_channels=len(checkpoint["stepper"]["config"]["in_names"]),
        n_out_channels=len(checkpoint["stepper"]["config"]["in_names"]),
        img_shape=loaded_img_shape,
    )
    state_dict = _strip_leading_module(checkpoint["stepper"]["module"])
    loaded_model.load_state_dict(state_dict)

    _overwrite_weights(loaded_model, module)

    return module


def _overwrite_weights(
    from_module: torch.nn.Module,
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
        from_module: module containing weights to be copied
        to_module: module whose weights will be overwritten
        exclude_parameters: list of parameter names to exclude from the loaded
            weights. Wildcards can be used, e.g. "decoder.*.weight".
    """
    if exclude_parameters is None:
        exclude_parameters = []
    from_names = set(from_module.state_dict().keys())
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
        from_param = from_module.state_dict()[name]
        to_param = to_module.state_dict()[name]
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
            set_nested_parameter(to_module, name, new_param_data)
