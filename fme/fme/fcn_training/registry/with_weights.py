import dataclasses
import re
from typing import Any, List, Mapping, Optional, Tuple

import torch
from torch import nn

from fme.core.device import get_device
from fme.fcn_training.registry.registry import ModuleConfig, ModuleSelector, register


@dataclasses.dataclass
class FrozenParameterConfig:
    """
    Configuration for freezing parameters in a model.

    Parameter names can include wildcards, e.g. "encoder.*" will select
    all parameters in the encoder, while "encoder.*.bias" will select all
    bias parameters in the encoder. All parameters must be specified
    in either the frozen_parameters or unfrozen_parameters list, or
    an exception will be raised.

    An exception is raised if a parameter is included by both lists.

    Attributes:
        include: list of parameter names to freeze
        exclude: list of parameter names to unfreeze, taking
            priority over frozen_parameters
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
        missing_parameters = []
        for name in model.state_dict().keys():
            if any(wildcard_match(pattern, name) for pattern in self.include):
                if any(wildcard_match(pattern, name) for pattern in self.exclude):
                    raise ValueError(
                        f"Parameter {name} is included in both include "
                        f"{self.include} and exclude {self.exclude}"
                    )
                model.get_parameter(name).requires_grad = False
            elif any(wildcard_match(pattern, name) for pattern in self.exclude):
                model.get_parameter(name).requires_grad = True
            else:
                missing_parameters.append(name)
        if len(missing_parameters) > 0:
            raise ValueError(
                f"Model has parameters {missing_parameters} which are not "
                f"specified in either include {self.include} "
                f"or exclude {self.exclude}"
            )
        return model


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


@register("BuilderWithWeights")
@dataclasses.dataclass
class BuilderWithWeights(ModuleConfig):
    """
    A builder which initializes a model from another builder and loads weights
    from disk, and then initializes each parameter in the built model with
    the corresponding parameter in the loaded model.

    When the built model has a larger number of parameters than the loaded model,
    only the initial slice is initialized. For example, if the loaded model has
    a parameter `a` of shape [10, 10], and the built model has a parameter `a`
    of shape [20, 10], then only the first 10 rows of `a` will be initialized
    from the weights on disk.

    This is particularly helpful for fine-tuning a model, as it allows us to
    initialize a model with weights from a pre-trained model and then train
    the model on a new dataset potentially with new weights. For example, these
    weights could correspond to new inputs or output variables, or
    increased model resolution.

    Attributes:
        module: configuration to build the model
        weights_path: path to a SingleModuleStepper checkpoint
            containing weights to load
        allow_missing_parameters: if True, allow the built model to have new
            parameters not defined in the loaded model. The built model is still
            not allowed to be missing parameters defined in the loaded model.
        exclude_parameters: list of parameter names to exclude from the loaded
            weights. Used for example to keep the random initialization for
            final layer(s) of a model, and only overwrite the weights for
            earlier layers. Takes values like "decoder.2.weight".
        frozen_parameters: configuration for freezing parameters in the built model
    """

    module: ModuleSelector
    weights_path: str
    allow_missing_parameters: bool = False
    exclude_parameters: List[str] = dataclasses.field(default_factory=list)
    frozen_parameters: FrozenParameterConfig = dataclasses.field(
        default_factory=lambda: FrozenParameterConfig(exclude=["*"])
    )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and lon

        Returns:
            a nn.Module
        """
        model = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )
        checkpoint = torch.load(self.weights_path, map_location=get_device())
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
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=loaded_img_shape,
        )
        state_dict = _strip_leading_module(checkpoint["stepper"]["module"])
        loaded_model.load_state_dict(state_dict)

        _overwrite_weights(
            loaded_model, model, exclude_parameters=self.exclude_parameters
        )

        self.frozen_parameters.apply(model)

        return model

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleConfig":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        state = dict(state)  # make a copy so we can modify it
        if "builder" in state and "module" not in state:
            module_selector = ModuleSelector.from_state(state.pop("builder"))
        else:
            module_selector = ModuleSelector.from_state(state.pop("module"))
        if "frozen_parameters" in state:
            state["frozen_parameters"] = FrozenParameterConfig(
                **state.pop("frozen_parameters")
            )
        return cls(
            module=module_selector,
            **state,
        )


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
