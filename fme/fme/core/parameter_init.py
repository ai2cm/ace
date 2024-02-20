import dataclasses
from typing import Any, List, Mapping, Optional

import torch
from torch import nn

from fme.core.device import get_device
from fme.core.wildcard import apply_by_wildcard, wildcard_match

from .weight_ops import overwrite_weights, strip_leading_module


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
            loaded_state_dict = self.get_base_weights()
            if loaded_state_dict is not None:
                overwrite_weights(
                    loaded_state_dict,
                    module,
                    exclude_parameters=self.exclude_parameters,
                )
        self.frozen_parameters.apply(module)
        return module

    def get_base_weights(self) -> Optional[Mapping[str, Any]]:
        """
        If a weights_path is provided, return the model base weights used for
        initialization.
        """
        if self.weights_path is not None:
            checkpoint = torch.load(self.weights_path, map_location=get_device())
            return strip_leading_module(checkpoint["stepper"]["module"])
        else:
            return None
