import dataclasses
from typing import Any, Callable, List, Mapping, Optional

import torch
from torch import nn

from fme.core.device import get_device
from fme.core.weight_ops import overwrite_weights
from fme.core.wildcard import apply_by_wildcard, wildcard_match


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

    Parameters:
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


RegularizerFunction = Callable[[], torch.Tensor]


@dataclasses.dataclass
class ParameterClassification:
    """
    Specifies whether parameters are excluded from initialization or frozen.

    Parameters:
        exclude_parameters: list of parameter names to exclude from the loaded
            weights. Used for example to keep the random initialization for
            final layer(s) of a model, and only overwrite the weights for
            earlier layers. Takes values like "decoder.2.weight".
        frozen_parameters: configuration for freezing parameters in the built model
    """

    exclude: List[str] = dataclasses.field(default_factory=list)
    frozen: FrozenParameterConfig = dataclasses.field(
        default_factory=lambda: FrozenParameterConfig(exclude=["*"])
    )


@dataclasses.dataclass
class ParameterInitializationConfig:
    """
    A class which applies custom initialization to module parameters.

    Assumes the module weights have already been randomly initialized.

    Supports overwriting the weights of the built model with weights from a
    pre-trained model. If the built model has larger weights than the
    pre-trained model, only the initial slice of the weights is overwritten.

    Parameters:
        weight_path: path to a Stepper checkpoint
            containing weights to load
        parameters: list of ParameterClassification objects, each specifying
            whether parameters are excluded from initialization or frozen.
            By default modules are unfrozen and all parameters are included.
            Must be provided in the same order as provided by the stepper's
            `.modules` attribute.
        alpha: L2 regularization coefficient keeping initialized weights
            close to their intiial values
        beta: L2 regularization coefficient keeping uninitialized weights
            close to zero
        exclude_parameters: deprecated, kept for backwards compatibility
        frozen_parameters: deprecated, kept for backwards compatibility
    """

    weights_path: Optional[str] = None
    parameters: List[ParameterClassification] = dataclasses.field(default_factory=list)
    alpha: float = 0.0
    beta: float = 0.0
    exclude_parameters: Optional[List[str]] = None
    frozen_parameters: Optional[FrozenParameterConfig] = None

    def __post_init__(self):
        if self.exclude_parameters is not None or self.frozen_parameters is not None:
            if len(self.parameters) > 0:
                raise ValueError(
                    "exclude_parameters and frozen_parameters are deprecated, "
                    "do not provide both parameters and exclude_parameters or "
                    "frozen_parameters"
                )
            exclude = self.exclude_parameters or []
            frozen = self.frozen_parameters or FrozenParameterConfig(exclude=["*"])
            self.parameters = [ParameterClassification(exclude=exclude, frozen=frozen)]

    def _filled_parameters(self, n_modules: int) -> List[ParameterClassification]:
        return self.parameters + [
            ParameterClassification() for _ in range(n_modules - len(self.parameters))
        ]

    def apply(
        self,
        modules: List[nn.Module],
        init_weights: bool,
        load_weights: Callable[[str], List[Mapping[str, Any]]],
    ) -> RegularizerFunction:
        """
        Apply the weight initialization to a module.

        Args:
            modules: a list of nn.Modules to initialize
            init_weights: whether to initialize the weight values
            load_weights: a function which loads model weights from a path,
                specifically the configured weights_path

        Returns:
            a list of nn.Modules with initialization applied
            a function which returns the regularization loss term
        """
        filled_parameters = self._filled_parameters(len(modules))
        if init_weights and self.weights_path is not None:
            loaded_state_dicts = self.get_base_weights(load_weights)
            if loaded_state_dicts is not None:
                for module, state_dict, classification in zip(
                    modules, loaded_state_dicts, filled_parameters
                ):
                    overwrite_weights(
                        state_dict,
                        module,
                        exclude_parameters=classification.exclude,
                    )
        else:
            loaded_state_dicts = None
        for module, classification in zip(modules, filled_parameters):
            classification.frozen.apply(module)
        device = get_device()
        if loaded_state_dicts is None or (self.alpha == 0 and self.beta == 0):

            def regularizer():
                return torch.tensor(0.0, device=device)

            return regularizer

        else:
            for classification, state_dict in zip(
                filled_parameters, loaded_state_dicts
            ):
                state_dict = {
                    name: value.to(device) for name, value in state_dict.items()
                }
                from_names = set(state_dict.keys())
                to_names = set(module.state_dict().keys())
                if not from_names.issubset(to_names):
                    missing_parameters = from_names - to_names
                    raise ValueError(
                        f"Dest module is missing parameters {missing_parameters}, "
                        "which is not allowed"
                    )

            non_optional_state_dicts = loaded_state_dicts

            def regularizer():
                loss = torch.tensor(0.0, device=device)
                for module, state_dict, classification in zip(
                    modules, non_optional_state_dicts, filled_parameters
                ):
                    for name in state_dict.keys():
                        try:
                            param = module.get_parameter(name)
                        except AttributeError:  # non-trainable state data
                            continue
                        if any(
                            wildcard_match(pattern, name)
                            for pattern in classification.exclude
                        ):
                            loss += (
                                self.beta
                                / 2
                                * torch.linalg.norm(param.flatten(), ord=2)
                            )
                        else:
                            loss += (
                                self.alpha
                                / 2
                                * torch.linalg.norm(
                                    (param - state_dict[name]).flatten(),
                                    ord=2,
                                )
                            )

                return loss

        return regularizer

    def get_base_weights(
        self, load_weights: Callable[[str], List[torch.nn.Module]]
    ) -> Optional[List[Mapping[str, Any]]]:
        """
        If a weights_path is provided, return the model base weights used for
        initialization.

        Args:
            load_weights: a function which loads model weights from a path,
                specifically the configured weights_path

        Returns:
            a list of state_dicts for each module in the stepper
        """
        if self.weights_path is not None:
            return load_weights(self.weights_path)
        else:
            return None
