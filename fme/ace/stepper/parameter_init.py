import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from fme.core.device import get_device
from fme.core.training_history import TrainingHistory
from fme.core.weight_ops import overwrite_weights
from fme.core.wildcard import apply_by_wildcard, wildcard_match

Weights = list[Mapping[str, Any]]
StepperWeightsAndHistory = tuple[Weights, TrainingHistory]


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

    exclude: list[str] = dataclasses.field(default_factory=list)
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

    weights_path: str | None = None
    parameters: list[ParameterClassification] = dataclasses.field(default_factory=list)
    alpha: float = 0.0
    beta: float = 0.0
    exclude_parameters: list[str] | None = None
    frozen_parameters: FrozenParameterConfig | None = None

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
        self.exclude_parameters = None
        self.frozen_parameters = None

    def build(
        self,
        load_weights_and_history: Callable[[str], StepperWeightsAndHistory],
    ) -> "ParameterInitializer":
        """
        Build a ParameterInitializer instance with the current configuration.

        Args:
            load_weights_and_history: a function which loads weights and training
                history from a path, specifically the configured weights_path.
        """
        return ParameterInitializer(
            config=self, load_weights_and_history=load_weights_and_history
        )


@dataclasses.dataclass
class ParameterInitializer:
    config: ParameterInitializationConfig = dataclasses.field(
        default_factory=ParameterInitializationConfig
    )
    load_weights_and_history: Callable[[str], StepperWeightsAndHistory] = (
        dataclasses.field(default=lambda _: ([], TrainingHistory()))
    )

    def __post_init__(self):
        self._base_weights: Weights | None = None
        self._training_history: TrainingHistory | None = None

    @property
    def base_weights(self) -> Weights | None:
        if self.config.weights_path is not None and self._base_weights is None:
            self._base_weights, self._training_history = self.load_weights_and_history(
                self.config.weights_path
            )
        return self._base_weights

    @property
    def training_history(self) -> TrainingHistory | None:
        if self.config.weights_path is not None and self._training_history is None:
            self._base_weights, self._training_history = self.load_weights_and_history(
                self.config.weights_path
            )
        return self._training_history

    def _filled_parameters(self, n_modules: int) -> list[ParameterClassification]:
        return self.config.parameters + [
            ParameterClassification()
            for _ in range(n_modules - len(self.config.parameters))
        ]

    def apply_weights(
        self,
        modules: list[nn.Module],
    ) -> None:
        """
        Apply the weight initialization from a base model to a module.

        Args:
            modules: a list of nn.Modules to initialize
        """
        filled_parameters = self._filled_parameters(len(modules))
        if self.config.weights_path is not None:
            if self.base_weights is not None:
                for module, state_dict, classification in zip(
                    modules, self.base_weights, filled_parameters
                ):
                    overwrite_weights(
                        state_dict,
                        module,
                        exclude_parameters=classification.exclude,
                    )

    def freeze_weights(self, modules: list[nn.Module]):
        """
        Freeze the weights of the modules.

        Note this must be called before wrapping the modules in a DDP layer,
        otherwise the DistributedDataParallel will expect frozen
        weights to have gradients, and will raise an exception.

        Args:
            modules: a list of nn.Modules to freeze if configured to do so.
        """
        filled_parameters = self._filled_parameters(len(modules))
        for module, classification in zip(modules, filled_parameters):
            classification.frozen.apply(module)

    def get_l2_sp_tuning_regularizer(
        self,
        modules: list[nn.Module],
    ) -> RegularizerFunction:
        """Get L2-SP loss regularizer function for the parameters of the modules.
        The regularizer function computes the L2 regularization loss based on
        the base weights and the current weights of the modules.

        If the base weights are set, it computes the L2 regularization loss
        for each module based on the difference between the current weights
        and the base weights, and the L2 norm of the weights that are not
        initialized (i.e., those that are excluded from initialization).

        Args:
            modules: a list of nn.Modules to compute the regularization for

        Returns:
            A function that returns a tensor representing the regularization loss.
        """
        device = get_device()
        filled_parameters = self._filled_parameters(len(modules))
        base_weights = self.base_weights
        if base_weights is not None and (
            self.config.alpha != 0 or self.config.beta != 0
        ):
            for module, state_dict in zip(modules, base_weights):
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

            def regularizer():
                loss = torch.tensor(0.0, device=device)
                for module, state_dict, classification in zip(
                    modules, base_weights, filled_parameters
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
                                self.config.beta
                                / 2
                                * torch.linalg.norm(param.flatten(), ord=2)
                            )
                        else:
                            loss += (
                                self.config.alpha
                                / 2
                                * torch.linalg.norm(
                                    (param - state_dict[name]).flatten(),
                                    ord=2,
                                )
                            )

                return loss

        else:

            def regularizer():
                return torch.tensor(0.0, device=device)

            return regularizer

        return regularizer
