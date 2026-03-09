import dataclasses
import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.training_history import TrainingHistory
from fme.core.weight_ops import overwrite_weights
from fme.core.wildcard import apply_by_exclude, apply_by_include, wildcard_match

Weights = list[Mapping[str, Any]]
StepperWeightsAndHistory = tuple[Weights | None, TrainingHistory]
WeightsAndHistoryLoader = Callable[[str | None], StepperWeightsAndHistory]


@dataclasses.dataclass
class FrozenParameterConfig:
    """
    Configuration for freezing parameters in a model.

    Parameter names are the names used in the module's state_dict. Here
    they can include wildcards, e.g. "encoder.*" will select
    all parameters in the encoder, while "encoder.*.bias" will select all
    bias parameters in the encoder.

    An exception is raised when this configuration is applied (e.g.
    at the start of training) if both lists are non-empty.

    By default no parameters are frozen.

    Parameters:
        include: list of parameter names to freeze (set requires_grad = False),
          if given then all other parameters are left unfrozen
        exclude: list of parameter names to ignore, if given then all other
          parameters are frozen. Cannot be given if include is given.
    """

    include: list[str] = dataclasses.field(default_factory=list)
    exclude: list[str] | None = None

    def __post_init__(self):
        if len(self.include) > 0 and self.exclude is not None:
            warnings.warn(
                "Cannot provide both include and exclude lists "
                "for FrozenParameterConfig, will not be able to apply freezing."
            )  # defer exception to apply, for inference backwards compatibility

    def apply(self, model: nn.Module):
        if len(self.include) > 0 and self.exclude is not None:
            raise ValueError(
                "Cannot provide both include and exclude lists "
                "for FrozenParameterConfig"
            )
        if len(self.include) > 0:
            logging.info("applying freeze to parameters by include")
            apply_by_include(model, _freeze_weight, self.include)
        elif self.exclude is not None:
            logging.info("applying freeze to parameters by exclude")
            apply_by_exclude(model, _freeze_weight, self.exclude)


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
        exclude: list of parameter names to exclude from the loaded
            weights. Used for example to keep the random initialization for
            final layer(s) of a model, and only overwrite the weights for
            earlier layers. Takes values like "decoder.2.weight".
        frozen: configuration for freezing parameters in the built model
    """

    exclude: list[str] = dataclasses.field(default_factory=list)
    frozen: FrozenParameterConfig = dataclasses.field(
        default_factory=FrozenParameterConfig
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
        pretrain_labels: Optional list of label names in the order used by the
            pre-trained checkpoint (first dimension of label_pos_embed). When
            provided and the fine-tune dataset has fewer labels, the learned
            embeddings for the fine-tune labels are copied from the checkpoint.
            Must match the pretrain dataset's label order (e.g. sorted(all_labels)).
    """

    weights_path: str | None = None
    parameters: list[ParameterClassification] = dataclasses.field(default_factory=list)
    pretrain_labels: list[str] | None = None
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
        load_weights_and_history: WeightsAndHistoryLoader,
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


def null_weights_and_history(*_) -> StepperWeightsAndHistory:
    return None, TrainingHistory()


@dataclasses.dataclass
class ParameterInitializer:
    config: ParameterInitializationConfig = dataclasses.field(
        default_factory=ParameterInitializationConfig
    )
    load_weights_and_history: WeightsAndHistoryLoader = dataclasses.field(
        default=null_weights_and_history,
    )

    def __post_init__(self):
        self._base_weights: Weights | None = None
        self._base_weights_for_regularizer: Weights | None = None
        self._training_history: TrainingHistory | None = None

    @property
    def base_weights(self) -> Weights | None:
        if self._base_weights is None and self._training_history is None:
            self._base_weights, self._training_history = self.load_weights_and_history(
                self.config.weights_path
            )
        return self._base_weights

    @property
    def training_history(self) -> TrainingHistory | None:
        if self.base_weights is None and self._training_history is None:
            self._base_weights, self._training_history = self.load_weights_and_history(
                self.config.weights_path
            )
        return self._training_history

    def _filled_parameters(self, n_modules: int) -> list[ParameterClassification]:
        return self.config.parameters + [
            ParameterClassification()
            for _ in range(n_modules - len(self.config.parameters))
        ]

    def _slice_label_pos_embed(
        self,
        from_embed: torch.Tensor,
        to_n_labels: int,
        target_labels: list[str],
    ) -> torch.Tensor:
        """Copy rows from from_embed for each target label using pretrain_labels."""
        pretrain = self.config.pretrain_labels
        if pretrain is None or len(pretrain) != from_embed.shape[0]:
            return from_embed
        out = from_embed.new_zeros(to_n_labels, *from_embed.shape[1:])
        for i, label in enumerate(target_labels):
            if label in pretrain:
                j = pretrain.index(label)
                out[i].copy_(from_embed[j])
        return out

    def apply_weights(
        self,
        modules: list[nn.Module],
        dataset_info: DatasetInfo | None = None,
    ) -> None:
        """
        Apply the weight initialization from a base model to a module.

        Args:
            modules: a list of nn.Modules to initialize
            dataset_info: optional dataset info for the current (e.g. fine-tune)
                run. When provided with pretrain_labels, enables slice-loading
                of label_pos_embed so learned embeddings for the current labels
                are kept from the checkpoint.
        """
        filled_parameters = self._filled_parameters(len(modules))
        if self.base_weights is None:
            return
        pretrain_labels = self.config.pretrain_labels
        target_labels = (
            sorted(dataset_info.all_labels) if dataset_info is not None else None
        )
        modified_weights: Weights | None = None
        if pretrain_labels is not None and target_labels is not None:
            modified_weights = []
        for module, state_dict, classification in zip(
            modules, self.base_weights, filled_parameters
        ):
            state_to_use = dict(state_dict)
            if (
                "label_pos_embed" in state_to_use
                and pretrain_labels is not None
                and target_labels is not None
            ):
                from_embed = state_to_use["label_pos_embed"]
                try:
                    to_param = module.get_parameter("label_pos_embed")
                except AttributeError:
                    to_param = None
                if to_param is not None and from_embed.shape[0] > to_param.shape[0]:
                    sliced = self._slice_label_pos_embed(
                        from_embed, to_param.shape[0], target_labels
                    )
                    state_to_use["label_pos_embed"] = sliced
            if modified_weights is not None:
                modified_weights.append(state_to_use)
            overwrite_weights(
                state_to_use,
                module,
                exclude_parameters=classification.exclude,
            )
        if modified_weights is not None:
            self._base_weights_for_regularizer = modified_weights

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
        base_weights = (
            self._base_weights_for_regularizer
            if self._base_weights_for_regularizer is not None
            else self.base_weights
        )
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
