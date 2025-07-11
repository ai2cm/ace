import dataclasses
from collections.abc import Callable
from copy import copy
from typing import Any, TypeVar

import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.multi_call import MultiCall, MultiCallConfig, StepMethod
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean import OceanConfig
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


def replace_multi_call(
    selector: StepSelector, multi_call: MultiCallConfig | None, state: dict[str, Any]
) -> tuple[StepSelector, dict[str, Any]]:
    """
    Replace the multi-call configuration in a StepSelector and ensure the
    associated state can be loaded as a multi-call step.

    A value of `None` for `multi_call` will remove the multi-call configuration.

    If the selected type supports it, the multi-call configuration will be
    updated in place. Otherwise, it will be wrapped in the multi_call step
    configuration with the given multi_call config or None.

    Args:
        selector: StepSelector to replace the multi-call configuration of. If
            the StepSelector does not have "multi_call" type, then the step
            will be wrapped in a "multi_call" type StepSelector.
        multi_call: MultiCallConfig for returned StepSelector.
        state: state dictionary associated with the loaded step.

    Returns:
        A StepSelector with the specified MultiCallConfig and the state
        dictionary updated to ensure consistency with that of a serialized
        multi-call step.
    """
    state_copy = state.copy()
    if selector.type == "multi_call":
        wrapped_selector_dict: dict[str, Any] = selector.config["wrapped_step"]
        include_multi_call_in_loss = selector.config.get(
            "include_multi_call_in_loss", True
        )
        new_state = state_copy
    else:
        wrapped_selector_dict = dataclasses.asdict(selector)
        include_multi_call_in_loss = True
        new_state = {"wrapped_step": state_copy}
    if multi_call is None:
        include_multi_call_in_loss = False
    new_selector = StepSelector(
        type="multi_call",
        config={
            "wrapped_step": wrapped_selector_dict,
            "config": dataclasses.asdict(multi_call) if multi_call else None,
            "include_multi_call_in_loss": include_multi_call_in_loss,
        },
    )
    return new_selector, new_state


@StepSelector.register("multi_call")
@dataclasses.dataclass
class MultiCallStepConfig(StepConfigABC):
    """
    Configuration for a multi-call step.

    Parameters:
        wrapped_step: The step to wrap.
        config: The multi-call configuration.
        include_multi_call_in_loss: Whether to include multi-call diagnostics in the
            loss.
    """

    wrapped_step: StepSelector
    config: MultiCallConfig | None = None
    include_multi_call_in_loss: bool = True

    def __post_init__(self):
        if self.config is not None:
            self.config.validate(
                self.wrapped_step.input_names, self.wrapped_step.output_names
            )
        if self.config is None and self.include_multi_call_in_loss:
            raise ValueError("include_multi_call_in_loss is True, but config is None")

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "MultiCallStep":
        wrapped = self.wrapped_step.get_step(dataset_info, init_weights)
        if self.config is not None:
            self.config.validate(wrapped.input_names, wrapped.output_names)
        return MultiCallStep(
            wrapped_step=wrapped,
            config=self,
        )

    def build(
        self,
        step_method: StepMethod,
    ) -> "MultiCall | None":
        if self.config is None:
            return None
        else:
            return self.config.build(step_method)

    def extend_normalizer_with_multi_call_outputs(
        self, normalizer: StandardNormalizer
    ) -> StandardNormalizer:
        """
        Extend the normalizer by setting multi-call output names to use the same
        normalization as their base counterparts.
        """
        if self.config is None:
            return normalizer
        else:
            return _extend_normalizer_with_multi_call_outputs(self.config, normalizer)

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        """
        Get the loss normalizer for the multi-call step.

        Normalizer will use statistics from multi-call variables in the stats
        dataset, meaning the normalization for multi-call output versions will be
        different from the normalization for the base variables.

        Args:
            extra_names: Names of additional variables to include in the
                loss normalizer.
            extra_residual_scaled_names: extra_names which use residual scale factors,
                if enabled.
        """
        if self.config is not None:
            if extra_names is None:
                extra_names = []
            else:
                extra_names = list(extra_names)  # avoid mutating input
            if extra_residual_scaled_names is None:
                extra_residual_scaled_names = []
            else:
                extra_residual_scaled_names = list(extra_residual_scaled_names)
            for output_name in self.config.output_names:
                for name in self.config.get_multi_called_names(output_name):
                    extra_names.append(name)
                    if output_name in self.wrapped_step.input_names:
                        extra_residual_scaled_names.append(name)
        return self.wrapped_step.get_loss_normalizer(
            extra_names=extra_names,
            extra_residual_scaled_names=extra_residual_scaled_names,
        )

    @property
    def _multi_call_outputs(self) -> list[str]:
        if self.config is None:
            return []
        return self.config.names

    @property
    def input_names(self) -> list[str]:
        return self.wrapped_step.input_names

    def get_next_step_forcing_names(self) -> list[str]:
        return self.wrapped_step.get_next_step_forcing_names()

    @property
    def output_names(self) -> list[str]:
        return self.wrapped_step.output_names + self._multi_call_outputs

    @property
    def next_step_input_names(self) -> list[str]:
        return self.wrapped_step.next_step_input_names

    @property
    def loss_names(self) -> list[str]:
        if self.include_multi_call_in_loss:
            return self.wrapped_step.loss_names + self._multi_call_outputs
        else:
            return self.wrapped_step.loss_names

    def replace_ocean(self, ocean: OceanConfig | None):
        self.wrapped_step.replace_ocean(ocean)

    def get_ocean(self) -> OceanConfig | None:
        return self.wrapped_step.get_ocean()

    def replace_multi_call(self, multi_call: MultiCallConfig | None):
        self.config = multi_call

    @property
    def n_ic_timesteps(self) -> int:
        return self.wrapped_step.n_ic_timesteps

    def load(self):
        self.wrapped_step.load()


def _extend_normalizer_with_multi_call_outputs(
    config: MultiCallConfig, normalizer: StandardNormalizer
) -> StandardNormalizer:
    means = copy(normalizer.means)
    stds = copy(normalizer.stds)
    for name in config.output_names:
        if name not in means or name not in stds:
            raise ValueError(
                f"Normalizer does not contain {name} present in multi-call output names"
            )
        for multi_call_name in config.get_multi_called_names(name):
            means[multi_call_name] = means[name]
            stds[multi_call_name] = stds[name]
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=normalizer.fill_nans_on_denormalize,
    )


class MultiCallStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    SelfType = TypeVar("SelfType", bound="MultiCallStep")

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        wrapped_step: StepABC,
        config: MultiCallStepConfig,
    ):
        """
        Args:
            wrapped_step: The step to wrap.
            config: The multi-call step configuration.
        """
        super().__init__()
        self._wrapped_step = wrapped_step
        self._config = config
        self._multi_call = config.build(self._wrapped_step.step)
        self._include_multi_call_in_loss = config.include_multi_call_in_loss

    @property
    def config(self) -> MultiCallStepConfig:
        return self._config

    @property
    def modules(self) -> torch.nn.ModuleList:
        return self._wrapped_step.modules

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._config.extend_normalizer_with_multi_call_outputs(
            self._wrapped_step.normalizer
        )

    @property
    def surface_temperature_name(self) -> str | None:
        return self._wrapped_step.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> str | None:
        return self._wrapped_step.ocean_fraction_name

    def get_regularizer_loss(self) -> torch.Tensor:
        return self._wrapped_step.get_regularizer_loss()

    def step(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        wrapper: Callable[[torch.nn.Module], torch.nn.Module] = lambda x: x,
    ) -> TensorDict:
        state = self._wrapped_step.step(
            input,
            next_step_input_data,
            wrapper=wrapper,
        )
        if self._multi_call is not None:
            multi_called_outputs = self._multi_call.step(
                input, next_step_input_data, wrapper=wrapper
            )
            state = {**multi_called_outputs, **state}
        return state

    def get_state(self) -> dict[str, Any]:
        """
        Get the ML model state of the multi-call step.

        Returns:
            The ML model state of the multi-call step.
        """
        return {
            "wrapped_step": self._wrapped_step.get_state(),
        }

    def load_state(self, state: dict[str, Any]):
        """
        Load the ML model state of the multi-call step.

        Does not load the multi-call configuration.

        Args:
            state: The ML model state of the multi-call step.
        """
        self._wrapped_step.load_state(state["wrapped_step"])
