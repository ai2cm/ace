import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.dicts import add_names
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.secondary_decoder import (
    NoSecondaryDecoder,
    SecondaryDecoder,
    SecondaryDecoderConfig,
)
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


def _build_mask_map(
    mask_names: list[str],
    data_names: list[str],
    all_mask_names: set[str],
    prefix: str,
) -> dict[str, list[str]]:
    """Map each mask variable to the data variables it covers.

    The link is the suffix after ``prefix``: ``below_surface_mask1000``
    covers every non-mask variable whose name ends with ``1000``.
    """
    result: dict[str, list[str]] = {}
    for mask_name in mask_names:
        suffix = mask_name[len(prefix) :]
        result[mask_name] = [
            n for n in data_names if n.endswith(suffix) and n not in all_mask_names
        ]
    return result


@StepSelector.register("cmip6")
@dataclasses.dataclass
class Cmip6StepConfig(StepConfigABC):
    """
    Configuration for a CMIP6 pressure-level stepper.

    Extends the single-module step pattern with handling for time-varying
    below-surface masks.  Mask variables (identified by ``mask_variable_prefix``)
    bypass normalization, have sigmoid applied to their network outputs, and
    are excluded from residual prediction.

    Parameters:
        builder: The module builder.
        in_names: Names of input variables.
        out_names: Names of output variables.
        normalization: The normalization configuration.
        mask_variable_prefix: Prefix that identifies below-surface mask variables
            in in_names/out_names (e.g. "below_surface_mask").
        secondary_decoder: Configuration for the secondary decoder that computes
            additional diagnostic variables from outputs.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        prescribed_prognostic_names: Prognostic variable names to overwrite from
            forcing data at each step (e.g. for inference with observed values).
        residual_prediction: Whether to use residual prediction.
    """

    builder: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    mask_variable_prefix: str = "below_surface_mask"
    secondary_decoder: SecondaryDecoderConfig | None = None
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False

    def __post_init__(self):
        self.crps_training = None  # unused, kept for backwards compatibility
        self.mask_in_names = [
            n for n in self.in_names if n.startswith(self.mask_variable_prefix)
        ]
        self.mask_out_names = [
            n for n in self.out_names if n.startswith(self.mask_variable_prefix)
        ]
        for name in self.prescribed_prognostic_names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
                )
        for name in self.next_step_forcing_names:
            if name not in self.in_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in in_names: {self.in_names}"
                )
            if name in self.out_names:
                raise ValueError(
                    f"next_step_forcing_name is an output variable: '{name}'"
                )
        if self.secondary_decoder is not None:
            for name in self.secondary_decoder.secondary_diagnostic_names:
                if name in self.in_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an input variable: '{name}'"
                    )
                if name in self.out_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an output variable: '{name}'"
                    )

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def get_state(self):
        return dataclasses.asdict(self)

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        if extra_names is None:
            extra_names = []
        if extra_residual_scaled_names is None:
            extra_residual_scaled_names = []
        return self.normalization.get_loss_normalizer(
            names=self._normalize_names + extra_names,
            residual_scaled_names=self.prognostic_names + extra_residual_scaled_names,
        )

    @classmethod
    def from_state(cls, state) -> "Cmip6StepConfig":
        state = cls._remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization.

        Mask variables are excluded — they bypass the normalizer.
        """
        all_mask = set(self.mask_in_names) | set(self.mask_out_names)
        return list(set(self.in_names).union(self.output_names) - all_mask)

    @property
    def input_names(self) -> list[str]:
        """
        Names of variables required as inputs to `step`,
        either in `input` or `next_step_input_data`.
        """
        if self.ocean is None:
            return self.in_names
        else:
            return list(set(self.in_names).union(self.ocean.forcing_names))

    def get_next_step_forcing_names(self) -> list[str]:
        """Names of input-only variables which come from the output timestep."""
        return self.next_step_forcing_names

    @property
    def diagnostic_names(self) -> list[str]:
        """Names of variables which are outputs only."""
        return list(set(self.output_names).difference(self.in_names))

    @property
    def output_names(self) -> list[str]:
        secondary_names = (
            self.secondary_decoder.secondary_diagnostic_names
            if self.secondary_decoder is not None
            else []
        )
        return list(set(self.out_names).union(secondary_names))

    @property
    def next_step_input_names(self) -> list[str]:
        """Names of variables provided in next_step_input_data."""
        input_only_names = set(self.input_names).difference(self.output_names)
        result = set(input_only_names)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        result = result.union(self.prescribed_prognostic_names)
        return list(result)

    @property
    def loss_names(self) -> list[str]:
        return self.output_names

    def replace_ocean(self, ocean: OceanConfig | None):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        """Replace prescribed prognostic names (e.g. when loading from checkpoint)."""
        for name in names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
                )
        self.prescribed_prognostic_names = names

    @classmethod
    def _remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        if "crps_training" in state_copy:
            del state_copy["crps_training"]
        return state_copy

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "Cmip6Step":
        logging.info("Initializing stepper from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return Cmip6Step(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class Cmip6Step(StepABC):
    """
    Step class for CMIP6 pressure-level data with time-varying below-surface
    masks.

    Mask variables bypass the normalizer, have sigmoid applied to their
    network outputs, and are excluded from residual prediction.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: Cmip6StepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, dataset_info.timestep
            )
        else:
            self.ocean = None
        module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
        )
        self.module = module.to(get_device())

        dist = Distributed.get_instance()

        if config.secondary_decoder is not None:
            self.secondary_decoder: SecondaryDecoder | NoSecondaryDecoder = (
                config.secondary_decoder.build(
                    n_in_channels=n_out_channels,
                ).to(get_device())
            )
        else:
            self.secondary_decoder = NoSecondaryDecoder()

        init_weights(self.modules)
        self._img_shape = dataset_info.img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        self.module = self.module.wrap_module(dist.wrap_module)
        self.secondary_decoder = self.secondary_decoder.wrap_module(dist.wrap_module)
        self._timestep = dataset_info.timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names
        self._mask_in_names = set(config.mask_in_names)
        self._mask_out_names = set(config.mask_out_names)
        self._non_mask_prognostic_names = [
            n for n in self.prognostic_names if n not in self._mask_out_names
        ]
        all_mask = self._mask_in_names | self._mask_out_names
        prefix = config.mask_variable_prefix
        self._input_mask_map = _build_mask_map(
            config.mask_in_names, config.in_names, all_mask, prefix
        )

    @property
    def config(self) -> Cmip6StepConfig:
        return self._config

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def surface_temperature_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.surface_temperature_name
        return None

    @property
    def ocean_fraction_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.ocean_fraction_name
        return None

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        if self.ocean is None:
            raise RuntimeError(
                "The Ocean interface is missing but required to prescribe "
                "sea surface temperature."
            )
        return self.ocean.prescriber(mask_data, gen_data, target_data)

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        modules = [self.module.torch_module]
        modules.extend(self.secondary_decoder.torch_modules)
        return nn.ModuleList(modules)

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        input_data = args.input

        # Mask variables bypass the normalizer.
        mask_input = {k: input_data[k] for k in self._mask_in_names if k in input_data}
        input_norm = self.normalizer.normalize(input_data)
        input_norm.update(mask_input)

        # Zero out below-surface cells in normalized space so NaN doesn't
        # enter the network.
        for mask_name, data_names in self._input_mask_map.items():
            if mask_name in input_norm:
                mask = input_norm[mask_name] > 0.5
                for data_name in data_names:
                    if data_name in input_norm:
                        input_norm[data_name] = input_norm[data_name].masked_fill(
                            mask, 0.0
                        )

        # Network forward pass.
        input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        output_tensor = self.module.wrap_module(wrapper)(
            input_tensor, labels=args.labels
        )
        output_norm = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
        secondary_output = self.secondary_decoder.wrap_module(wrapper)(
            output_tensor.detach()
        )
        output_norm.update(secondary_output)

        # Sigmoid on mask outputs (network produces logits, we want probabilities).
        for name in self._mask_out_names:
            if name in output_norm:
                output_norm[name] = torch.sigmoid(output_norm[name])

        # Residual prediction — masks are excluded (they are binary, not residuals).
        if self._config.residual_prediction:
            output_norm = add_names(
                input_norm, output_norm, self._non_mask_prognostic_names
            )

        # Extract mask outputs before denormalization (they bypass it).
        mask_output = {
            k: output_norm.pop(k)
            for k in list(self._mask_out_names)
            if k in output_norm
        }
        output = self.normalizer.denormalize(output_norm)
        output.update(mask_output)

        if self._corrector is not None:
            output = self._corrector(input_data, output, args.next_step_input_data)
        if self.ocean is not None:
            output = self.ocean(input_data, output, args.next_step_input_data)
        for name in self._config.prescribed_prognostic_names:
            if name in args.next_step_input_data:
                output = {**output, name: args.next_step_input_data[name]}
            else:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' not in next_step_input_data"
                )

        return output

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        state = {
            "module": self.module.get_state(),
            "secondary_decoder": self.secondary_decoder.get_module_state(),
        }
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        module = state["module"]
        if "module.device_buffer" in module:
            # for backwards compatibility with old checkpoints
            del module["module.device_buffer"]
        self.module.load_state(module)
        if "secondary_decoder" in state:
            self.secondary_decoder.load_module_state(state["secondary_decoder"])
