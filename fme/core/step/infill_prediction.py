import dataclasses
import logging
from collections.abc import Callable, Mapping
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
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
from fme.core.step.single_module import (
    _apply_input_mask,
    _build_channel_mask_dict,
    step_with_adjustments,
)
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.stepper_state import StepperState
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class InferenceSchemeConfig:
    """Defines how the model behaves at inference time.

    This mirrors the variable routing of SingleModuleStepConfig for
    standard forward prediction.

    Attributes:
        in_names: Input variable names for inference.
        out_names: Output variable names for inference.
        next_step_forcing_names: Input variables that come from the output
            timestep.
        prescribed_prognostic_names: Prognostic variables overwritten from
            forcing.
    """

    in_names: list[str]
    out_names: list[str]
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)


@StepSelector.register("infill_prediction")
@dataclasses.dataclass
class InfillPredictionStepConfig(StepConfigABC):
    """Configuration for a step that trains on a broader variable pool than
    it uses at inference time.

    At inference time, the step behaves like ``SingleModuleStepConfig`` using
    the ``inference_scheme``'s variable routing. During training, the data
    loader fetches all variables in ``all_names`` and the step zero-fills any
    missing variables, attaching per-channel mask indicators so the network
    can distinguish real inputs from fill values.

    Parameters:
        builder: The module builder.
        all_names: All variable names used during training.
        forcing_names: Names of forcing (input-only) variables.
        normalization: The normalization configuration.
        inference_scheme: Defines variable routing at inference time.
        include_channel_mask_inputs: Must be True. Appends per-variable
            mask indicator channels so the network knows which inputs
            are real vs masked.
        secondary_decoder: Configuration for the secondary decoder.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
    """

    builder: ModuleSelector
    all_names: list[str]
    forcing_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    inference_scheme: InferenceSchemeConfig
    include_channel_mask_inputs: bool = True
    secondary_decoder: SecondaryDecoderConfig | None = None
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )

    def __post_init__(self):
        if not self.include_channel_mask_inputs:
            raise ValueError(
                "include_channel_mask_inputs must be True for "
                "InfillPredictionStepConfig"
            )
        all_set = set(self.all_names)
        for name in self.inference_scheme.in_names:
            if name not in all_set:
                raise ValueError(
                    f"inference_scheme.in_names contains '{name}' "
                    f"which is not in all_names"
                )
        for name in self.inference_scheme.out_names:
            if name not in all_set:
                raise ValueError(
                    f"inference_scheme.out_names contains '{name}' "
                    f"which is not in all_names"
                )
        for name in self.forcing_names:
            if name not in all_set:
                raise ValueError(
                    f"forcing_names contains '{name}' which is not in all_names"
                )
            if name in self.inference_scheme.out_names:
                raise ValueError(
                    f"forcing variable '{name}' must not be in "
                    f"inference_scheme.out_names"
                )
        for name in self.inference_scheme.next_step_forcing_names:
            if name not in self.inference_scheme.in_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in "
                    f"inference_scheme.in_names"
                )
            if name in self.inference_scheme.out_names:
                raise ValueError(
                    f"next_step_forcing_name is an output variable: '{name}'"
                )
        for name in self.inference_scheme.prescribed_prognostic_names:
            if name not in self.inference_scheme.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in "
                    f"inference_scheme.out_names"
                )
        if self.secondary_decoder is not None:
            for name in self.secondary_decoder.secondary_diagnostic_names:
                if name in self.inference_scheme.in_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an input variable: '{name}'"
                    )
                if name in self.inference_scheme.out_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is an output variable: '{name}'"
                    )

    @property
    def non_forcing_names(self) -> list[str]:
        forcing_set = set(self.forcing_names)
        return [n for n in self.all_names if n not in forcing_set]

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    @property
    def input_names(self) -> list[str]:
        if self.ocean is None:
            return self.inference_scheme.in_names
        return list(set(self.inference_scheme.in_names).union(self.ocean.forcing_names))

    @property
    def output_names(self) -> list[str]:
        secondary_names = (
            self.secondary_decoder.secondary_diagnostic_names
            if self.secondary_decoder is not None
            else []
        )
        return list(set(self.inference_scheme.out_names).union(secondary_names))

    @property
    def next_step_input_names(self) -> list[str]:
        input_only_names = set(self.input_names).difference(self.output_names)
        result = set(input_only_names)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        result = result.union(self.inference_scheme.prescribed_prognostic_names)
        return list(result)

    @property
    def loss_names(self) -> list[str]:
        return self.non_forcing_names

    @property
    def all_training_names(self) -> list[str] | None:
        return list(self.all_names)

    @property
    def allow_missing_variables(self) -> bool:
        return True

    def get_next_step_forcing_names(self) -> list[str]:
        return self.inference_scheme.next_step_forcing_names

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        if extra_names is None:
            extra_names = []
        return self.normalization.get_loss_normalizer(
            names=list(set(self.all_names)) + extra_names,
            residual_scaled_names=[],
        )

    def replace_ocean(self, ocean: OceanConfig | None):
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        for name in names:
            if name not in self.inference_scheme.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in "
                    f"inference_scheme.out_names: {self.inference_scheme.out_names}"
                )
        self.inference_scheme.prescribed_prognostic_names = names

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "InfillPredictionStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "InfillPredictionStep":
        logging.info("Initializing InfillPredictionStep from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(
            list(set(self.all_names))
        )
        return InfillPredictionStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class InfillPredictionStep(StepABC):
    """Step that trains with all_names but infers with a narrower routing.

    Uses all_names for input channels (with channel mask indicators) and
    non-forcing names for output channels. At training time, missing
    variables are zero-filled and flagged via the per-channel mask so the
    network learns to handle partial inputs. At inference time, the routing
    matches ``inference_scheme``.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: InfillPredictionStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        self._config = config
        self._normalizer = normalizer
        self._corrector = corrector

        n_in_channels = len(config.all_names) * 2
        n_out_channels = len(config.non_forcing_names)
        self.in_packer = Packer(config.all_names)
        self.out_packer = Packer(config.non_forcing_names)

        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.inference_scheme.in_names,
                config.inference_scheme.out_names,
                dataset_info.timestep,
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
                    dataset_info=dataset_info,
                ).to(get_device())
            )
        else:
            self.secondary_decoder = NoSecondaryDecoder()

        init_weights(self.modules)
        self._img_shape = dataset_info.img_shape
        self._no_optimization = NullOptimization()

        self.module = self.module.wrap_module(dist.wrap_module)
        self.secondary_decoder = self.secondary_decoder.wrap_module(dist.wrap_module)
        self._timestep = dataset_info.timestep

    @property
    def config(self) -> InfillPredictionStepConfig:
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
        modules = [self.module.torch_module]
        modules.extend(self.secondary_decoder.torch_modules)
        return nn.ModuleList(modules)

    def _fill_missing_inputs(
        self, input_data: TensorMapping, data_mask: TensorMapping | None
    ) -> tuple[TensorDict, TensorMapping]:
        """Fill missing variables with zeros and build augmented data_mask."""
        ref_tensor = next(iter(input_data.values()))
        batch = ref_tensor.shape[0]
        spatial = ref_tensor.shape[-2:]
        device = ref_tensor.device

        filled: TensorDict = {}
        augmented_mask: dict[str, torch.Tensor] = {}

        if data_mask is not None:
            augmented_mask.update(data_mask)

        for name in self._config.all_names:
            if name in input_data:
                filled[name] = input_data[name]
                if name not in augmented_mask:
                    augmented_mask[name] = torch.ones(
                        batch, dtype=torch.bool, device=device
                    )
            else:
                filled[name] = torch.zeros(batch, *spatial, device=device)
                augmented_mask[name] = torch.zeros(
                    batch, dtype=torch.bool, device=device
                )

        return filled, augmented_mask

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> tuple[TensorDict, StepperState | None]:
        has_all_names = all(n in args.input for n in self._config.all_names)
        if has_all_names:
            full_input = dict(args.input)
            full_data_mask = args.data_mask
        else:
            full_input, full_data_mask = self._fill_missing_inputs(
                args.input, args.data_mask
            )

        def network_call(input_norm: TensorDict) -> TensorDict:
            if full_data_mask is not None:
                input_norm = _apply_input_mask(input_norm, full_data_mask)
            input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            mask_dict = _build_channel_mask_dict(
                self._config.all_names, full_data_mask, input_tensor
            )
            mask_tensor = self.in_packer.pack(mask_dict, axis=self.CHANNEL_DIM)
            input_tensor = torch.cat([input_tensor, mask_tensor], dim=self.CHANNEL_DIM)
            output_tensor = self.module.wrap_module(wrapper)(
                input_tensor,
                labels=args.labels,
            )
            output_dict = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
            secondary_output_dict = self.secondary_decoder.wrap_module(wrapper)(
                output_tensor.detach()
            )
            output_dict.update(secondary_output_dict)
            return output_dict

        return step_with_adjustments(
            input=full_input,
            next_step_input_data=args.next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=False,
            prognostic_names=self.prognostic_names,
            prescribed_prognostic_names=(
                self._config.inference_scheme.prescribed_prognostic_names
            ),
            data_mask=full_data_mask,
            stepper_state=args.stepper_state,
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        return {
            "module": self.module.get_state(),
            "secondary_decoder": self.secondary_decoder.get_module_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        module = state["module"]
        if "module.device_buffer" in module:
            del module["module.device_buffer"]
        self.module.load_state(module)
        if "secondary_decoder" in state:
            self.secondary_decoder.load_module_state(state["secondary_decoder"])
