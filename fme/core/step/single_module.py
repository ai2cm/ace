import dataclasses
import datetime
import logging
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.state import CorrectorState
from fme.core.dataset.utils import encode_timestep
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
from fme.core.step.global_mean_removal import (
    GlobalMeanRemoval,
    GlobalMeanRemovalConfigUnion,
    GlobalMeanRemovalState,
    NoGlobalMeanRemoval,
)
from fme.core.step.secondary_decoder import (
    NoSecondaryDecoder,
    SecondaryDecoder,
    SecondaryDecoderConfig,
)
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.stepper_state import StepperState
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.var_masking import VariableMaskingConfig

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


@StepSelector.register("single_module")
@StepSelector.register("default")
@dataclasses.dataclass
class SingleModuleStepConfig(StepConfigABC):
    """
    Configuration for a single module stepper.

    Parameters:
        builder: The module builder.
        in_names: Names of input variables.
        out_names: Names of output variables.
        normalization: The normalization configuration.
        secondary_decoder: Configuration for the secondary decoder that computes
            additional diagnostic variables from outputs.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        prescribed_prognostic_names: Prognostic variable names to overwrite from
            forcing data at each step (e.g. for inference with observed values).
        residual_prediction: Whether to use residual prediction.
        include_channel_mask_inputs: Whether to append per-variable mask indicator
            channels to the network input. When True, the network receives
            ``len(in_names)`` additional float channels (1.0 = present, 0.0 =
            masked) after the regular input channels, doubling the total input
            channel count.
        global_mean_removal: Optional configuration for removing global means
            from fields before normalization and restoring them after
            denormalization. Supports shared (single reference field) or
            per-channel removal, with optional extra input channels.
        input_dropout: Optional training-time input channel dropout. When set,
            a random subset of input channels is zeroed per sample during
            training. Disabled during inference (eval mode).
    """

    builder: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    secondary_decoder: SecondaryDecoderConfig | None = None
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False
    include_channel_mask_inputs: bool = False
    global_mean_removal: GlobalMeanRemovalConfigUnion | None = None
    input_dropout: VariableMaskingConfig | None = None

    def __post_init__(self):
        self.crps_training = None  # unused, kept for backwards compatibility
        if self.global_mean_removal is not None:
            self.global_mean_removal.validate_names(self.in_names, self.out_names)
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
    def from_state(cls, state) -> "SingleModuleStepConfig":
        state = cls._remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        return list(set(self.in_names).union(self.output_names))

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

    @property
    def allow_missing_variables(self) -> bool:
        return self.builder.allow_missing_variables

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
    ) -> "SingleModuleStep":
        logging.info("Initializing stepper from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return SingleModuleStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class SingleModuleStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        """
        Args:
            config: The configuration.
            dataset_info: Information about the dataset.
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
            init_weights: Function to initialize the weights of the module.
        """
        super().__init__()
        if config.global_mean_removal is not None:
            self._global_mean_removal: GlobalMeanRemoval = (
                config.global_mean_removal.build(
                    normalizer=normalizer, in_names=config.in_names
                )
            )
        else:
            self._global_mean_removal = NoGlobalMeanRemoval()
        # Synthetic GMR channels are packed alongside real inputs so they
        # flow through the packer and channel-mask machinery uniformly.
        packed_in_names = (
            list(config.in_names) + self._global_mean_removal.extra_channel_names
        )
        n_in_channels = len(packed_in_names)
        if config.include_channel_mask_inputs:
            n_in_channels *= 2
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(packed_in_names)
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
                    dataset_info=dataset_info,
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

    @property
    def config(self) -> SingleModuleStepConfig:
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
    ) -> tuple[TensorDict, StepperState | None]:
        def network_call(input_norm: TensorDict) -> TensorDict:
            if args.data_mask is not None:
                input_norm = _apply_input_mask(input_norm, args.data_mask)
            input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            channel_mask: torch.Tensor | None = None
            if (
                self._config.input_dropout is not None
                and self.module.torch_module.training
            ):
                batch_size = input_tensor.shape[0]
                n_channels = input_tensor.shape[self.CHANNEL_DIM]
                channel_mask = self._config.input_dropout.sample_mask(
                    n_channels,
                    batch_size,
                    input_tensor.device,
                    n_ensemble=args.n_ensemble,
                )
                input_tensor = input_tensor * channel_mask.view(
                    batch_size, n_channels, 1, 1
                ).to(dtype=input_tensor.dtype)
            if self._config.include_channel_mask_inputs:
                mask_dict = _build_channel_mask_dict(
                    self.in_packer.names, args.data_mask, input_tensor
                )
                mask_tensor = self.in_packer.pack(mask_dict, axis=self.CHANNEL_DIM)
                if channel_mask is not None:
                    mask_tensor = mask_tensor * channel_mask.view(
                        channel_mask.shape[0], n_channels, 1, 1
                    ).to(dtype=mask_tensor.dtype)
                input_tensor = torch.cat(
                    [input_tensor, mask_tensor], dim=self.CHANNEL_DIM
                )
            output_tensor = self.module.wrap_module(wrapper)(
                input_tensor,
                labels=args.labels,
            )
            output_dict = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
            secondary_output_dict = self.secondary_decoder.wrap_module(wrapper)(
                output_tensor.detach()  # detach avoids changing base outputs
            )
            output_dict.update(secondary_output_dict)
            return output_dict

        return step_with_adjustments(
            input=args.input,
            next_step_input_data=args.next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=self.prognostic_names,
            prescribed_prognostic_names=self._config.prescribed_prognostic_names,
            global_mean_removal=self._global_mean_removal,
            data_mask=args.data_mask,
            stepper_state=args.stepper_state,
        )

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


def _apply_input_mask(input_norm: TensorDict, data_mask: TensorMapping) -> TensorDict:
    """Zero out masked input variables in normalized space.

    For each variable in data_mask with False entries, sets those batch
    members' values to 0 in the normalized input. This is equivalent to
    replacing with the climatological mean in physical space.
    """
    result = dict(input_norm)
    for name, mask in data_mask.items():
        if name in result:
            # mask shape: [batch], data shape: [batch, ...spatial...]
            broadcast_mask = mask.view(mask.shape[0], *([1] * (result[name].ndim - 1)))
            result[name] = torch.where(broadcast_mask, result[name], 0.0)
    return result


def _build_channel_mask_dict(
    in_names: list[str],
    data_mask: TensorMapping | None,
    packed_input: torch.Tensor,
) -> TensorDict:
    """Build a dict of per-variable spatial mask tensors.

    Returns a ``TensorDict`` keyed by variable name, with each value a
    ``(batch, *spatial)`` float tensor (1.0 = present, 0.0 = masked).
    The caller is responsible for packing this dict into the correct
    channel order.

    Args:
        in_names: Input variable names.
        data_mask: Per-variable boolean masks of shape ``[batch]``, or None.
        packed_input: The packed input tensor, used to infer shape and device.
    """
    batch = packed_input.shape[0]
    spatial = packed_input.shape[-2:]
    device = packed_input.device
    result: TensorDict = {}
    for name in in_names:
        if data_mask is not None and name in data_mask:
            mask_1d = data_mask[name].to(device=device, dtype=torch.float)
            result[name] = mask_1d.view(batch, 1, 1).expand(batch, *spatial)
        else:
            result[name] = torch.ones(batch, *spatial, device=device)
    return result


def step_with_adjustments(
    input: TensorMapping,
    next_step_input_data: TensorMapping,
    network_calls: Callable[[TensorDict], TensorDict],
    normalizer: StandardNormalizer,
    corrector: CorrectorABC | None,
    ocean: Ocean | None,
    residual_prediction: bool,
    prognostic_names: list[str],
    prescribed_prognostic_names: list[str] | None = None,
    global_mean_removal: GlobalMeanRemoval | None = None,
    data_mask: TensorMapping | None = None,
    stepper_state: StepperState | None = None,
) -> tuple[TensorDict, StepperState | None]:
    """
    Step the model forward one timestep given input data.

    Args:
        input: Mapping from variable name to tensor of shape
            [n_batch, n_lat, n_lon] containing denormalized data from the
            initial timestep. In practice this contains the ML inputs.
        next_step_input_data: Mapping from variable name to tensor of shape
            [n_batch, n_lat, n_lon] containing denormalized data from
            the output timestep. In practice this contains the necessary data
            at the output timestep for the ocean model and corrector.
        network_calls: Callable[[TensorMapping], TensorDict] that takes a
            normalized input and returns a normalized output.
        normalizer: The normalizer to use.
        corrector: The corrector to use at the end of each step.
        ocean: The ocean model to use.
        residual_prediction: Whether to use residual prediction.
        prognostic_names: Names of prognostic variables.
        prescribed_prognostic_names: Prognostic names to overwrite from
            next_step_input_data after the ocean step (e.g. for inference).
        global_mean_removal: Optional transform that removes per-sample
            global means before normalization and restores them after
            denormalization. When provided, ``forward_transform`` is called
            before the normalizer and ``inverse_transform`` after
            denormalization but before the corrector/ocean/prescribed steps,
            so those adjustments operate in physical space.
        data_mask: Per-variable boolean masks passed to
            ``global_mean_removal.forward_transform``.
        stepper_state: Per-sample state carried across step calls. The
            corrector's slice (``stepper_state.corrector_state``) is passed to
            the corrector and any updates are written back into a returned
            ``StepperState``. Pass-through unchanged when no corrector is set.

    Returns:
        A tuple ``(output, stepper_state)`` where ``output`` is the
        denormalized data at the next time step.
    """
    if prescribed_prognostic_names is None:
        prescribed_prognostic_names = []
    gmr_state: GlobalMeanRemovalState | None = None
    if global_mean_removal is not None:
        network_input, gmr_state = global_mean_removal.forward_transform(
            input, data_mask
        )
    else:
        network_input = dict(input)
    input_norm = normalizer.normalize(network_input)
    if global_mean_removal is not None:
        assert gmr_state is not None
        # Synthetic GMR channels are produced in normalized space; merge
        # them in after normalization so the network sees a single uniform
        # input dict.
        input_norm = {**input_norm, **global_mean_removal.extras_normalized(gmr_state)}
    output_norm = network_calls(input_norm)
    if residual_prediction:
        output_norm = add_names(input_norm, output_norm, prognostic_names)
    output = normalizer.denormalize(output_norm)
    if global_mean_removal is not None:
        assert gmr_state is not None
        output = global_mean_removal.inverse_transform(output, gmr_state)
    if corrector is not None:
        corrector_state: CorrectorState | None = (
            stepper_state.corrector_state if stepper_state is not None else None
        )
        output, corrector_state = corrector(
            input, output, next_step_input_data, corrector_state
        )
        if corrector_state is not None:
            stepper_state = StepperState(corrector_state=corrector_state)
    if ocean is not None:
        output = ocean(input, output, next_step_input_data)
    for name in prescribed_prognostic_names:
        if name in next_step_input_data:
            output = {**output, name: next_step_input_data[name]}
        else:
            raise ValueError(
                f"prescribed_prognostic_name '{name}' not in next_step_input_data"
            )
    return output, stepper_state
