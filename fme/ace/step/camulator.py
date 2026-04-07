import dataclasses
import datetime
import logging
from collections.abc import Callable
from typing import Any, Literal

import dacite
import torch
from torch import nn

from fme.ace.models.miles_credit.crossformer import CrossFormer  # type: ignore
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset.utils import encode_timestep
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector
from fme.core.step.args import StepArgs
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


@dataclasses.dataclass
class CrossFormerConfig:
    """
    Configuration for a CrossFormer model.
    """

    image_height: int = 640
    patch_height: int = 1
    image_width: int = 1280
    patch_width: int = 1
    frames: int = 2
    channels: int = 4
    surface_channels: int = 7
    input_only_channels: int = 3
    output_only_channels: int = 0
    levels: int = 15
    dim: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024, 2048])
    depth: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 8, 2])
    dim_head: int = 32
    global_window_size: list[int] = dataclasses.field(
        default_factory=lambda: [4, 4, 2, 1]
    )
    local_window_size: int = 3
    cross_embed_kernel_sizes: list[list[int]] = dataclasses.field(
        default_factory=lambda: [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]]
    )
    cross_embed_strides: list[int] = dataclasses.field(
        default_factory=lambda: [2, 2, 2, 2]
    )
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    use_spectral_norm: bool = True
    interp: bool = True
    padding_conf: dict | None = None
    post_conf: dict | None = None

    def build(
        self,
        n_atmo_channels: int,
        n_atmo_groups: int,
        n_surf_channels: int,
        n_aux_channels: int,
        n_atmo_diagnostic_channels: int,
        n_surf_diagnostic_channels: int,
        img_shape: tuple[int, int],
    ) -> CrossFormer:
        return CrossFormer(
            image_height=img_shape[0],
            patch_height=self.patch_height,
            image_width=img_shape[1],
            patch_width=self.patch_width,
            frames=self.frames,
            channels=n_atmo_channels,
            surface_channels=n_surf_channels,
            input_only_channels=n_aux_channels,
            output_only_channels=(
                n_surf_diagnostic_channels + n_atmo_diagnostic_channels * n_atmo_groups
            ),
            levels=n_atmo_groups,
            dim=self.dim,
            depth=self.depth,
            dim_head=self.dim_head,
            global_window_size=self.global_window_size,
            local_window_size=self.local_window_size,
            cross_embed_kernel_sizes=self.cross_embed_kernel_sizes,
            cross_embed_strides=self.cross_embed_strides,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            use_spectral_norm=self.use_spectral_norm,
            interp=self.interp,
            padding_conf=self.padding_conf,
            post_conf=self.post_conf,
        )


@dataclasses.dataclass
class CrossFormerSelector:
    type: Literal["CrossFormer"]
    config: CrossFormerConfig

    def build(
        self,
        n_atmo_channels: int,
        n_atmo_groups: int,
        n_surf_channels: int,
        n_aux_channels: int,
        n_atmo_diagnostic_channels: int,
        n_surf_diagnostic_channels: int,
        img_shape: tuple[int, int],
    ) -> CrossFormer:
        return self.config.build(
            n_atmo_channels=n_atmo_channels,
            n_atmo_groups=n_atmo_groups,
            n_surf_channels=n_surf_channels,
            n_aux_channels=n_aux_channels,
            n_atmo_diagnostic_channels=n_atmo_diagnostic_channels,
            n_surf_diagnostic_channels=n_surf_diagnostic_channels,
            img_shape=img_shape,
        )


@StepSelector.register("CrossFormer")
@dataclasses.dataclass
class CrossFormerStepConfig(StepConfigABC):
    """
    Configuration for a CrossFormer stepper.

    Parameters:
        builder: The module builder.
        forcing_names: Names of forcing variables.
        atmosphere_prognostic_names: Names of atmosphere prognostic variables.
        atmosphere_levels: Number of atmosphere levels.
        surface_prognostic_names: Names of surface prognostic variables.
        surface_diagnostic_names: Names of surface diagnostic variables.
        atmosphere_diagnostic_names: Names of atmosphere diagnostic variables.
        normalization: The normalization configuration.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        prescribed_prognostic_names: Names of prognostic variables prescribed from data.
        residual_prediction: Whether to use residual prediction.
    """

    builder: CrossFormerSelector
    forcing_names: list[str]
    atmosphere_prognostic_names: list[str]
    atmosphere_levels: int
    surface_prognostic_names: list[str]
    surface_diagnostic_names: list[str]
    atmosphere_diagnostic_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False
    atmosphere_level_start: int = 0

    def __post_init__(self):
        for name in self.next_step_forcing_names:
            if name not in self.forcing_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in forcing_names: "
                    f"{self.forcing_names}"
                )
        atmosphere_out_names = []
        atmosphere_in_names = []
        level_start = self.atmosphere_level_start
        # the CrossFormer model expects atmosphere "channels" to be the faster
        # dimension so that they can be encoded together, meaning we must
        # replicate that ordering here.
        for name in self.atmosphere_prognostic_names:
            for i in range(level_start, level_start + self.atmosphere_levels):
                atmosphere_in_names.append(f"{name}_{i}")
                atmosphere_out_names.append(f"{name}_{i}")
        if self.atmosphere_diagnostic_names is not None:
            for name in self.atmosphere_diagnostic_names:
                for i in range(level_start, level_start + self.atmosphere_levels):
                    atmosphere_out_names.append(f"{name}_{i}")
        self.atmosphere_input_names = atmosphere_in_names
        self.atmosphere_output_names = atmosphere_out_names
        self.surface_input_names = self.surface_prognostic_names
        self.surface_output_names = (
            self.surface_prognostic_names + self.surface_diagnostic_names
        )
        self.in_names = (
            self.forcing_names + self.atmosphere_input_names + self.surface_input_names
        )
        self.out_names = self.atmosphere_output_names + self.surface_output_names
        for name in self.prescribed_prognostic_names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
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
    def from_state(cls, state) -> "CrossFormerStepConfig":
        state = cls._remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        return list(set(self.in_names).union(self.out_names))

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
        return []

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self) -> list[str]:
        """Names of variables provided in next_step_input_data."""
        result = set(self.input_names).difference(self.output_names)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        result = result.union(self.prescribed_prognostic_names)
        return list(result)

    @property
    def loss_names(self) -> list[str]:
        return self.output_names

    def replace_ocean(self, ocean: OceanConfig | None):
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
        return state_copy

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "CrossFormerStep":
        logging.info("Initializing stepper from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return CrossFormerStep(
            config=self,
            img_shape=dataset_info.img_shape,
            corrector=corrector,
            normalizer=normalizer,
            timestep=dataset_info.timestep,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class CrossFormerStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: CrossFormerStepConfig,
        img_shape: tuple[int, int],
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        self.input_packer = Packer(
            config.forcing_names
            + config.atmosphere_input_names
            + config.surface_input_names
        )
        self.output_packer = Packer(
            config.atmosphere_output_names + config.surface_output_names
        )
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, timestep
            )
        else:
            self.ocean = None
        module: nn.Module = config.builder.build(
            n_atmo_channels=len(config.atmosphere_prognostic_names),
            n_atmo_groups=config.atmosphere_levels,
            n_atmo_diagnostic_channels=len(config.atmosphere_diagnostic_names),
            n_surf_channels=len(config.surface_prognostic_names),
            n_surf_diagnostic_channels=len(config.surface_diagnostic_names),
            n_aux_channels=len(config.forcing_names),
            img_shape=img_shape,
        )
        module = module.to(get_device())
        init_weights([module])

        dist = Distributed.get_instance()
        self.module = dist.wrap_module(module)
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        self._timestep = timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

    @property
    def config(self) -> CrossFormerStepConfig:
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
        return nn.ModuleList([self.module])

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            args: The arguments to the step function.
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """
        if args.labels is not None:
            raise ValueError("Labels are not supported for CrossFormer")

        def network_call(input_norm: TensorDict) -> TensorDict:
            input_tensor = self.input_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            output_tensor = wrapper(self.module)(input_tensor)
            output_tensor = output_tensor.squeeze(2)
            return self.output_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)

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
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        return {
            "module": self.module.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
