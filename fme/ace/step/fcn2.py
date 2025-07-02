import dataclasses
import datetime
import logging
from collections.abc import Callable
from typing import Any, Literal

import dacite
import torch
from torch import nn

from fme.ace.models.makani_fcn2.models.networks.fourcastnet2 import (  # type: ignore
    AtmoSphericNeuralOperatorNet,
)
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
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


@dataclasses.dataclass
class FCN2Config:
    """
    Configuration for a FCN2 model.
    """

    model_grid_type: str = "legendre-gauss"
    sht_grid_type: str = "legendre-gauss"
    kernel_width: int = 3
    filter_basis_type: str = "morlet"
    filter_basis_norm_mode: str = "mean"
    scale_factor: int = 8
    encoder_mlp: bool = False
    upsample_sht: bool = False
    n_history: int = 0
    atmo_embed_dim: int = 8
    surf_embed_dim: int = 8
    aux_embed_dim: int = 8
    num_layers: int = 4
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    layer_scale: bool = True
    pos_drop_rate: float = 0.0
    path_drop_rate: float = 0.0
    mlp_drop_rate: float = 0.0
    normalization_layer: str = "none"
    max_modes: int | None = None
    hard_thresholding_fraction: float = 1.0
    sfno_block_frequency: int = 2
    bias: bool = False
    checkpointing: int = 0
    freeze_encoder: bool = False
    freeze_processor: bool = False

    def build(
        self,
        n_atmo_channels: int,
        n_atmo_groups: int,
        n_surf_channels: int,
        n_aux_channels: int,
        n_atmo_diagnostic_channels: int,
        n_surf_diagnostic_channels: int,
        img_shape: tuple[int, int],
    ) -> AtmoSphericNeuralOperatorNet:
        return AtmoSphericNeuralOperatorNet(
            n_atmo_channels=n_atmo_channels,
            n_atmo_groups=n_atmo_groups,
            n_surf_channels=n_surf_channels,
            n_aux_channels=n_aux_channels,
            n_atmo_diagnostic_channels=n_atmo_diagnostic_channels,
            n_surf_diagnostic_channels=n_surf_diagnostic_channels,
            model_grid_type=self.model_grid_type,
            sht_grid_type=self.sht_grid_type,
            inp_shape=img_shape,
            out_shape=img_shape,
            kernel_shape=(self.kernel_width, self.kernel_width),
            filter_basis_type=self.filter_basis_type,
            filter_basis_norm_mode=self.filter_basis_norm_mode,
            scale_factor=self.scale_factor,
            encoder_mlp=self.encoder_mlp,
            upsample_sht=self.upsample_sht,
            n_history=self.n_history,
            atmo_embed_dim=self.atmo_embed_dim,
            surf_embed_dim=self.surf_embed_dim,
            aux_embed_dim=self.aux_embed_dim,
            num_layers=self.num_layers,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            layer_scale=self.layer_scale,
            pos_drop_rate=self.pos_drop_rate,
            path_drop_rate=self.path_drop_rate,
            mlp_drop_rate=self.mlp_drop_rate,
            normalization_layer=self.normalization_layer,
            max_modes=self.max_modes,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            sfno_block_frequency=self.sfno_block_frequency,
            bias=self.bias,
            checkpointing=self.checkpointing,
            freeze_encoder=self.freeze_encoder,
            freeze_processor=self.freeze_processor,
        )


@dataclasses.dataclass
class FCN2Selector:
    type: Literal["FCN2"]
    config: FCN2Config

    def build(
        self,
        n_atmo_channels: int,
        n_atmo_groups: int,
        n_surf_channels: int,
        n_aux_channels: int,
        n_atmo_diagnostic_channels: int,
        n_surf_diagnostic_channels: int,
        img_shape: tuple[int, int],
    ) -> AtmoSphericNeuralOperatorNet:
        return self.config.build(
            n_atmo_channels=n_atmo_channels,
            n_atmo_groups=n_atmo_groups,
            n_surf_channels=n_surf_channels,
            n_aux_channels=n_aux_channels,
            n_atmo_diagnostic_channels=n_atmo_diagnostic_channels,
            n_surf_diagnostic_channels=n_surf_diagnostic_channels,
            img_shape=img_shape,
        )


@StepSelector.register("FCN2")
@dataclasses.dataclass
class FCN2StepConfig(StepConfigABC):
    """
    Configuration for a single module stepper.

    Parameters:
        builder: The module builder.
        in_names: Names of input variables.
        out_names: Names of output variables.
        normalization: The normalization configuration.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        residual_prediction: Whether to use residual prediction.
    """

    builder: FCN2Selector
    forcing_names: list[str]
    atmosphere_prognostic_names: list[str]
    atmosphere_diagnostic_names: list[str]
    atmosphere_levels: int
    surface_prognostic_names: list[str]
    surface_diagnostic_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False

    def __post_init__(self):
        for name in self.next_step_forcing_names:
            if name not in self.forcing_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in forcing_names: "
                    f"{self.forcing_names}"
                )
        atmosphere_out_names = []
        atmosphere_in_names = []
        # the FCN2 model expects atmosphere "channels" to be the faster dimension
        # so that they can be encoded together, meaning we must replicate that
        # ordering here.
        for i in range(self.atmosphere_levels):
            for name in self.atmosphere_prognostic_names:
                atmosphere_in_names.append(f"{name}_{i}")
                atmosphere_out_names.append(f"{name}_{i}")
            for name in self.atmosphere_diagnostic_names:
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
    def from_state(cls, state) -> "FCN2StepConfig":
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
        return []  # not currently supported

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self) -> list[str]:
        """Names of variables provided in next_step_input_data."""
        input_only_names = set(self.input_names).difference(self.output_names)
        if self.ocean is None:
            return list(input_only_names)
        return list(input_only_names.union(self.ocean.forcing_names))

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

    @classmethod
    def _remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        return state_copy

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "FCN2Step":
        logging.info("Initializing stepper from provided config")
        corrector = dataset_info.vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=dataset_info.gridded_operations,
            timestep=dataset_info.timestep,
        )
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return FCN2Step(
            config=self,
            img_shape=dataset_info.img_shape,
            corrector=corrector,
            normalizer=normalizer,
            timestep=dataset_info.timestep,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class FCN2Step(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: FCN2StepConfig,
        img_shape: tuple[int, int],
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
            init_weights: Function to initialize the weights of the model.
        """
        super().__init__()
        self.forcing_packer = Packer(config.forcing_names)
        self.atmosphere_input_packer = Packer(config.atmosphere_input_names)
        self.atmosphere_output_packer = Packer(config.atmosphere_output_names)
        self.surface_input_packer = Packer(config.surface_input_names)
        self.surface_output_packer = Packer(config.surface_output_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, timestep
            )
        else:
            self.ocean = None
        module: nn.Module = config.builder.build(
            n_atmo_channels=len(config.atmosphere_prognostic_names)
            + len(config.atmosphere_diagnostic_names),
            n_atmo_groups=config.atmosphere_levels,
            n_atmo_diagnostic_channels=len(config.atmosphere_diagnostic_names),
            n_surf_channels=len(config.surface_prognostic_names)
            + len(config.surface_diagnostic_names),
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
    def config(self) -> FCN2StepConfig:
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

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module])

    def step(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
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
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """

        def network_call(input_norm: TensorDict) -> TensorDict:
            forcing_tensor = self.forcing_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            atmosphere_tensor = self.atmosphere_input_packer.pack(
                input_norm, axis=self.CHANNEL_DIM
            )
            surface_tensor = self.surface_input_packer.pack(
                input_norm, axis=self.CHANNEL_DIM
            )
            atmosphere_output_tensor, surface_output_tensor = wrapper(self.module)(
                atmosphere_tensor, surface_tensor, forcing_tensor
            )
            atmosphere_output = self.atmosphere_output_packer.unpack(
                atmosphere_output_tensor, axis=self.CHANNEL_DIM
            )
            surface_output = self.surface_output_packer.unpack(
                surface_output_tensor, axis=self.CHANNEL_DIM
            )
            return {
                **atmosphere_output,
                **surface_output,
            }

        return step_with_adjustments(
            input=input,
            next_step_input_data=next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=self.prognostic_names,
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        self.module.load_state_dict(state["module"])
