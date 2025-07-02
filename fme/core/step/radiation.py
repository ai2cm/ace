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
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("separate_radiation")
@dataclasses.dataclass
class SeparateRadiationStepConfig(StepConfigABC):
    """
    Configuration for a separate radiation stepper.

    Parameters:
        builder: The module builder.
        radiation_builder: The radiation module builder.
        main_prognostic_names: Names of prognostic variables. These are provided
            as input to both the main and radiation models, and output by
            the main model.
        shared_forcing_names: Names of forcing variables.
        radiation_only_forcing_names: Names of forcing variables for the radiation
            model, in addition to the ones specified in `shared_forcing_names`.
        radiation_diagnostic_names: Names of diagnostic variables for the radiation
            model.
        main_diagnostic_names: Names of diagnostic variables for the main model.
        normalization: The normalization configuration.
        next_step_forcing_names: Names of forcing variables which come from
            the output timestep.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        detach_radiation: Whether to detach the output of the radiation model before
            passing it to the main model. The radiation outputs returned by
            .step() will not be detached.
        residual_prediction: Whether to use residual prediction.
    """

    builder: ModuleSelector
    radiation_builder: ModuleSelector
    main_prognostic_names: list[str]
    shared_forcing_names: list[str]
    radiation_only_forcing_names: list[str]
    radiation_diagnostic_names: list[str]
    main_diagnostic_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    detach_radiation: bool = False
    residual_prediction: bool = False

    def __post_init__(self):
        seen_names: dict[str, str] = {}
        for name_list, label in (
            (self.main_prognostic_names, "main_prognostic_names"),
            (self.shared_forcing_names, "shared_forcing_names"),
            (self.radiation_only_forcing_names, "radiation_only_forcing_names"),
            (self.main_diagnostic_names, "main_diagnostic_names"),
            (self.radiation_diagnostic_names, "radiation_diagnostic_names"),
        ):
            for name in name_list:
                if name in seen_names:
                    raise ValueError(
                        f"Name '{name}' appears in multiple name lists: "
                        f"{seen_names[name]} and {label}."
                    )
            seen_names[name] = label
        for name in self.next_step_forcing_names:
            if name not in self._forcing_names:
                raise ValueError(
                    "next_step_forcing_name not in forcing_names: "
                    f"'{name}' not in {self._forcing_names}"
                )

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def get_state(self):
        return dataclasses.asdict(self)

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "SeparateRadiationStep":
        logging.info("Initializing stepper from provided config")
        corrector = dataset_info.vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=dataset_info.gridded_operations,
            timestep=dataset_info.timestep,
        )
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return SeparateRadiationStep(
            config=self,
            img_shape=dataset_info.img_shape,
            corrector=corrector,
            normalizer=normalizer,
            timestep=dataset_info.timestep,
            init_weights=init_weights,
        )

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
    def from_state(cls, state) -> "SeparateRadiationStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self) -> list[str]:
        """Names of variables which require normalization. I.e. inputs/outputs."""
        all_names = set()
        for names in (
            self.main_prognostic_names,
            self.shared_forcing_names,
            self.radiation_only_forcing_names,
            self.main_diagnostic_names,
            self.radiation_diagnostic_names,
        ):
            all_names.update(names)
        return list(all_names)

    @property
    def _forcing_names(self) -> list[str]:
        return list(
            set(self.shared_forcing_names).union(self.radiation_only_forcing_names)
        )

    def get_next_step_forcing_names(self) -> list[str]:
        return self.next_step_forcing_names

    @property
    def diagnostic_names(self) -> list[str]:
        return list(
            set(self.main_diagnostic_names).union(self.radiation_diagnostic_names)
        )

    @property
    def radiation_in_names(self) -> list[str]:
        return (
            self.main_prognostic_names
            + self.shared_forcing_names
            + self.radiation_only_forcing_names
        )

    @property
    def radiation_out_names(self) -> list[str]:
        return self.radiation_diagnostic_names

    @property
    def main_in_names(self) -> list[str]:
        return (
            self.main_prognostic_names
            + self.shared_forcing_names
            + self.radiation_out_names
        )

    @property
    def main_out_names(self) -> list[str]:
        return self.main_prognostic_names + self.main_diagnostic_names

    @property
    def input_names(self) -> list[str]:
        ml_in_names = (
            self.main_prognostic_names
            + self.shared_forcing_names
            + self.radiation_only_forcing_names
        )
        if self.ocean is None:
            return ml_in_names
        else:
            return list(set(ml_in_names).union(self.ocean.forcing_names))

    @property
    def output_names(self) -> list[str]:
        return (
            self.main_prognostic_names
            + self.main_diagnostic_names
            + self.radiation_diagnostic_names
        )

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
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def load(self):
        self.normalization.load()


class SeparateRadiationStep(StepABC):
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SeparateRadiationStepConfig,
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
            init_weights: Function to initialize the weights of the step.
        """
        super().__init__()
        self.in_packer = Packer(config.main_in_names)
        self.out_packer = Packer(config.main_out_names)
        self.radiation_in_packer = Packer(config.radiation_in_names)
        self.radiation_out_packer = Packer(config.radiation_out_names)
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.input_names,
                config.output_names,
                timestep,
            )
        else:
            self.ocean = None
        self.module: nn.Module = config.builder.build(
            n_in_channels=len(config.main_in_names),
            n_out_channels=len(config.main_out_names),
            img_shape=img_shape,
        ).to(get_device())
        self.radiation_module: nn.Module = config.radiation_builder.build(
            n_in_channels=len(config.radiation_in_names),
            n_out_channels=len(config.radiation_out_names),
            img_shape=img_shape,
        ).to(get_device())
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        init_weights(self.modules)
        dist = Distributed.get_instance()
        self.module = dist.wrap_module(self.module)
        self.radiation_module = dist.wrap_module(self.radiation_module)
        self._timestep = timestep
        self._corrector = corrector

    @property
    def config(self) -> SeparateRadiationStepConfig:
        return self._config

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
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module, self.radiation_module])

    def step(
        self,
        input: TensorMapping,
        next_step_forcing_data: TensorMapping,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for `self.module`
                and is assumed to contain all input variables and be denormalized.
            next_step_forcing_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary forcing
                data at the output timestep for the ocean model and corrector.
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """

        def network_calls(input_norm: TensorDict) -> TensorDict:
            radiation_input_tensor = self.radiation_in_packer.pack(
                input_norm, axis=self.CHANNEL_DIM
            )
            radiation_output_tensor = wrapper(self.radiation_module)(
                radiation_input_tensor
            )
            radiation_output_norm = self.radiation_out_packer.unpack(
                radiation_output_tensor, axis=self.CHANNEL_DIM
            )
            main_input_data = input_norm.copy()
            if self._config.detach_radiation:
                main_input_data = {
                    **input_norm,
                    **{k: v.detach() for k, v in radiation_output_norm.items()},
                }
            else:
                main_input_data = {**input_norm, **radiation_output_norm}
            input_tensor = self.in_packer.pack(main_input_data, axis=self.CHANNEL_DIM)
            output_tensor = wrapper(self.module)(input_tensor)
            main_output_norm = self.out_packer.unpack(
                output_tensor, axis=self.CHANNEL_DIM
            )
            return {
                **radiation_output_norm,
                **main_output_norm,
            }

        return step_with_adjustments(
            input=input,
            next_step_input_data=next_step_forcing_data,
            network_calls=network_calls,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=self.prognostic_names,
        )

    def get_regularizer_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def get_state(self):
        """
        Returns:
            The state of the ML modules.
        """
        return {
            "module": self.module.state_dict(),
            "radiation_module": self.radiation_module.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the ML modules.

        Args:
            state: The state to load.
        """
        self.module.load_state_dict(state["module"])
        self.radiation_module.load_state_dict(state["radiation_module"])
