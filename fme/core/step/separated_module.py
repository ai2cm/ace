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
from fme.core.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector, SeparatedModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.secondary_decoder import (
    NoSecondaryDecoder,
    SecondaryDecoder,
    SecondaryDecoderConfig,
)
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("separated_module")
@dataclasses.dataclass
class SeparatedModuleStepConfig(StepConfigABC):
    """
    Configuration for a step using a module with separated channel interface.

    Unlike SingleModuleStepConfig which uses in_names/out_names and derives
    forcing/prognostic/diagnostic from set operations, this config specifies
    the three channel categories explicitly.

    Parameters:
        builder: The separated module builder.
        forcing_names: Names of input-only (forcing) variables.
        prognostic_names_: Names of input-output (prognostic) variables.
        diagnostic_names: Names of output-only (diagnostic) variables.
        normalization: The normalization configuration.
        secondary_decoder: Configuration for the secondary decoder that computes
            additional diagnostic variables from outputs.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        prescribed_prognostic_names: Prognostic variable names to overwrite from
            forcing data at each step (e.g. for inference with observed values).
        residual_prediction: Whether to use residual prediction.
    """

    builder: SeparatedModuleSelector
    forcing_names: list[str]
    prognostic_names_: list[str]
    diagnostic_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    secondary_decoder: SecondaryDecoderConfig | None = None
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False

    def __post_init__(self):
        all_names = self.forcing_names + self.prognostic_names_ + self.diagnostic_names
        if len(all_names) != len(set(all_names)):
            seen: dict[str, str] = {}
            for name_list, label in (
                (self.forcing_names, "forcing_names"),
                (self.prognostic_names_, "prognostic_names_"),
                (self.diagnostic_names, "diagnostic_names"),
            ):
                for name in name_list:
                    if name in seen:
                        raise ValueError(
                            f"Name '{name}' appears in both "
                            f"{seen[name]} and {label}."
                        )
                    seen[name] = label
        for name in self.prescribed_prognostic_names:
            if name not in self.prognostic_names_:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in "
                    f"prognostic_names_: {self.prognostic_names_}"
                )
        for name in self.next_step_forcing_names:
            if name not in self.forcing_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in "
                    f"forcing_names: {self.forcing_names}"
                )
        if self.secondary_decoder is not None:
            for name in self.secondary_decoder.secondary_diagnostic_names:
                if name in self.forcing_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is a forcing variable: '{name}'"
                    )
                if name in self.prognostic_names_:
                    raise ValueError(
                        f"secondary_diagnostic_name is a prognostic variable: "
                        f"'{name}'"
                    )
                if name in self.diagnostic_names:
                    raise ValueError(
                        f"secondary_diagnostic_name is a diagnostic variable: "
                        f"'{name}'"
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
            residual_scaled_names=(
                self.prognostic_names_ + extra_residual_scaled_names
            ),
        )

    @classmethod
    def from_state(cls, state) -> "SeparatedModuleStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self):
        """Names of variables which require normalization."""
        return list(set(self.forcing_names).union(self.output_names))

    @property
    def input_names(self) -> list[str]:
        names = self.forcing_names + self.prognostic_names_
        if self.ocean is not None:
            names = list(set(names).union(self.ocean.forcing_names))
        return names

    def get_next_step_forcing_names(self) -> list[str]:
        return self.next_step_forcing_names

    @property
    def output_names(self) -> list[str]:
        names = self.prognostic_names_ + self.diagnostic_names
        if self.secondary_decoder is not None:
            names = list(
                set(names).union(self.secondary_decoder.secondary_diagnostic_names)
            )
        return names

    @property
    def next_step_input_names(self) -> list[str]:
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
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        for name in names:
            if name not in self.prognostic_names_:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in "
                    f"prognostic_names_: {self.prognostic_names_}"
                )
        self.prescribed_prognostic_names = names

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "SeparatedModuleStep":
        logging.info("Initializing separated module stepper from provided config")
        corrector = dataset_info.vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=dataset_info.gridded_operations,
            timestep=dataset_info.timestep,
        )
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        return SeparatedModuleStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class SeparatedModuleStep(StepABC):
    """
    Step class for a module with separated forcing/prognostic/diagnostic
    channel interface.
    """

    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SeparatedModuleStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        n_forcing = len(config.forcing_names)
        n_prognostic = len(config.prognostic_names_)
        n_diagnostic = len(config.diagnostic_names)

        self.forcing_packer = Packer(config.forcing_names)
        self.prognostic_packer = Packer(config.prognostic_names_)
        self.diagnostic_packer = Packer(config.diagnostic_names)

        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.forcing_names + config.prognostic_names_,
                config.prognostic_names_ + config.diagnostic_names,
                dataset_info.timestep,
            )
        else:
            self.ocean = None

        module = config.builder.build(
            n_forcing_channels=n_forcing,
            n_prognostic_channels=n_prognostic,
            n_diagnostic_channels=n_diagnostic,
            dataset_info=dataset_info,
        )
        self.module = module.to(get_device())

        dist = Distributed.get_instance()

        if config.secondary_decoder is not None:
            self.secondary_decoder: SecondaryDecoder | NoSecondaryDecoder = (
                config.secondary_decoder.build(
                    n_in_channels=n_prognostic + n_diagnostic,
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

    @property
    def config(self) -> SeparatedModuleStepConfig:
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

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        def network_call(input_norm: TensorDict) -> TensorDict:
            forcing = self.forcing_packer.pack(input_norm, axis=self.CHANNEL_DIM)

            # Get a reference tensor for shape/device info
            ref = forcing

            if len(self.prognostic_packer.names) > 0:
                prognostic_in = self.prognostic_packer.pack(
                    input_norm, axis=self.CHANNEL_DIM
                )
            else:
                prognostic_in = torch.zeros(
                    *ref.shape[:-3],
                    0,
                    *ref.shape[-2:],
                    dtype=ref.dtype,
                    device=ref.device,
                )

            prog_out, diag_out = self.module.wrap_module(wrapper)(
                forcing, prognostic_in, labels=args.labels
            )

            output_dict: TensorDict = {}
            if len(self.prognostic_packer.names) > 0:
                output_dict.update(
                    self.prognostic_packer.unpack(prog_out, axis=self.CHANNEL_DIM)
                )
            if len(self.diagnostic_packer.names) > 0:
                output_dict.update(
                    self.diagnostic_packer.unpack(diag_out, axis=self.CHANNEL_DIM)
                )

            # Secondary decoder gets concatenated prog+diag output
            combined_out = torch.cat([prog_out, diag_out], dim=self.CHANNEL_DIM)
            secondary_output_dict = self.secondary_decoder.wrap_module(wrapper)(
                combined_out.detach()
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
            prognostic_names=self._config.prognostic_names_,
            prescribed_prognostic_names=self._config.prescribed_prognostic_names,
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        return {
            "module": self.module.get_state(),
            "secondary_decoder": self.secondary_decoder.get_module_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.module.load_state(state["module"])
        if "secondary_decoder" in state:
            self.secondary_decoder.load_module_state(state["secondary_decoder"])
