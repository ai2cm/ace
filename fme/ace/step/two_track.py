import dataclasses
import logging
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.ace.registry.two_track_sfno import TwoTrackSFNOBuilder
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.labels import LabelEncoding
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector
from fme.core.registry.module import Module
from fme.core.step.args import StepArgs
from fme.core.step.output import StepOutput
from fme.core.step.saturation_normalization import (
    SaturationNormalization,
    SaturationNormalizationConfig,
    add_identity_names,
    build_saturation_normalization,
    derived_normalization_names,
    resolve_all_fields,
)
from fme.core.step.single_module import step_with_adjustments
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


@StepSelector.register("two_track")
@dataclasses.dataclass
class TwoTrackStepConfig(StepConfigABC):
    """Configuration for a two-track (global / local latent) SFNO step.

    Variables are explicitly assigned to a global track (whose latents pass
    through the spherical harmonic transform) or a local, pointwise-only track,
    separately on the input and output sides. Inputs are packed as
    ``global_in_names + local_in_names`` into a single tensor; the network
    splits at ``len(global_in_names)``. Outputs are unpacked as
    ``global_out_names + local_out_names``. Aside from the two-track packing
    and passing the four per-track channel counts to the builder, the step
    mirrors ``SingleModuleStep``'s orchestration
    (normalization / corrector / ocean / residual prediction).

    Parameters:
        builder: The two-track SFNO builder.
        global_in_names: Input variables on the global (spectral) track.
        local_in_names: Input variables on the local (pointwise) track.
        global_out_names: Output variables on the global track.
        local_out_names: Output variables on the local track.
        normalization: The normalization configuration.
        ocean: The ocean configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Forcing variables taken from the output step.
        prescribed_prognostic_names: Prognostic names overwritten from forcing.
        residual_prediction: Whether to use residual prediction.
        saturation_normalization: Optional list of rules expressing humidity
            fields in a relative-humidity-like ``q / q_sat`` space (see
            ``fme.core.step.saturation_normalization``). Every derived
            ``relative_humidity_<level>`` channel is routed to the local
            (never-spectral) track, since the point of the two-track split is to
            keep the pointwise RH representation off the spherical-harmonic
            path. Only ``input='append'`` with ``prediction=False`` is
            supported here: the source ``q`` stays on its declared track and the
            derived RH rides as an additional local input. ``input='replace'``
            and ``prediction=True`` would move the source ``q`` across tracks and
            are rejected in ``TwoTrackStep`` (use the single-module step for
            those).
    """

    builder: TwoTrackSFNOBuilder
    global_in_names: list[str]
    local_in_names: list[str]
    global_out_names: list[str]
    local_out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
    residual_prediction: bool = False
    saturation_normalization: list[SaturationNormalizationConfig] | None = None

    def __post_init__(self):
        in_overlap = set(self.global_in_names) & set(self.local_in_names)
        if in_overlap:
            raise ValueError(
                "a variable may not appear in both global_in_names and "
                f"local_in_names: {sorted(in_overlap)}"
            )
        out_overlap = set(self.global_out_names) & set(self.local_out_names)
        if out_overlap:
            raise ValueError(
                "a variable may not appear in both global_out_names and "
                f"local_out_names: {sorted(out_overlap)}"
            )
        self.in_names = self.global_in_names + self.local_in_names
        self.out_names = self.global_out_names + self.local_out_names
        if self.saturation_normalization is not None:
            # resolve_all_fields validates each rule (patterns matching nothing,
            # missing level suffix, input/prediction referencing a side the
            # field is not on, derived-name collisions) and rejects a field or
            # derived channel appearing in more than one rule.
            saturation_fields = resolve_all_fields(
                self.saturation_normalization, self.in_names, self.out_names
            )
            if self.residual_prediction:
                # Residual prediction adds the input to the output in network
                # space, so a prognostic field predicted in RH must also be fed
                # in RH (and vice versa) for the residual base to be defined.
                in_set, out_set = set(self.in_names), set(self.out_names)
                for field in saturation_fields:
                    prognostic = field.name in in_set and field.name in out_set
                    if prognostic and field.rh_out != field.rh_in:
                        raise ValueError(
                            "saturation_normalization with residual_prediction "
                            f"requires prognostic field '{field.name}' to be RH "
                            "on both input and output: prediction=True must pair "
                            f"with input='replace'. Got prediction={field.rh_out}"
                            f", input replaces q={field.rh_in}."
                        )
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
        loss_normalizer = self.normalization.get_loss_normalizer(
            names=self._normalize_names + extra_names,
            residual_scaled_names=self.prognostic_names + extra_residual_scaled_names,
        )
        # Derived RH channels are O(1) and need identity statistics in the loss
        # scaler (they are otherwise dropped, as the normalizer filters to known
        # names).
        return add_identity_names(loss_normalizer, self.saturation_derived_names)

    @classmethod
    def from_state(cls, state) -> "TwoTrackStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _normalize_names(self) -> list[str]:
        """Names of variables which require normalization (inputs/outputs)."""
        return list(set(self.in_names).union(self.output_names))

    @property
    def saturation_derived_names(self) -> list[str]:
        """Derived RH channel names (internal), which need identity normalizer
        statistics in the network and loss normalizers.
        """
        if self.saturation_normalization is None:
            return []
        fields = resolve_all_fields(
            self.saturation_normalization, self.in_names, self.out_names
        )
        return derived_normalization_names(fields)

    @property
    def input_names(self) -> list[str]:
        if self.ocean is None:
            return self.in_names
        return list(set(self.in_names).union(self.ocean.forcing_names))

    def get_next_step_forcing_names(self) -> list[str]:
        return self.next_step_forcing_names

    @property
    def diagnostic_names(self) -> list[str]:
        return list(set(self.output_names).difference(self.in_names))

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self) -> list[str]:
        result = set(self.input_names).difference(self.output_names)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        result = result.union(self.prescribed_prognostic_names)
        return list(result)

    @property
    def loss_names(self) -> list[str]:
        return self.output_names

    @property
    def allow_missing_variables(self) -> bool:
        return False

    def replace_ocean(self, ocean: OceanConfig | None):
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def replace_prescribed_prognostic_names(self, names: list[str]) -> None:
        for name in names:
            if name not in self.out_names:
                raise ValueError(
                    f"prescribed_prognostic_name '{name}' must be in out_names: "
                    f"{self.out_names}"
                )
        self.prescribed_prognostic_names = names

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "TwoTrackStep":
        logging.info("Initializing two-track stepper from provided config")
        corrector = self.corrector.get_corrector(dataset_info)
        normalizer = self.normalization.get_network_normalizer(self._normalize_names)
        # Register the derived RH channels (identity) so they survive the
        # normalizer's name filter on the network input and output.
        normalizer = add_identity_names(normalizer, self.saturation_derived_names)
        return TwoTrackStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )

    def load(self):
        self.normalization.load()


class TwoTrackStep(StepABC):
    """Step for the two-track (global / local latent) SFNO network."""

    CHANNEL_DIM = -3

    def __init__(
        self,
        config: TwoTrackStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        self._saturation_normalization: SaturationNormalization | None = (
            build_saturation_normalization(
                config.saturation_normalization,
                dataset_info.atmosphere_vertical_coordinate,
                config.in_names,
                config.out_names,
            )
        )
        # Saturation normalization carries humidity under derived RH channel
        # names; those channels always ride the local (never-spectral) track, so
        # they are appended after the local segment on each side. Only the
        # append/prediction-off case is supported: moving the source q across
        # tracks (replace on input, prediction on output) is rejected here.
        derived_input_names: list[str] = []
        derived_output_names: list[str] = []
        if self._saturation_normalization is not None:
            assert config.saturation_normalization is not None
            fields = resolve_all_fields(
                config.saturation_normalization, config.in_names, config.out_names
            )
            for field in fields:
                if field.rh_in:  # input='replace' removes the source q channel
                    raise ValueError(
                        "saturation_normalization input='replace' is not "
                        f"supported by the two-track step (field '{field.name}'): "
                        "it would move the source q off its declared track. Use "
                        "input='append' (RH rides as an extra local input) or the "
                        "single-module step."
                    )
                if field.rh_out:  # prediction=True swaps the source q on output
                    raise ValueError(
                        "saturation_normalization prediction=True is not "
                        f"supported by the two-track step (field '{field.name}'): "
                        "it would move the predicted q off its declared track. "
                        "Use prediction=False or the single-module step."
                    )
                if field.rh_extra:  # input='append'
                    derived_input_names.append(field.rh_name)
        # Packing order matches the network's split: global inputs first, then
        # local (with any derived RH channels appended to the local segment).
        # The network splits the input tensor at len(global_in_names) and
        # returns [global_out | local_out].
        self.in_packer = Packer(
            config.global_in_names + config.local_in_names + derived_input_names
        )
        self.out_packer = Packer(
            config.global_out_names + config.local_out_names + derived_output_names
        )
        self._normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, dataset_info.timestep
            )
        else:
            self.ocean = None

        net = config.builder.build_two_track(
            global_in_channels=len(config.global_in_names),
            local_in_channels=len(config.local_in_names) + len(derived_input_names),
            global_out_channels=len(config.global_out_names),
            local_out_channels=len(config.local_out_names) + len(derived_output_names),
            dataset_info=dataset_info,
        )
        # Wrap in Module so the tensor boundary stays single-tensor in / single
        # tensor out with optional label conditioning, as for SingleModuleStep.
        if len(dataset_info.all_labels) > 0:
            label_encoding: LabelEncoding | None = LabelEncoding(
                sorted(dataset_info.all_labels)
            )
        else:
            label_encoding = None
        self.module = Module(net, label_encoding).to(get_device())

        init_weights(self.modules)
        self._img_shape = dataset_info.img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self.module = self.module.wrap_module(dist.wrap_module)
        self._timestep = dataset_info.timestep
        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

    @property
    def config(self) -> TwoTrackStepConfig:
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
        return nn.ModuleList([self.module.torch_module])

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> StepOutput:
        def network_call(input_norm: TensorDict) -> TensorDict:
            input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
            output_tensor = self.module.wrap_module(wrapper)(
                input_tensor,
                labels=args.labels,
            )
            return self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)

        # Residual prediction adds the input to the output in network space, so
        # prognostic fields predicted in RH use their derived channel name there
        # (a no-op with the supported prediction=False routing).
        prognostic_names = self.prognostic_names
        if self._saturation_normalization is not None:
            prognostic_names = self._saturation_normalization.residual_names(
                prognostic_names
            )

        return step_with_adjustments(
            input=args.input,
            next_step_input_data=args.next_step_input_data,
            network_calls=network_call,
            normalizer=self.normalizer,
            corrector=self._corrector,
            ocean=self.ocean,
            residual_prediction=self._config.residual_prediction,
            prognostic_names=prognostic_names,
            prescribed_prognostic_names=self._config.prescribed_prognostic_names,
            saturation_normalization=self._saturation_normalization,
            stepper_state=args.stepper_state,
        )

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def train(self, mode: bool = True) -> StepABC:
        super().train(mode)
        self._corrector.train(mode)
        return self

    def set_epoch(self, epoch: int) -> None:
        self._corrector.set_epoch(epoch)

    def get_state(self):
        state: dict[str, Any] = {"module": self.module.get_state()}
        corrector_state = self._corrector.get_state()
        if len(corrector_state) > 0:
            state["corrector"] = corrector_state
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        self.module.load_state(state["module"])
        self._corrector.load_state(state.get("corrector", {}))
