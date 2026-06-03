import dataclasses
import enum
import logging
import random
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
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class TaskWeights:
    """Relative sampling weights and loss scaling for each task type.

    Sampling probabilities are derived from the weight values
    (normalized to sum to 1). Loss scaling is applied as a multiplier
    on the task step's loss.

    A weight of 0 disables a task entirely.
    """

    auto_encode: float = 1.0
    infill: float = 1.0
    prediction: float = 1.0
    infill_prediction: float = 1.0
    combined_all: float = 1.0

    auto_encode_loss_scale: float = 1.0
    infill_loss_scale: float = 1.0
    prediction_loss_scale: float = 1.0
    infill_prediction_loss_scale: float = 1.0
    combined_all_loss_scale: float = 1.0


@dataclasses.dataclass
class TaskSamplingConfig:
    """Configuration for random task selection during training.

    Attributes:
        task_weights: Sampling weights and loss scaling per task type.
        min_input_variables: Minimum number of input variables to select.
        min_output_variables: Minimum number of output variables to select.
    """

    task_weights: TaskWeights = dataclasses.field(default_factory=TaskWeights)
    min_input_variables: int = 1
    min_output_variables: int = 1

    def __post_init__(self):
        if self.min_output_variables < 1:
            raise ValueError(
                f"min_output_variables must be >= 1, got {self.min_output_variables}"
            )
        if self.min_input_variables < 1:
            raise ValueError(
                f"min_input_variables must be >= 1, got {self.min_input_variables}"
            )


@dataclasses.dataclass
class InferenceSchemeConfig:
    """Defines how the model behaves at inference time.

    This mirrors the variable routing of SingleModuleStepConfig for
    standard forward prediction. Future schemes could add post-hoc
    auto-encoding or iterative infill.

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


@dataclasses.dataclass
class SampledTasks:
    """Per-sample task assignments for one training batch.

    Attributes:
        previous_step_input_mask: Per-variable boolean masks of shape [batch].
            True where the variable should be taken from the previous-step
            model state.
        current_step_input_mask: Per-variable boolean masks of shape [batch].
            True where the variable should be taken from current-step ground
            truth.
        output_data_mask: Per-variable float masks of shape [batch]. Values
            are the task's loss_scale where the variable is a prediction
            target, 0.0 otherwise.
    """

    previous_step_input_mask: dict[str, torch.Tensor]
    current_step_input_mask: dict[str, torch.Tensor]
    output_data_mask: dict[str, torch.Tensor]


class TaskType(enum.Enum):
    AUTO_ENCODE = "auto_encode"
    INFILL = "infill"
    PREDICTION = "prediction"
    INFILL_PREDICTION = "infill_prediction"
    COMBINED_ALL = "combined_all"


def _get_task_weight(weights: TaskWeights, task: TaskType) -> float:
    return getattr(weights, task.value)


def _get_task_loss_scale(weights: TaskWeights, task: TaskType) -> float:
    return getattr(weights, f"{task.value}_loss_scale")


def _randint(low: int, high: int) -> int:
    """Sample uniformly from [low, high] inclusive."""
    return random.randint(low, high)


class TaskSampler:
    """Samples training tasks for each batch element.

    Built from TaskSamplingConfig + all_names + forcing_names. Each call
    to sample() returns per-sample masks for one batch.
    """

    def __init__(
        self,
        config: TaskSamplingConfig,
        all_names: list[str],
        forcing_names: list[str],
    ):
        self._config = config
        self._all_names = list(all_names)
        self._forcing_names = set(forcing_names)
        self._non_forcing_names = [n for n in all_names if n not in self._forcing_names]
        self._forcing_name_list = [n for n in all_names if n in self._forcing_names]

        task_weights: list[tuple[TaskType, float]] = []
        for task_type in TaskType:
            w = _get_task_weight(config.task_weights, task_type)
            if w < 0:
                raise ValueError(
                    f"Task weight for {task_type.value} must be >= 0, got {w}"
                )
            if w > 0:
                self._validate_task_feasibility(task_type)
                task_weights.append((task_type, w))

        if not task_weights:
            raise ValueError("At least one task must have a non-zero weight")

        total = sum(w for _, w in task_weights)
        self._task_types = [t for t, _ in task_weights]
        self._task_probabilities = [w / total for _, w in task_weights]

    def _validate_task_feasibility(self, task_type: TaskType) -> None:
        n_nf = len(self._non_forcing_names)
        n_all = len(self._all_names)
        n_forcing = len(self._forcing_name_list)
        min_in = self._config.min_input_variables
        min_out = self._config.min_output_variables

        if task_type == TaskType.AUTO_ENCODE:
            required_nf = max(min_in, min_out)
            if n_nf < required_nf:
                raise ValueError(
                    f"auto_encode requires at least {required_nf} non-forcing "
                    f"variables, but only {n_nf} available"
                )
        elif task_type == TaskType.INFILL:
            min_in_contested = max(0, min_in - n_forcing)
            if n_nf < min_out + min_in_contested:
                raise ValueError(
                    f"infill requires at least {min_out + min_in_contested} "
                    f"non-forcing variables (with {n_forcing} forcing available "
                    f"to cover inputs), but only {n_nf} available"
                )
        elif task_type == TaskType.PREDICTION:
            if n_nf < min_out:
                raise ValueError(
                    f"prediction requires at least {min_out} non-forcing "
                    f"variables for outputs, but only {n_nf} available"
                )
            if n_all < min_in:
                raise ValueError(
                    f"prediction requires at least {min_in} variables for "
                    f"inputs, but only {n_all} available"
                )
        elif task_type == TaskType.INFILL_PREDICTION:
            min_in_contested = max(0, min_in - n_forcing)
            if n_nf < min_out + min_in_contested:
                raise ValueError(
                    f"infill_prediction requires at least "
                    f"{min_out + min_in_contested} non-forcing variables, "
                    f"but only {n_nf} available"
                )
            if n_all < 1:
                raise ValueError(
                    "infill_prediction requires at least 1 variable for "
                    "previous-step inputs"
                )
        elif task_type == TaskType.COMBINED_ALL:
            if n_nf < min_out:
                raise ValueError(
                    f"combined_all requires at least {min_out} non-forcing "
                    f"variables for outputs, but only {n_nf} available"
                )
            if n_all < min_in:
                raise ValueError(
                    f"combined_all requires at least {min_in} variables for "
                    f"inputs, but only {n_all} available"
                )

    def sample(
        self,
        data_mask: TensorMapping | None,
        batch_size: int,
    ) -> SampledTasks:
        prev_mask: dict[str, list[bool]] = {n: [] for n in self._all_names}
        curr_mask: dict[str, list[bool]] = {n: [] for n in self._all_names}
        out_mask: dict[str, list[float]] = {n: [] for n in self._non_forcing_names}

        for i in range(batch_size):
            available_nf = self._get_available(self._non_forcing_names, data_mask, i)
            available_forcing = self._get_available(
                self._forcing_name_list, data_mask, i
            )
            available_all = available_nf + available_forcing

            task_type = self._sample_task_type()
            loss_scale = _get_task_loss_scale(self._config.task_weights, task_type)

            prev_inputs, curr_inputs, outputs = self._sample_assignment(
                task_type, available_nf, available_forcing, available_all
            )

            prev_set = set(prev_inputs)
            curr_set = set(curr_inputs)
            out_set = set(outputs)

            for name in self._all_names:
                prev_mask[name].append(name in prev_set)
                curr_mask[name].append(name in curr_set)
            for name in self._non_forcing_names:
                out_mask[name].append(loss_scale if name in out_set else 0.0)

        device = _get_device(data_mask)
        return SampledTasks(
            previous_step_input_mask={
                n: torch.tensor(v, dtype=torch.bool, device=device)
                for n, v in prev_mask.items()
            },
            current_step_input_mask={
                n: torch.tensor(v, dtype=torch.bool, device=device)
                for n, v in curr_mask.items()
            },
            output_data_mask={
                n: torch.tensor(v, dtype=torch.float, device=device)
                for n, v in out_mask.items()
            },
        )

    def _get_available(
        self,
        names: list[str],
        data_mask: TensorMapping | None,
        sample_idx: int,
    ) -> list[str]:
        if data_mask is None:
            return list(names)
        return [n for n in names if n not in data_mask or data_mask[n][sample_idx]]

    def _sample_task_type(self) -> TaskType:
        (result,) = random.choices(
            self._task_types, weights=self._task_probabilities, k=1
        )
        return result

    def _sample_assignment(
        self,
        task_type: TaskType,
        available_nf: list[str],
        available_forcing: list[str],
        available_all: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Returns (previous_step_inputs, current_step_inputs, outputs)."""
        min_in = self._config.min_input_variables
        min_out = self._config.min_output_variables

        if task_type == TaskType.AUTO_ENCODE:
            return self._sample_auto_encode(available_nf, min_in, min_out)
        elif task_type == TaskType.INFILL:
            return self._sample_infill(available_nf, available_forcing, min_in, min_out)
        elif task_type == TaskType.PREDICTION:
            return self._sample_prediction(available_nf, available_all, min_in, min_out)
        elif task_type == TaskType.INFILL_PREDICTION:
            return self._sample_infill_prediction(
                available_nf, available_forcing, available_all, min_in, min_out
            )
        else:
            assert task_type == TaskType.COMBINED_ALL
            return self._sample_combined_all(
                available_nf, available_all, min_in, min_out
            )

    def _sample_auto_encode(
        self,
        available_nf: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str], list[str]]:
        n = _randint(max(min_in, min_out), len(available_nf))
        selected = random.sample(available_nf, n)
        return [], selected, selected

    def _sample_disjoint_with_forcing(
        self,
        available_nf: list[str],
        available_forcing: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str]]:
        """Shared logic for tasks with disjoint current-step inputs/outputs.

        Returns (current_step_inputs, outputs).
        """
        n_additional_in = _randint(0, len(available_forcing))
        min_in_contested = max(0, min_in - n_additional_in)
        n_total = _randint(min_out + min_in_contested, len(available_nf))
        n_out = _randint(min_out, n_total - min_in_contested)
        participants = random.sample(available_nf, n_total)
        outputs = participants[:n_out]
        contested_inputs = participants[n_out:]
        forcing_inputs = random.sample(available_forcing, n_additional_in)
        current_inputs = contested_inputs + forcing_inputs
        return current_inputs, outputs

    def _sample_infill(
        self,
        available_nf: list[str],
        available_forcing: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str], list[str]]:
        current_inputs, outputs = self._sample_disjoint_with_forcing(
            available_nf, available_forcing, min_in, min_out
        )
        return [], current_inputs, outputs

    def _sample_prediction(
        self,
        available_nf: list[str],
        available_all: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str], list[str]]:
        n_out = _randint(min_out, len(available_nf))
        n_in = _randint(min_in, len(available_all))
        outputs = random.sample(available_nf, n_out)
        prev_inputs = random.sample(available_all, n_in)
        return prev_inputs, [], outputs

    def _sample_infill_prediction(
        self,
        available_nf: list[str],
        available_forcing: list[str],
        available_all: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str], list[str]]:
        current_inputs, outputs = self._sample_disjoint_with_forcing(
            available_nf, available_forcing, min_in, min_out
        )
        n_prev = _randint(1, len(available_all))
        prev_inputs = random.sample(available_all, n_prev)
        return prev_inputs, current_inputs, outputs

    def _sample_combined_all(
        self,
        available_nf: list[str],
        available_all: list[str],
        min_in: int,
        min_out: int,
    ) -> tuple[list[str], list[str], list[str]]:
        n_out = _randint(min_out, len(available_nf))
        n_in = _randint(min_in, len(available_all))
        outputs = random.sample(available_nf, n_out)
        inputs = random.sample(available_all, n_in)
        prev_inputs: list[str] = []
        curr_inputs: list[str] = []
        for name in inputs:
            role = random.randint(0, 2)
            if role == 0:
                prev_inputs.append(name)
            elif role == 1:
                curr_inputs.append(name)
            else:
                prev_inputs.append(name)
                curr_inputs.append(name)
        return prev_inputs, curr_inputs, outputs


def _get_device(data_mask: TensorMapping | None) -> torch.device:
    if data_mask is not None:
        for v in data_mask.values():
            return v.device
    return torch.device("cpu")


@StepSelector.register("infill_prediction")
@dataclasses.dataclass
class InfillPredictionStepConfig(StepConfigABC):
    """Configuration for a step that trains on multiple tasks per batch.

    At inference time, behaves like SingleModuleStepConfig using the
    inference_scheme's variable routing. During training, all variables
    in ``all_names`` are used, with task-specific masking.

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
    def from_state(cls, state) -> "InfillPredictionStepConfig":
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
    """Step that supports multiple training tasks via variable masking.

    Uses all_names for input channels (with channel mask indicators)
    and non-forcing names for output channels.
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

        self._inference_in_names = set(config.inference_scheme.in_names)
        self._all_names_set = set(config.all_names)

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
    ) -> TensorDict:
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
