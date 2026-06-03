import dataclasses
import enum
import random

import torch

from fme.core.typing_ import TensorMapping


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
