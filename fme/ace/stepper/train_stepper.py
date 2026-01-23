import contextlib
import dataclasses
from typing import Any

import torch
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.stepper.single_module import (
    Stepper,
    TrainOutput,
    process_ensemble_prediction_generator_list,
)
from fme.ace.stepper.time_length_probabilities import (
    TimeLength,
    TimeLengthProbabilities,
    TimeLengthSchedule,
)
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.loss import StepLoss, StepLossConfig
from fme.core.tensors import add_ensemble_dim, unfold_ensemble_dim
from fme.core.training_history import TrainingHistory, TrainingJob
from fme.core.typing_ import EnsembleTensorDict, TensorDict


class EpochNotProvidedError(ValueError):
    pass


def probabilities_from_time_length(value: TimeLength) -> TimeLengthProbabilities:
    if isinstance(value, TimeLengthProbabilities):
        return value
    else:
        return TimeLengthProbabilities.from_constant(value)


@dataclasses.dataclass
class TrainStepperConfig:
    """
    Configuration for training-specific aspects of a stepper.

    Parameters:
        loss: The loss configuration.
        optimize_last_step_only: Whether to optimize only the last step.
        n_ensemble: The number of ensemble members evaluated for each training
            batch member. Default is 2 if the loss type is EnsembleLoss, otherwise
            the default is 1. Must be 2 for EnsembleLoss to be valid.
        train_n_forward_steps: The number of timesteps to train on and associated
            sampling probabilities. By default, the stepper will train on the full
            number of timesteps present in the training dataset samples. Values must
            be less than or equal to the number of timesteps present
            in the training dataset samples.
    """

    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    optimize_last_step_only: bool = False
    n_ensemble: int = -1  # sentinel value to avoid None typing of attribute
    train_n_forward_steps: TimeLength | TimeLengthSchedule | None = None

    def __post_init__(self):
        if self.n_ensemble == -1:
            if self.loss.type == "EnsembleLoss":
                self.n_ensemble = 2
            else:
                self.n_ensemble = 1

    @property
    def train_n_forward_steps_schedule(self) -> TimeLengthSchedule | None:
        if self.train_n_forward_steps is None:
            return None
        if isinstance(self.train_n_forward_steps, TimeLengthSchedule):
            return self.train_n_forward_steps
        return TimeLengthSchedule.from_constant(self.train_n_forward_steps)


class TrainStepper(
    TrainStepperABC[
        PrognosticState,
        BatchData,
        BatchData,
        PairedData,
        TrainOutput,
    ]
):
    """
    Wrapper around Stepper that adds training functionality.

    This class composes a Stepper (for inference) with training-specific
    configuration and implements the train_on_batch method.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        stepper: Stepper,
        config: TrainStepperConfig,
        loss_obj: StepLoss,
    ):
        """
        Args:
            stepper: The underlying stepper for inference operations.
            config: Training-specific configuration.
            loss_obj: The loss function object for training.
        """
        self._stepper = stepper
        self._config = config
        self._loss_obj = loss_obj

        self._train_n_forward_steps_sampler: TimeLengthProbabilities | None = None
        self._train_n_forward_steps_schedule: TimeLengthSchedule | None = None
        if config.train_n_forward_steps_schedule is not None:
            self._train_n_forward_steps_schedule = config.train_n_forward_steps_schedule

        self._epoch: int | None = None  # to keep track of cached values

    # ---- Delegated inference properties and methods ----

    @property
    def modules(self) -> nn.ModuleList:
        return self._stepper.modules

    @property
    def n_ic_timesteps(self) -> int:
        return self._stepper.n_ic_timesteps

    @property
    def prognostic_names(self) -> list[str]:
        return self._stepper.prognostic_names

    @property
    def normalizer(self):
        return self._stepper.normalizer

    @property
    def loss_names(self) -> list[str]:
        return self._stepper.loss_names

    @property
    def derive_func(self):
        return self._stepper.derive_func

    @property
    def training_history(self) -> TrainingHistory:
        return self._stepper.training_history

    def predict(
        self,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
        compute_derived_forcings: bool = True,
    ) -> tuple[BatchData, PrognosticState]:
        return self._stepper.predict(
            initial_condition,
            forcing,
            compute_derived_variables,
            compute_derived_forcings,
        )

    def predict_paired(
        self,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[PairedData, PrognosticState]:
        return self._stepper.predict_paired(
            initial_condition, forcing, compute_derived_variables
        )

    def get_state(self) -> dict[str, Any]:
        return self._stepper.get_state()

    def load_state(self, state: dict[str, Any]) -> None:
        self._stepper.load_state(state)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TrainStepper":
        # Note: This method is primarily for interface compliance.
        # In practice, TrainStepper is created via TrainConfig.get_train_stepper()
        raise NotImplementedError(
            "TrainStepper.from_state is not directly supported. "
            "Use TrainConfig to create a TrainStepper from a checkpoint."
        )

    # ---- Training-specific methods ----

    @property
    def loss_obj(self) -> StepLoss:
        return self._loss_obj

    @property
    def effective_loss_scaling(self) -> TensorDict:
        """
        Effective loss scalings used to normalize outputs before computing loss.
        y_loss_normalized_i = (y_i - y_mean_i) / loss_scaling_i
        where loss_scaling_i = loss_normalizer_std_i / weight_i.
        """
        return self.loss_obj.effective_loss_scaling

    def _init_for_epoch(self, epoch: int | None):
        if (
            epoch is None
            and self._train_n_forward_steps_schedule is not None
            and len(self._train_n_forward_steps_schedule.milestones) > 0
        ):
            raise EpochNotProvidedError(
                "current configuration requires epoch to be provided "
                "on BatchData during training"
            )
        if self._epoch == epoch:
            return
        if self._train_n_forward_steps_schedule is not None:
            assert epoch is not None  # already checked, but needed for mypy
            self._train_n_forward_steps_sampler = probabilities_from_time_length(
                self._train_n_forward_steps_schedule.get_value(epoch)
            )
        else:
            self._train_n_forward_steps_sampler = None
        self._epoch = epoch

    def train_on_batch(
        self,
        data: BatchData,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TrainOutput:
        """
        Train the model on a batch of data with one or more forward steps.

        If gradient accumulation is used by the optimization, the computational graph is
        detached between steps to reduce memory consumption. This means the model learns
        how to deal with inputs on step N but does not try to improve the behavior at
        step N by modifying the behavior for step N-1.

        Args:
            data: The batch data where each tensor in data.data has shape
                [n_sample, n_forward_steps + self.n_ic_timesteps, <horizontal_dims>].
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            compute_derived_variables: Whether to compute derived variables for the
                prediction and target data.

        Returns:
            The loss metrics, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        self._init_for_epoch(data.epoch)
        metrics: dict[str, float] = {}
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        target_data = self._stepper.get_forward_data(
            data, compute_derived_variables=False
        )
        data = self._stepper._forcing_deriver(data)

        optimization.set_mode(self._stepper._step_obj.modules)
        output_list = self._accumulate_loss(
            input_data,
            data,
            target_data,
            optimization,
            metrics,
        )

        regularizer_loss = self._stepper._get_regularizer_loss()
        if torch.any(regularizer_loss > 0):
            optimization.accumulate_loss(regularizer_loss)
        metrics["loss"] = optimization.get_accumulated_loss().detach()
        optimization.step_weights()

        gen_data = process_ensemble_prediction_generator_list(output_list)

        stepped = TrainOutput(
            metrics=metrics,
            gen_data=gen_data,
            target_data=add_ensemble_dim(target_data.data),
            time=target_data.time,
            normalize=self.normalizer.normalize,
            derive_func=self.derive_func,
        )
        ic = data.get_start(
            set(data.data.keys()), self.n_ic_timesteps
        )  # full data and not just prognostic get prepended
        stepped = stepped.prepend_initial_condition(ic)
        if compute_derived_variables:
            stepped = stepped.compute_derived_variables()
        # apply post-processing and return
        return stepped

    def _accumulate_loss(
        self,
        input_data: PrognosticState,
        data: BatchData,
        target_data: BatchData,
        optimization: OptimizationABC,
        metrics: dict[str, float],
    ) -> list[EnsembleTensorDict]:
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        # output from self.predict_paired does not include initial condition
        n_forward_steps = data.time.shape[1] - self.n_ic_timesteps
        n_ensemble = self._config.n_ensemble
        input_batch_data = input_data.as_batch_data()
        if input_batch_data.labels != data.labels:
            raise ValueError(
                "Initial condition and forcing data must have the same labels, "
                f"got {input_batch_data.labels} and {data.labels}."
            )
        input_ensemble_data = input_data.as_batch_data().broadcast_ensemble(n_ensemble)
        forcing_ensemble_data = data.broadcast_ensemble(n_ensemble)
        output_generator = self._stepper._predict_generator(
            input_ensemble_data.data,
            forcing_ensemble_data.data,
            n_forward_steps,
            optimization,
            labels=input_ensemble_data.labels,
        )
        output_list: list[EnsembleTensorDict] = []
        output_iterator = iter(output_generator)
        if self._train_n_forward_steps_sampler is not None:
            stochastic_n_forward_steps = self._train_n_forward_steps_sampler.sample()
            if stochastic_n_forward_steps > n_forward_steps:
                raise RuntimeError(
                    "The number of forward steps to train on "
                    f"({stochastic_n_forward_steps}) is greater than the number of "
                    f"forward steps in the data ({n_forward_steps}), "
                    "This is supposed to be ensured by the TrainConfig when train "
                    "data requirements are retrieved, so this is a bug."
                )
            n_forward_steps = stochastic_n_forward_steps
        for step in range(n_forward_steps):
            optimize_step = (
                step == n_forward_steps - 1 or not self._config.optimize_last_step_only
            )
            if optimize_step:
                context = contextlib.nullcontext()
            else:
                context = torch.no_grad()
            with context:
                gen_step = next(output_iterator)
                gen_step = unfold_ensemble_dim(gen_step, n_ensemble=n_ensemble)
                output_list.append(gen_step)
                # Note: here we examine the loss for a single timestep,
                # not a single model call (which may contain multiple timesteps).
                target_step = add_ensemble_dim(
                    {
                        k: v.select(self.TIME_DIM, step)
                        for k, v in target_data.data.items()
                    }
                )
                step_loss = self.loss_obj(gen_step, target_step, step=step)
                metrics[f"loss_step_{step}"] = step_loss.detach()
            if optimize_step:
                optimization.accumulate_loss(step_loss)
        return output_list

    def update_training_history(self, training_job: TrainingJob) -> None:
        """
        Update the stepper's history of training jobs.

        Args:
            training_job: The training job to add to the history.
        """
        self._stepper.update_training_history(training_job)

    def get_base_weights(self):
        """Get the base weights of the underlying stepper."""
        return self._stepper.get_base_weights()
