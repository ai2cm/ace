import copy
import dataclasses
import logging
from collections.abc import Callable
from typing import Protocol

import torch

from fme.core.ema import EMATracker
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.typing_ import Slice


class ValidateStepper(Protocol):
    def __call__(
        self,
        stepper: TrainStepperABC,
        ema: EMATracker,
        epoch: int,
    ) -> float:
        """Validate a stepper and return the validation loss."""
        ...


@dataclasses.dataclass
class LRTuningConfig:
    """
    Configuration for periodic learning rate tuning trials.

    At the start of epochs contained in ``epochs``, the trainer forks the
    current model into a baseline and a candidate copy. Both are trained for
    ``num_batches`` on the first batches of the epoch; the candidate uses a
    learning rate of ``current_lr * lr_factor``. Both are then validated. If
    the candidate's validation loss is less than the baseline's by at least
    ``improvement_threshold`` times the baseline's validation loss, the trainer
    adopts the candidate's learning rate.

    Parameters:
        epochs: A Slice selecting which epochs to run trials on. For example,
            ``Slice(start=1, step=2)`` runs at epochs 1, 3, 5, … (skipping
            epoch 0).
        lr_factor: Multiply the current LR by this to get the candidate LR.
        num_batches: Number of training batches for each fork in the trial.
        improvement_threshold: The candidate must beat the baseline's
            validation loss by at least this fraction of the baseline's
            validation loss (e.g. 0.01 means the candidate must be lower
            by at least 1% of the baseline loss).
    """

    lr_factor: float = 0.5
    num_batches: int = 200
    epochs: Slice = dataclasses.field(default_factory=Slice)
    improvement_threshold: float = 0.001


def run_lr_tuning_trial(
    train_data: GriddedDataABC,
    optimization: OptimizationABC,
    copy_stepper: Callable[[], TrainStepperABC],
    build_optimization: Callable[[torch.nn.ModuleList], OptimizationABC],
    copy_ema: Callable[[torch.nn.ModuleList], EMATracker],
    config: LRTuningConfig,
    current_lr: float,
    epoch: int,
    validate_stepper: ValidateStepper,
) -> float | None:
    """
    Run an isolated LR tuning trial comparing the current LR against a candidate.

    Creates two stepper forks, trains both, validates both, and compares
    validation loss improvements. Does not mutate the original stepper or
    optimization. Does not log to wandb.

    Args:
        train_data: Training data; ``subset_loader`` is used for the first N
            batches. The caller must have already called ``set_epoch``.
        optimization: The current optimization (used to copy momentum state
            into the forks).
        copy_stepper: Factory that returns a new stepper initialized from the
            current stepper's state. Called twice (baseline and candidate).
            The caller is responsible for ensuring proper deep copy semantics
            (e.g. using get_state/load_state rather than copy.deepcopy).
        build_optimization: Factory to build a fresh optimization for a
            given ModuleList.
        copy_ema: Factory that returns a new EMA tracker initialized from the
            current EMA state but tracking the given modules. Called twice.
        config: The LR tuning configuration.
        current_lr: The current learning rate.
        epoch: The epoch the trial is training for. Forwarded to
            ``validate_stepper`` so the validation data is advanced to the same
            epoch as the trial's training forks (e.g. so an epoch-based loss
            schedule selects the matching n_forward_steps distribution).
        validate_stepper: Callback that validates a stepper and returns the
            validation loss. Called twice (baseline and candidate).

    Returns:
        The candidate learning rate if the candidate wins, otherwise None.
    """
    candidate_lr = current_lr * config.lr_factor
    optimization_state = copy.deepcopy(optimization.get_state())

    baseline_stepper = copy_stepper()
    candidate_stepper = copy_stepper()

    baseline_opt = build_optimization(baseline_stepper.modules)
    baseline_opt.load_state(copy.deepcopy(optimization_state))
    baseline_opt.set_learning_rate(current_lr)

    candidate_opt = build_optimization(candidate_stepper.modules)
    candidate_opt.load_state(copy.deepcopy(optimization_state))
    candidate_opt.set_learning_rate(candidate_lr)

    baseline_ema = copy_ema(baseline_stepper.modules)
    candidate_ema = copy_ema(candidate_stepper.modules)

    baseline_stepper.set_train()
    candidate_stepper.set_train()
    for batch in train_data.subset_loader(stop_batch=config.num_batches):
        baseline_stepper.train_on_batch(batch, baseline_opt)
        baseline_ema(baseline_stepper.modules)

        candidate_stepper.train_on_batch(batch, candidate_opt)
        candidate_ema(candidate_stepper.modules)

    baseline_val_loss = validate_stepper(baseline_stepper, baseline_ema, epoch)
    candidate_val_loss = validate_stepper(candidate_stepper, candidate_ema, epoch)

    threshold = baseline_val_loss - config.improvement_threshold * baseline_val_loss

    logging.info(
        f"LR tuning trial: baseline LR={current_lr}, candidate LR={candidate_lr}, "
        f"baseline val loss={baseline_val_loss:.6f}, "
        f"candidate val loss={candidate_val_loss:.6f}, "
        f"threshold={threshold:.6f}"
    )

    if candidate_val_loss < threshold:
        logging.info(
            f"LR tuning trial: candidate wins "
            f"(candidate loss {candidate_val_loss:.6f} < "
            f"threshold {threshold:.6f})"
        )
        return candidate_lr

    logging.info("LR tuning trial: baseline wins, keeping current LR")
    return None
