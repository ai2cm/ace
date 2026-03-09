import dataclasses
import logging
from collections.abc import Callable

import torch

from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.generics.trainer import run_validation
from fme.core.optimization import Optimization


@dataclasses.dataclass
class LRTuningConfig:
    """
    Configuration for periodic learning rate tuning trials.

    At the start of every ``epoch_frequency`` epochs, the trainer forks the
    current model into a baseline and a candidate copy. Both are trained for
    ``num_batches`` on the first batches of the epoch; the candidate uses a
    learning rate of ``current_lr * lr_factor``. Both are then validated. If
    both improve the validation loss and the candidate improves it by more than
    ``improvement_threshold`` fractionally more than the baseline, the trainer
    adopts the candidate's learning rate.

    Parameters:
        epoch_frequency: Run a trial every N epochs.
        lr_factor: Multiply the current LR by this to get the candidate LR.
        num_batches: Number of training batches for each fork in the trial.
        improvement_threshold: The candidate must improve validation loss by
            at least this fraction more than the baseline (e.g. 0.1 means
            the candidate must improve at least 10% more).
    """

    epoch_frequency: int
    lr_factor: float
    num_batches: int
    improvement_threshold: float


def run_lr_tuning_trial(
    train_data: GriddedDataABC,
    valid_data: GriddedDataABC,
    optimization: Optimization,
    copy_stepper: Callable[[], TrainStepperABC],
    build_optimization: Callable[[torch.nn.ModuleList], Optimization],
    copy_ema: Callable[[torch.nn.ModuleList], EMATracker],
    config: LRTuningConfig,
    current_lr: float,
    pre_trial_val_loss: float,
    get_validation_aggregator: Callable[[], AggregatorABC],
    validate_using_ema: bool,
) -> float | None:
    """
    Run an isolated LR tuning trial comparing the current LR against a candidate.

    Creates two stepper forks, trains both, validates both, and compares
    validation loss improvements. Does not mutate the original stepper or
    optimization. Does not log to wandb.

    Args:
        train_data: Training data; ``subset_loader`` is used for the first N
            batches. The caller must have already called ``set_epoch``.
        valid_data: Validation data.
        optimization: The current optimization (used to copy momentum state
            into the forks).
        copy_stepper: Factory that returns a new stepper initialized from the
            current stepper's state. Called twice (baseline and candidate).
            The caller is responsible for ensuring proper deep copy semantics
            (e.g. using get_state/load_state rather than copy.deepcopy).
        build_optimization: Factory to build a fresh Optimization for a
            given ModuleList.
        copy_ema: Factory that returns a new EMATracker initialized from the
            current EMA state but tracking the given modules. Called twice.
        config: The LR tuning configuration.
        current_lr: The current learning rate.
        pre_trial_val_loss: Validation loss from the end of the previous epoch.
        get_validation_aggregator: Factory for validation aggregators.
        validate_using_ema: Whether to use EMA parameters during validation.

    Returns:
        The candidate learning rate if the candidate wins, otherwise None.
    """
    candidate_lr = current_lr * config.lr_factor
    optimization_state = optimization.get_state()

    baseline_stepper = copy_stepper()
    candidate_stepper = copy_stepper()

    baseline_opt = build_optimization(baseline_stepper.modules)
    baseline_opt.load_state(optimization_state)
    baseline_opt.set_learning_rate(current_lr)

    candidate_opt = build_optimization(candidate_stepper.modules)
    candidate_opt.load_state(optimization_state)
    candidate_opt.set_learning_rate(candidate_lr)

    baseline_ema = copy_ema(baseline_stepper.modules)
    candidate_ema = copy_ema(candidate_stepper.modules)

    # Train both forks
    baseline_stepper.set_train()
    candidate_stepper.set_train()
    for batch in train_data.subset_loader(stop_batch=config.num_batches):
        baseline_stepper.train_on_batch(batch, baseline_opt)
        baseline_ema(baseline_stepper.modules)

        candidate_stepper.train_on_batch(batch, candidate_opt)
        candidate_ema(candidate_stepper.modules)

    # Validate both forks
    baseline_val_logs = run_validation(
        stepper=baseline_stepper,
        valid_data=valid_data,
        aggregator=get_validation_aggregator(),
        ema=baseline_ema,
        validate_using_ema=validate_using_ema,
    )
    candidate_val_logs = run_validation(
        stepper=candidate_stepper,
        valid_data=valid_data,
        aggregator=get_validation_aggregator(),
        ema=candidate_ema,
        validate_using_ema=validate_using_ema,
    )

    baseline_val_loss = baseline_val_logs["val/mean/loss"]
    candidate_val_loss = candidate_val_logs["val/mean/loss"]

    baseline_improvement = pre_trial_val_loss - baseline_val_loss
    candidate_improvement = pre_trial_val_loss - candidate_val_loss

    logging.info(
        f"LR tuning trial: baseline LR={current_lr}, candidate LR={candidate_lr}, "
        f"pre-trial val loss={pre_trial_val_loss:.6f}, "
        f"baseline val loss={baseline_val_loss:.6f} "
        f"(improvement={baseline_improvement:.6f}), "
        f"candidate val loss={candidate_val_loss:.6f} "
        f"(improvement={candidate_improvement:.6f})"
    )

    if baseline_improvement > 0 and candidate_improvement > 0:
        threshold = baseline_improvement * (1 + config.improvement_threshold)
        if candidate_improvement > threshold:
            logging.info(
                f"LR tuning trial: candidate wins "
                f"(improvement {candidate_improvement:.6f} > "
                f"threshold {threshold:.6f})"
            )
            return candidate_lr

    logging.info("LR tuning trial: baseline wins, keeping current LR")
    return None
