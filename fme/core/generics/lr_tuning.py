import copy
import dataclasses
import logging
from collections.abc import Callable

import torch

from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.generics.validation import run_validation_loop
from fme.core.typing_ import Slice


@dataclasses.dataclass
class LRTuningConfig:
    """
    Configuration for periodic learning rate tuning trials.

    At the start of epochs contained in ``epochs``, the trainer forks the
    current model into a baseline and a candidate copy. Both are trained for
    ``num_batches`` on the first batches of the epoch; the candidate uses a
    learning rate of ``current_lr * lr_factor``. Both are then validated. If
    the candidate's validation loss is less than the baseline's by at least
    ``improvement_threshold`` times the pre-trial validation loss, the trainer
    adopts the candidate's learning rate.

    Parameters:
        epochs: A Slice selecting which epochs to run trials on. For example,
            ``Slice(start=1, step=2)`` runs at epochs 1, 3, 5, … (skipping
            epoch 0).
        lr_factor: Multiply the current LR by this to get the candidate LR.
        num_batches: Number of training batches for each fork in the trial.
        improvement_threshold: The candidate must beat the baseline's
            validation loss by at least this fraction of the pre-trial
            validation loss (e.g. 0.01 means the candidate must be lower
            by at least 1% of the pre-trial loss).
    """

    lr_factor: float
    num_batches: int
    epochs: Slice = dataclasses.field(default_factory=Slice)
    improvement_threshold: float = 0.0


def run_lr_tuning_trial(
    train_data: GriddedDataABC,
    valid_data: GriddedDataABC,
    optimization: OptimizationABC,
    copy_stepper: Callable[[], TrainStepperABC],
    build_optimization: Callable[[torch.nn.ModuleList], OptimizationABC],
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
        build_optimization: Factory to build a fresh optimization for a
            given ModuleList.
        copy_ema: Factory that returns a new EMA tracker initialized from the
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

    # Train both forks
    baseline_stepper.set_train()
    candidate_stepper.set_train()
    for batch in train_data.subset_loader(stop_batch=config.num_batches):
        baseline_stepper.train_on_batch(batch, baseline_opt)
        baseline_ema(baseline_stepper.modules)

        candidate_stepper.train_on_batch(batch, candidate_opt)
        candidate_ema(candidate_stepper.modules)

    # Validate both forks
    baseline_agg = get_validation_aggregator()
    run_validation_loop(
        stepper=baseline_stepper,
        valid_data=valid_data,
        aggregator=baseline_agg,
        ema=baseline_ema,
        validate_using_ema=validate_using_ema,
    )
    baseline_val_logs = baseline_agg.get_logs(label="val")

    candidate_agg = get_validation_aggregator()
    run_validation_loop(
        stepper=candidate_stepper,
        valid_data=valid_data,
        aggregator=candidate_agg,
        ema=candidate_ema,
        validate_using_ema=validate_using_ema,
    )
    candidate_val_logs = candidate_agg.get_logs(label="val")

    baseline_val_loss = baseline_val_logs["val/mean/loss"]
    candidate_val_loss = candidate_val_logs["val/mean/loss"]

    threshold = baseline_val_loss - config.improvement_threshold * pre_trial_val_loss

    logging.info(
        f"LR tuning trial: baseline LR={current_lr}, candidate LR={candidate_lr}, "
        f"pre-trial val loss={pre_trial_val_loss:.6f}, "
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
