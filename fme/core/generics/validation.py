import contextlib
import logging
from collections.abc import Callable
from typing import TypeVar

import torch

from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.optimization import NullOptimization
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

PS = TypeVar("PS")  # prognostic state
BD = TypeVar("BD")  # batch data
FD = TypeVar("FD")  # forcing data
SD = TypeVar("SD")  # stepped data
TO = TypeVar("TO", bound=TrainOutputABC)  # train output


def run_validation_loop(
    stepper: TrainStepperABC,
    valid_data: GriddedDataABC,
    aggregator: AggregatorABC,
    ema: EMATracker | None = None,
    validate_using_ema: bool = False,
    compute_derived_variables: bool = True,
    log_progress: bool = False,
) -> None:
    """Run the core validation loop: iterate batches and record to aggregator.

    This is the minimal validation loop. It does NOT call `aggregator.get_logs`,
    `aggregator.flush_diagnostics`, or log to WandB — callers are responsible
    for those.

    Args:
        stepper: The train stepper to evaluate.
        valid_data: The validation dataset.
        aggregator: The aggregator to record batch results into.
        ema: The EMA tracker, or None if EMA is not used.
        validate_using_ema: Whether to use EMA parameters during validation.
        compute_derived_variables: Whether to compute derived variables.
        log_progress: Whether to log per-batch progress messages.
    """
    timer = GlobalTimer.get_instance()
    stepper.set_eval()
    ema_context: contextlib.AbstractContextManager = (
        ema.applied_params(stepper.modules)
        if validate_using_ema and ema is not None
        else contextlib.nullcontext()
    )
    no_opt = NullOptimization()
    n_batches = len(valid_data.loader)
    with torch.no_grad(), ema_context:
        for i, batch in enumerate(valid_data.loader):
            if log_progress:
                logging.info(f"Validation: processing batch {i + 1} of {n_batches}.")
            stepped = stepper.train_on_batch(
                batch,
                optimization=no_opt,
                compute_derived_variables=compute_derived_variables,
            )
            with timer.context("aggregator"):
                aggregator.record_batch(batch=stepped)


def _get_record_to_wandb() -> Callable[[dict[str, float]], None]:
    wandb = WandB.get_instance()

    def record_logs(logs: dict[str, float]) -> None:
        if len(logs) > 0:
            wandb.log(logs, step=0)

    return record_logs


def run_validation(
    train_stepper: TrainStepperABC[PS, BD, FD, SD, TO],
    validation_data: GriddedDataABC[BD],
    aggregator: AggregatorABC[TO],
    *,
    label: str = "val",
    diagnostics_subdir: str | None = None,
    record_logs: Callable[[dict[str, float]], None] | None = None,
    compute_derived_variables: bool = True,
    ema: EMATracker | None = None,
    validate_using_ema: bool = False,
    log_progress: bool = False,
) -> dict[str, float]:
    """Run validation loop for a train stepper and validation dataset.

    High-level wrapper around `run_validation_loop` that also flushes
    diagnostics, collects logs, and optionally records them to WandB.

    Args:
        train_stepper: Train-stepper wrapper to compute validation outputs.
        validation_data: Validation dataset with a loader of batches.
        aggregator: Aggregator for collecting and reducing validation metrics.
        label: Label to pass through to the aggregator for log key prefixes.
        diagnostics_subdir: Optional subdirectory for diagnostic output.
        record_logs: Optional function to record logs (defaults to WandB).
        compute_derived_variables: Whether to compute derived variables.
        ema: The EMA tracker, or None if EMA is not used.
        validate_using_ema: Whether to use EMA parameters during validation.
        log_progress: Whether to log per-batch progress messages.

    Returns:
        Dictionary of validation metrics (keys prefixed by the label).
    """
    if record_logs is None:
        record_logs = _get_record_to_wandb()

    timer = GlobalTimer.get_instance()

    logging.info("Starting validation loop")

    run_validation_loop(
        stepper=train_stepper,
        valid_data=validation_data,
        aggregator=aggregator,
        ema=ema,
        validate_using_ema=validate_using_ema,
        compute_derived_variables=compute_derived_variables,
        log_progress=log_progress,
    )

    logging.info("Flushing validation diagnostics")
    with timer.context("flush_diagnostics"):
        aggregator.flush_diagnostics(subdir=diagnostics_subdir)

    logging.info("Getting validation aggregator logs")
    with timer.context("aggregator"):
        val_logs = aggregator.get_logs(label=label)

    with timer.context("wandb_logging"):
        record_logs(val_logs)

    logging.info("Validation complete")
    return val_logs
