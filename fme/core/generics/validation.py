import logging
from collections.abc import Callable
from typing import TypeVar

import torch

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
) -> dict[str, float]:
    """Run validation loop for a train stepper and validation dataset.

    Args:
        train_stepper: Train-stepper wrapper to compute validation outputs.
        validation_data: Validation dataset with a loader of batches.
        aggregator: Aggregator for collecting and reducing validation metrics.
        label: Label to pass through to the aggregator for log key prefixes.
        diagnostics_subdir: Optional subdirectory for diagnostic output.
        record_logs: Optional function to record logs (defaults to WandB).
        compute_derived_variables: Whether to compute derived variables.

    Returns:
        Dictionary of validation metrics (keys prefixed by the label).
    """
    if record_logs is None:
        record_logs = _get_record_to_wandb()

    timer = GlobalTimer.get_instance()

    logging.info("Starting validation loop")

    with timer.context("validation"):
        no_opt = NullOptimization()
        n_batches = len(validation_data.loader)
        with torch.no_grad():
            for i, batch in enumerate(validation_data.loader):
                logging.info(f"Validation: processing batch {i + 1} of {n_batches}.")
                stepped = train_stepper.train_on_batch(
                    batch,
                    optimization=no_opt,
                    compute_derived_variables=compute_derived_variables,
                )
                aggregator.record_batch(stepped)

        logging.info("Flushing validation diagnostics")
        aggregator.flush_diagnostics(subdir=diagnostics_subdir)
        logging.info("Getting validation aggregator logs")
        val_logs = aggregator.get_logs(label=label)
        record_logs(val_logs)

    logging.info("Validation complete")
    return val_logs
