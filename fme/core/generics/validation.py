import contextlib

import torch

from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.optimization import NullOptimization
from fme.core.timing import GlobalTimer


def run_validation(
    stepper: TrainStepperABC,
    valid_data: GriddedDataABC,
    aggregator: AggregatorABC,
    ema: EMATracker | None,
    validate_using_ema: bool,
) -> dict[str, float]:
    """
    Run validation on the given data and return logs.

    This is the core validation loop used by both the Trainer and
    LR tuning trials. It does NOT call aggregator.flush_diagnostics —
    the caller is responsible for flushing if needed.

    Args:
        stepper: The train stepper to evaluate.
        valid_data: The validation dataset.
        aggregator: The aggregator to record batch results into.
        ema: The EMA tracker, or None if EMA is not used.
        validate_using_ema: Whether to use EMA parameters during validation.

    Returns:
        Validation logs dict (e.g. {"val/mean/loss": ...}).
    """
    stepper.set_eval()
    ema_context: contextlib.AbstractContextManager = (
        ema.applied_params(stepper.modules)
        if validate_using_ema and ema is not None
        else contextlib.nullcontext()
    )
    with torch.no_grad(), ema_context, GlobalTimer():
        for batch in valid_data.loader:
            stepped = stepper.train_on_batch(
                batch,
                optimization=NullOptimization(),
                compute_derived_variables=True,
            )
            aggregator.record_batch(batch=stepped)
    return aggregator.get_logs(label="val")
