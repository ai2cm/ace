import unittest.mock

import torch

from fme.core.ema import EMATracker
from fme.core.generics.test_trainer import TrainData, TrainStepper, ValidationAggregator
from fme.core.generics.validation import run_validation


def test_run_validation_returns_logs():
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=3, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.5)
    ema = EMATracker(stepper.modules, decay=0.9999)

    logs = run_validation(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=ema,
        validate_using_ema=False,
    )

    assert "val/mean/loss" in logs
    assert logs["val/mean/loss"] == 0.5
    assert stepper.validation_batches_seen == [0, 1, 2]


def test_run_validation_with_ema():
    """When validate_using_ema=True, EMA params are applied during validation."""
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=2, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.3)

    # Set a non-zero weight so EMA differs from the initial zero weight
    stepper.modules[0].weight.data.fill_(1.0)
    ema = EMATracker(stepper.modules, decay=0.5)
    # Update EMA with current params, then change model weight
    ema(stepper.modules)
    stepper.modules[0].weight.data.fill_(2.0)

    weight_before = stepper.modules[0].weight.data.clone()

    logs = run_validation(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=ema,
        validate_using_ema=True,
    )

    # After run_validation, the original weights should be restored
    assert torch.allclose(stepper.modules[0].weight.data, weight_before)
    assert "val/mean/loss" in logs


def test_run_validation_without_ema_none():
    """When ema is None and validate_using_ema=False, validation still works."""
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=2, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.7)

    logs = run_validation(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=None,
        validate_using_ema=False,
    )

    assert logs["val/mean/loss"] == 0.7


def test_run_validation_does_not_flush_diagnostics():
    """run_validation should not call flush_diagnostics on the aggregator."""
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=2, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.1)
    aggregator.flush_diagnostics = unittest.mock.MagicMock()  # type: ignore

    run_validation(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=None,
        validate_using_ema=False,
    )

    aggregator.flush_diagnostics.assert_not_called()
