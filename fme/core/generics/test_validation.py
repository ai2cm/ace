import torch

from fme.core.ema import EMATracker
from fme.core.generics.test_trainer import TrainData, TrainStepper, ValidationAggregator
from fme.core.generics.validation import run_validation_loop


def test_run_validation_loop():
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=3, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.5)
    ema = EMATracker(stepper.modules, decay=0.9999)

    run_validation_loop(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=ema,
        validate_using_ema=False,
    )

    assert stepper.validation_batches_seen == [0, 1, 2]
    logs = aggregator.get_logs(label="val")
    assert "val/mean/loss" in logs
    assert logs["val/mean/loss"] == 0.5


def test_run_validation_loop_with_ema():
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

    run_validation_loop(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=ema,
        validate_using_ema=True,
    )

    # After run_validation_loop, the original weights should be restored
    assert torch.allclose(stepper.modules[0].weight.data, weight_before)
    logs = aggregator.get_logs(label="val")
    assert "val/mean/loss" in logs


def test_run_validation_loop_without_ema():
    """When ema is None and validate_using_ema=False, validation still works."""
    stepper = TrainStepper()
    valid_data = TrainData(n_batches=2, shuffle=False)
    aggregator = ValidationAggregator(validation_loss=0.7)

    run_validation_loop(
        stepper=stepper,
        valid_data=valid_data,
        aggregator=aggregator,
        ema=None,
        validate_using_ema=False,
    )

    logs = aggregator.get_logs(label="val")
    assert logs["val/mean/loss"] == 0.7
