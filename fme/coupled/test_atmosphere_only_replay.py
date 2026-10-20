"""Prototype tests for atmosphere-only replay batches in coupled training
(AtmosphereOnlyTrainingConfig)."""

import torch

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.core.loss import StepLossConfig
from fme.core.optimization import NullOptimization
from fme.coupled.stepper import (
    AtmosphereOnlyTrainingConfig,
    ComponentTrainingConfig,
    CoupledTrainStepperConfig,
)
from fme.coupled.test_stepper import get_data, get_train_stepper_and_batch


def _train_stepper(coupled_batches_update: str):
    config = CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=ComponentTrainingConfig(loss=StepLossConfig(type="MSE"), n_steps=1),
        atmosphere=ComponentTrainingConfig(loss=StepLossConfig(type="MSE"), n_steps=2),
        atmosphere_only=AtmosphereOnlyTrainingConfig(
            fraction=0.5,
            n_steps=TimeLengthProbabilities(outcomes=[TimeLengthProbability(2, 1.0)]),
            coupled_batches_update=coupled_batches_update,  # type: ignore[arg-type]
        ),
    )
    return get_train_stepper_and_batch(
        train_stepper_config=config,
        ocean_in_names=["sst"],
        ocean_out_names=["sst"],
        atmosphere_in_names=["surface_temperature", "ocean_fraction"],
        atmosphere_out_names=["surface_temperature"],
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=2,
    )


def test_atmosphere_only_batch_trains_atmosphere_alone():
    train_stepper, _, _, _ = _train_stepper("both")
    atmos_batch = get_data(
        ["surface_temperature", "ocean_fraction"],
        n_samples=2,
        n_time=3,
        realm="atmosphere",
    ).data
    out = train_stepper.train_on_batch(atmos_batch, NullOptimization())
    assert out.ocean.metrics == {} and out.ocean.gen_data == {}
    metrics = out.get_metrics()
    assert "loss" in metrics and "loss/atmosphere" in metrics
    assert "loss/ocean" not in metrics
    assert torch.isfinite(metrics["loss"])
    assert "surface_temperature" in out.atmosphere.gen_data


def test_coupled_batch_still_works_and_can_be_ocean_only():
    for mode, expect_atmos_loss in (("both", True), ("ocean", False)):
        train_stepper, coupled_data, _, _ = _train_stepper(mode)
        out = train_stepper.train_on_batch(coupled_data.data, NullOptimization())
        metrics = out.get_metrics()
        assert "loss/ocean" in metrics and "loss/atmosphere" in metrics
        if expect_atmos_loss:
            assert metrics["loss/atmosphere"] > 0
        else:
            assert metrics["loss/atmosphere"] == 0
            assert metrics["loss/ocean"] > 0


def test_anchor_penalty_is_added_on_coupled_batches():
    from fme.core.optimization import OptimizationConfig
    from fme.core.registry.module import ModuleSelector

    class Atmos(torch.nn.Module):
        # in: [surface_temperature, ocean_fraction] -> out: [surface_temperature]
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 1, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    config = CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=ComponentTrainingConfig(loss=StepLossConfig(type="MSE"), n_steps=1),
        atmosphere=ComponentTrainingConfig(loss=StepLossConfig(type="MSE"), n_steps=2),
        anchor_atmosphere_weights=1.0,
    )
    train_stepper, coupled_data, _, _ = get_train_stepper_and_batch(
        train_stepper_config=config,
        ocean_in_names=["sst"],
        ocean_out_names=["sst"],
        atmosphere_in_names=["surface_temperature", "ocean_fraction"],
        atmosphere_out_names=["surface_temperature"],
        n_forward_times_ocean=1,
        n_forward_times_atmosphere=2,
        n_samples=2,
        atmosphere_builder=ModuleSelector(type="prebuilt", config={"module": Atmos()}),
    )
    optimization = OptimizationConfig(lr=1e-2).build(
        train_stepper.modules, max_epochs=1
    )
    out = train_stepper.train_on_batch(coupled_data.data, optimization)
    key = [k for k in out.get_metrics() if "anchor_penalty" in k][0]
    assert out.get_metrics()[key] == 0  # first batch: weights still at their anchor
    out2 = train_stepper.train_on_batch(coupled_data.data, optimization)
    assert out2.get_metrics()[key] > 0  # weights moved, the penalty is paid
    assert torch.isfinite(out2.get_metrics()["loss"])
