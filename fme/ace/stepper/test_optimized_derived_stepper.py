"""Stepper-level tests of optimized derived variables (rho_wright97 in the loss)."""

import dataclasses
import datetime
import unittest.mock

import dacite
import pytest
import torch

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.stepper.single_module import StepperConfig, TrainStepperConfig
from fme.core.coordinates import DepthCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.loss import StepLossConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.optimized_derived import OptimizedDerivedVariableConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.testing import trivial_network_and_loss_normalization

DEVICE = fme.get_device()
IMG_SHAPE = (4, 6)
N_LEVELS = 2
NAMES = [f"{v}_{k}" for v in ("so", "thetao") for k in range(N_LEVELS)]
RHO_NAMES = [f"rho_wright97_{k}" for k in range(N_LEVELS)]


class _AddBias(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x):
        return x + self.bias


def _dataset_info() -> DatasetInfo:
    mask = torch.ones(*IMG_SHAPE, N_LEVELS, device=DEVICE)
    mask[0] = 0.0  # a land row
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-60.0, 60.0, IMG_SHAPE[0], device=DEVICE),
            lon=torch.linspace(0.0, 300.0, IMG_SHAPE[1], device=DEVICE),
        ),
        vertical_coordinate=DepthCoordinate(
            idepth=torch.tensor([0.0, 10.0, 500.0], device=DEVICE), mask=mask
        ),
        timestep=datetime.timedelta(days=5),
    )


def _stepper_config(module: torch.nn.Module) -> StepperConfig:
    return StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(type="prebuilt", config={"module": module}),
                    in_names=NAMES,
                    out_names=NAMES,
                    normalization=trivial_network_and_loss_normalization(NAMES),
                    corrector=CorrectorSelector("ocean_corrector", {}),
                )
            ),
        ),
    )


def _data(n_timesteps: int = 3, seed: int = 0) -> BatchData:
    data = BatchData.new_for_testing(
        names=NAMES, n_samples=2, n_timesteps=n_timesteps, img_shape=IMG_SHAPE
    )
    g = torch.Generator().manual_seed(seed)
    shape = data.data[NAMES[0]].shape
    for k in range(N_LEVELS):
        for name, lo, span in ((f"so_{k}", 34.0, 2.0), (f"thetao_{k}", 2.0, 20.0)):
            data.data[name].copy_(lo + span * torch.rand(shape, generator=g))
    return data


def _train_stepper(module: torch.nn.Module, **train_config_kwargs):
    return TrainStepperConfig(**train_config_kwargs).get_train_stepper(
        _stepper_config(module), _dataset_info()
    )


def test_train_on_batch_with_rho():
    """Only rho_wright97 carries loss weight, so the parameter update is driven through
    the EOS."""
    torch.manual_seed(0)
    module = _AddBias(len(NAMES))
    stepper = _train_stepper(
        module,
        loss=StepLossConfig(type="MSE", weights={n: 0.0 for n in NAMES}),
        optimized_derived_variables=[OptimizedDerivedVariableConfig(weight=1.0)],
    )
    optimization = OptimizationConfig(lr=1e-2).build(
        modules=stepper.modules, max_epochs=1
    )
    stepped = stepper.train_on_batch(_data(), optimization=optimization)
    loss = stepped.metrics["loss"]
    assert torch.isfinite(loss) and loss > 0
    assert stepped.per_channel_losses is not None
    assert set(stepped.per_channel_losses) == set(NAMES + RHO_NAMES)
    for name in NAMES:
        assert stepped.per_channel_losses[name].loss == 0.0
    for name in RHO_NAMES:
        assert stepped.per_channel_losses[name].loss > 0.0
    # every so and thetao channel moved: the rho gradient reached all four
    (bias,) = [p for p in stepper.modules.parameters()]
    assert (bias.detach().flatten() != 0).all()
    for name in RHO_NAMES:
        assert name not in stepped.gen_data


def test_no_config_is_unchanged():
    """Absent and None configs give the same loss, only output-name channels,
    and a checkpoint state that does not mention the feature."""
    data = _data()
    module = _AddBias(len(NAMES))
    outputs, states = [], []
    for kwargs in (
        {},
        {"optimized_derived_variables": None},
        {"optimized_derived_variables": [OptimizedDerivedVariableConfig()]},
    ):
        torch.manual_seed(0)
        stepper = _train_stepper(module, **kwargs)
        outputs.append(stepper.train_on_batch(data, optimization=NullOptimization()))
        states.append(stepper.get_state())
    absent, none, on = outputs
    torch.testing.assert_close(absent.metrics["loss"], none.metrics["loss"])
    assert absent.per_channel_losses is not None
    assert set(absent.per_channel_losses) == set(NAMES)
    assert set(on.per_channel_losses or {}) == set(NAMES + RHO_NAMES)
    # repr, since the prebuilt module in each config is its own copy
    assert len({repr(state["config"]) for state in states}) == 1
    assert "optimized_derived" not in str(states[2])


def test_train_stepper_config_parses_from_yaml_dict():
    config = dacite.from_dict(
        data_class=TrainStepperConfig,
        data={
            "loss": {"type": "MSE"},
            "optimized_derived_variables": [
                {"name": "rho_wright97", "weight": 0.5, "levels": [0, 1]}
            ],
        },
        config=dacite.Config(strict=True),
    )
    assert config.optimized_derived_variables == [
        OptimizedDerivedVariableConfig(name="rho_wright97", weight=0.5, levels=[0, 1])
    ]
    with pytest.raises(dacite.WrongTypeError):
        dacite.from_dict(
            data_class=TrainStepperConfig,
            data={"optimized_derived_variables": [{"name": "sigma0"}]},
            config=dacite.Config(strict=True),
        )


def test_coupled_ocean_config_passed_to_ocean_build_loss():
    from fme.coupled.stepper import ComponentTrainingConfig, CoupledTrainStepperConfig

    derived = [OptimizedDerivedVariableConfig()]
    config = CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=ComponentTrainingConfig(
            loss=StepLossConfig(), optimized_derived_variables=derived
        ),
        atmosphere=ComponentTrainingConfig(loss=StepLossConfig()),
    )
    stepper = unittest.mock.Mock()
    stepper.n_inner_steps = 2
    config._build_loss(stepper, n_coupled_steps=1)
    stepper.ocean.build_loss.assert_called_once_with(config.ocean.loss, derived)
    stepper.atmosphere.build_loss.assert_called_once_with(config.atmosphere.loss, None)
