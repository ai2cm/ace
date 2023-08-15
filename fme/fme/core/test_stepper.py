from unittest.mock import MagicMock

import pytest
import torch

import fme
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.optimization import OptimizationConfig
from fme.core.packer import Packer
from fme.core.prescriber import NullPrescriber, Prescriber, PrescriberConfig
from fme.core.stepper import (
    SingleModuleStepper,
    SingleModuleStepperConfig,
    run_on_batch,
)
from fme.fcn_training.registry import ModuleSelector


def get_data(names, n_samples, n_time):
    data = {}
    n_lat, n_lon = 5, 5
    lats = torch.linspace(-89.5, 89.5, n_lat)  # arbitary choice
    for name in names:
        data[name] = torch.rand(
            n_samples, n_time, n_lat, n_lon, device=fme.get_device()
        )
    area_weights = fme.spherical_area_weights(lats, n_lon).to(fme.get_device())
    return data, area_weights


def get_scalar_data(names, value):
    return {n: torch.tensor([value], device=fme.get_device()) for n in names}


@pytest.mark.parametrize(
    "in_names,out_names,prescriber_config,expected_all_names",
    [
        (["a"], ["b"], None, ["a", "b"]),
        (["a"], ["a", "b"], None, ["a", "b"]),
        (["a", "b"], ["b"], None, ["a", "b"]),
        (["a", "b"], ["a", "b"], None, ["a", "b"]),
        (["a", "b"], ["a", "b"], PrescriberConfig("a", "mask", 0), ["a", "b", "mask"]),
        (["a", "b"], ["a", "b"], PrescriberConfig("a", "b", 0), ["a", "b"]),
    ],
)
def test_stepper_config_all_names_property(
    in_names, out_names, prescriber_config, expected_all_names
):
    config = SingleModuleStepperConfig(
        builder=MagicMock(),
        in_names=in_names,
        out_names=out_names,
        normalization=MagicMock(),
        optimization=MagicMock(),
        prescriber=prescriber_config,
    )
    # check there are no duplications
    assert len(config.all_names) == len(set(config.all_names))
    # check the right items are in there using sets to ignore order
    assert set(config.all_names) == set(expected_all_names)


def test_run_on_batch_normalizer_changes_only_norm_data():
    torch.manual_seed(0)
    data, _ = get_data(["a", "b"], n_samples=5, n_time=2)
    stepped = run_on_batch(
        data=data,
        module=torch.nn.Identity(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=1,
        aggregator=MagicMock(),
    )
    assert torch.allclose(
        stepped.gen_data["a"], stepped.gen_data_norm["a"]
    )  # as std=1, mean=0, no change
    stepped_double_std = run_on_batch(
        data=data,
        module=torch.nn.Identity(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 2.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=1,
        aggregator=MagicMock(),
    )
    assert torch.allclose(
        stepped.gen_data["a"], stepped_double_std.gen_data["a"], rtol=1e-4
    )
    assert torch.allclose(
        stepped.gen_data["a"], 2.0 * stepped_double_std.gen_data_norm["a"], rtol=1e-4
    )
    assert torch.allclose(
        stepped.target_data["a"],
        2.0 * stepped_double_std.target_data_norm["a"],
        rtol=1e-4,
    )
    assert torch.allclose(
        stepped.loss, 4.0 * stepped_double_std.loss, rtol=1e-4
    )  # mse scales with std**2


def test_run_on_batch_addition_series():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data, _ = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1)
    stepped = run_on_batch(
        data=data,
        module=AddOne(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b"], 0.0),
            stds=get_scalar_data(["a", "b"], 1.0),
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        prescriber=NullPrescriber(),
        n_forward_steps=n_steps,
        aggregator=MagicMock(),
    )
    assert stepped.gen_data["a"].shape == (5, n_steps + 1, 5, 5)
    for i in range(n_steps - 1):
        assert torch.allclose(
            stepped.gen_data_norm["a"][:, i] + 1, stepped.gen_data_norm["a"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data_norm["b"][:, i] + 1, stepped.gen_data_norm["b"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data["a"][:, i] + 1, stepped.gen_data["a"][:, i + 1]
        )
        assert torch.allclose(
            stepped.gen_data["b"][:, i] + 1, stepped.gen_data["b"][:, i + 1]
        )
    assert torch.allclose(stepped.target_data_norm["a"], data["a"])
    assert torch.allclose(stepped.target_data_norm["b"], data["b"])
    assert torch.allclose(stepped.gen_data["a"][:, 0], data["a"][:, 0])
    assert torch.allclose(stepped.gen_data["b"][:, 0], data["b"][:, 0])
    assert torch.allclose(
        stepped.gen_data_norm["a"][:, 0], stepped.target_data_norm["a"][:, 0]
    )
    assert torch.allclose(
        stepped.gen_data_norm["b"][:, 0], stepped.target_data_norm["b"][:, 0]
    )


def test_run_on_batch_with_prescriber():
    torch.manual_seed(0)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 3
    data, _ = get_data(["a", "b", "mask"], n_samples=5, n_time=n_steps + 1)
    data["mask"] = torch.zeros_like(data["mask"], dtype=torch.int)
    data["mask"][:, :, :, 0] = 1
    stds = {
        "a": torch.tensor([2.0], device=fme.get_device()),
        "b": torch.tensor([3.0], device=fme.get_device()),
        "mask": torch.tensor([1.0], device=fme.get_device()),
    }
    stepped = run_on_batch(
        data=data,
        module=AddOne(),
        normalizer=StandardNormalizer(
            means=get_scalar_data(["a", "b", "mask"], 0.0),
            stds=stds,
        ),
        in_packer=Packer(["a", "b"]),
        out_packer=Packer(["a", "b"]),
        optimization=MagicMock(),
        loss_obj=torch.nn.MSELoss(),
        n_forward_steps=n_steps,
        prescriber=Prescriber("b", "mask", 1),
        aggregator=MagicMock(),
    )
    for i in range(n_steps):
        # "a" should be increasing by 1 according to AddOne
        torch.testing.assert_close(
            stepped.gen_data_norm["a"][:, i] + 1, stepped.gen_data_norm["a"][:, i + 1]
        )
        # "b" should be increasing by 1 where the mask says don't prescribe
        # note the 1: selection for the last dimension in following two assertions
        torch.testing.assert_close(
            stepped.gen_data_norm["b"][:, i, :, 1:] + 1,
            stepped.gen_data_norm["b"][:, i + 1, :, 1:],
        )
        # now check that the 0th index in last dimension has been overwritten
        torch.testing.assert_close(
            stepped.gen_data_norm["b"][:, i, :, 0],
            stepped.target_data_norm["b"][:, i, :, 0],
        )


def test_reloaded_stepper_gives_same_prediction():
    torch.manual_seed(0)
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(
            type="FourierNeuralOperatorNet", config={"scale_factor": 1}
        ),
        in_names=["a", "b"],
        out_names=["a", "b"],
        optimization=OptimizationConfig(
            optimizer_type="Adam",
            lr=0.001,
            enable_automatic_mixed_precision=False,
        ),
        normalization=NormalizationConfig(
            means={"a": 0.0, "b": 0.0},
            stds={"a": 1.0, "b": 1.0},
        ),
    )
    shapes = {
        "a": (1, 1, 5, 5),
        "b": (1, 1, 5, 5),
    }
    stepper = config.get_stepper(
        shapes=shapes,
        max_epochs=1,
    )
    new_stepper = SingleModuleStepper.from_state(stepper.get_state())
    data, _ = get_data(["a", "b"], n_samples=5, n_time=2)
    first_result = stepper.run_on_batch(
        data=data,
        train=False,
        n_forward_steps=1,
    )
    second_result = new_stepper.run_on_batch(
        data=data,
        train=False,
        n_forward_steps=1,
    )
    assert torch.allclose(first_result.loss, second_result.loss)
    assert torch.allclose(first_result.gen_data["a"], second_result.gen_data["a"])
    assert torch.allclose(first_result.gen_data["b"], second_result.gen_data["b"])
    assert torch.allclose(
        first_result.gen_data_norm["a"], second_result.gen_data_norm["a"]
    )
    assert torch.allclose(
        first_result.gen_data_norm["b"], second_result.gen_data_norm["b"]
    )
    assert torch.allclose(
        first_result.target_data_norm["a"], second_result.target_data_norm["a"]
    )
    assert torch.allclose(
        first_result.target_data_norm["b"], second_result.target_data_norm["b"]
    )
    assert torch.allclose(first_result.target_data["a"], second_result.target_data["a"])
    assert torch.allclose(first_result.target_data["b"], second_result.target_data["b"])
