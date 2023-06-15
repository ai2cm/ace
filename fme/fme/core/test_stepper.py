from fme.core.stepper import (
    SingleModuleStepper,
    run_on_batch,
    SingleModuleStepperConfig,
)
from fme.fcn_training.registry import ModuleSelector
import torch
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.packer import Packer
from fme.core.optimization import OptimizationConfig
import fme
from unittest.mock import MagicMock


def get_data(names, n_samples, n_time):
    data = {}
    for name in names:
        data[name] = torch.rand(n_samples, n_time, 5, 5, device=fme.get_device())
    return data


def get_scalar_data(names, value):
    return {n: torch.tensor([value], device=fme.get_device()) for n in names}


def test_run_on_batch_normalizer_changes_only_norm_data():
    data = get_data(["a", "b"], n_samples=5, n_time=2)
    loss, gen_data, gen_data_norm, full_data_norm = run_on_batch(
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
        n_forward_steps=1,
        aggregator=MagicMock(),
    )
    assert torch.allclose(
        gen_data["a"], gen_data_norm["a"]
    )  # as std=1, mean=0, no change
    (
        loss_double_std,
        gen_data_double_std,
        gen_data_norm_double_std,
        full_data_norm_double_std,
    ) = run_on_batch(
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
        n_forward_steps=1,
        aggregator=MagicMock(),
    )
    assert torch.allclose(gen_data["a"], gen_data_double_std["a"])
    assert torch.allclose(gen_data["a"], 2.0 * gen_data_norm_double_std["a"])
    assert torch.allclose(full_data_norm["a"], 2.0 * full_data_norm_double_std["a"])
    assert torch.allclose(loss, 4.0 * loss_double_std)  # mse scales with std**2


def test_run_on_batch_addition_series():
    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    n_steps = 4
    data = get_data(["a", "b"], n_samples=5, n_time=n_steps + 1)
    _, gen_data, gen_data_norm, full_data_norm = run_on_batch(
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
        n_forward_steps=n_steps,
        aggregator=MagicMock(),
    )
    assert gen_data["a"].shape == (5, n_steps + 1, 5, 5)
    for i in range(n_steps - 1):
        assert torch.allclose(
            gen_data_norm["a"][:, i] + 1, gen_data_norm["a"][:, i + 1]
        )
        assert torch.allclose(
            gen_data_norm["b"][:, i] + 1, gen_data_norm["b"][:, i + 1]
        )
        assert torch.allclose(gen_data["a"][:, i] + 1, gen_data["a"][:, i + 1])
        assert torch.allclose(gen_data["b"][:, i] + 1, gen_data["b"][:, i + 1])
    assert torch.allclose(full_data_norm["a"], data["a"])
    assert torch.allclose(full_data_norm["b"], data["b"])
    assert torch.allclose(gen_data["a"][:, 0], data["a"][:, 0])
    assert torch.allclose(gen_data["b"][:, 0], data["b"][:, 0])
    assert torch.allclose(gen_data_norm["a"][:, 0], full_data_norm["a"][:, 0])
    assert torch.allclose(gen_data_norm["b"][:, 0], full_data_norm["b"][:, 0])


def test_reloaded_stepper_gives_same_prediction():
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
    data = get_data(["a", "b"], n_samples=5, n_time=2)
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
    assert torch.allclose(first_result[0], second_result[0])
    for i in range(1, 4):
        assert torch.allclose(first_result[i]["a"], second_result[i]["a"]), i
        assert torch.allclose(first_result[i]["b"], second_result[i]["b"]), i
