from fme.fcn_training.stepper import run_on_batch
import torch
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
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
