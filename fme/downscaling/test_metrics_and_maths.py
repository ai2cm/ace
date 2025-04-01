import dataclasses
from typing import Set

import pytest
import torch

from fme.downscaling.metrics_and_maths import (
    compute_crps,
    compute_psnr,
    compute_ssim,
    compute_zonal_power_spectrum,
    filter_tensor_mapping,
    map_tensor_mapping,
    min_max_normalization,
)


def test_map_named_tensors():
    mapped_function = map_tensor_mapping(torch.add)
    tensors_1 = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    tensors_2 = {"a": torch.tensor(3.0), "b": torch.tensor(4.0)}

    # Test for correct function mapping
    result = mapped_function(tensors_1, tensors_2)
    assert torch.equal(result["a"], torch.tensor(4.0))
    assert torch.equal(result["b"], torch.tensor(6.0))

    # Test for mismatched keys
    with pytest.raises(ValueError):
        tensors_mismatched = {"c": torch.tensor(3.0)}
        mapped_function(tensors_1, tensors_mismatched)


@dataclasses.dataclass
class CRPSExperiment:
    name: str
    truth_amount: float
    random_amount: float


def test_crps():
    nx = 1
    ny = 1
    n_batch = 1000
    n_sample = 10
    truth_amount = 0.8
    random_amount = 0.5
    experiments = [
        CRPSExperiment("perfect", truth_amount, random_amount),
        CRPSExperiment("extra_variance", truth_amount, random_amount * 1.1),
        CRPSExperiment("less_variance", truth_amount, random_amount * 0.9),
        CRPSExperiment("deterministic", truth_amount, random_amount * 1e-5),
    ]
    torch.manual_seed(0)
    x_predictable = torch.rand(n_batch, 1, nx, ny)
    x = truth_amount * x_predictable + random_amount * torch.rand(n_batch, 1, nx, ny)
    crps_values = {}
    for experiment in experiments:
        x_sample = (
            experiment.truth_amount * x_predictable
            + experiment.random_amount * torch.rand(n_batch, n_sample, nx, ny)
        )
        crps_values[experiment.name] = compute_crps(x, x_sample).mean()
    assert crps_values["perfect"] < crps_values["extra_variance"]
    assert crps_values["perfect"] < crps_values["less_variance"]
    assert crps_values["extra_variance"] < crps_values["deterministic"]
    assert crps_values["less_variance"] < crps_values["deterministic"]


@pytest.mark.parametrize(
    "tensor, expected",
    [
        (torch.tensor([0.0, 0.5]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.5, 0.7]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-10.0, -1.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-2.0, -0.5]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-1.0, -0.5]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-0.8, -0.5]), torch.tensor([0.0, 1.0])),
        (
            torch.tensor([-2.0, 0.0, 2.0, 4.0]),
            torch.tensor([0.0, 1.0 / 3, 2.0 / 3, 1.0]),
        ),
        (torch.tensor([0.0, 0.0]), torch.tensor([0.5, 0.5])),
    ],
    ids=[
        "0.0_0.5",
        "0.0_1.0",
        "0.5_0.7",
        "-10.0_-1.0",
        "-2.0_-0.5",
        "-1.0_-0.5",
        "-0.8_-0.5",
        "-2.0_0.0_2.0_4.0",
        "0.0_0.0_0.0_0.0",
    ],
)
def test_normalize_to_unit_range(tensor, expected):
    normalized_tensor = min_max_normalization(tensor, tensor.min(), tensor.max())
    assert all(torch.isclose(normalized_tensor, expected))


@pytest.mark.parametrize(
    "constant",
    [pytest.param(0, id="c=0"), pytest.param(1, id="c=1"), pytest.param(-1, id="c=-1")],
)
@pytest.mark.parametrize(
    "shape,add_channel_dim",
    [
        pytest.param((1, 1, 4, 4), None, id="b,c,h,w"),
        pytest.param((1, 4, 4), True, id="b,h,w"),
    ],
)
def test_psnr_between_constants(constant, shape, add_channel_dim):
    error = 2.0
    target = torch.full(shape, constant, dtype=torch.float32)
    pred = torch.full(shape, constant) + error
    expected = 80.0  # -10.0 * log10(ε), where ε = 1e-8
    actual = compute_psnr(pred, target, add_channel_dim=add_channel_dim)
    assert torch.isclose(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    "shape,add_channel_dim",
    [((1, 1, 16, 16), False), ((2, 3, 16, 16), False), ((2, 16, 16), True)],
)
@pytest.mark.parametrize("metric", (compute_psnr, compute_ssim))
def test_shapes(shape, add_channel_dim, metric):
    target = torch.randn(shape)
    pred = torch.randn(shape)
    actual_shape = metric(pred, target, add_channel_dim=add_channel_dim).shape
    assert actual_shape == torch.Size([]), f"Expected scalar, got {actual_shape}"


@pytest.mark.parametrize("const", (1.0, 2.0))
def test_compute_zonal_power_spectrum_constant_value(const):
    batch_size, nlat, nlon = 2, 8, 16
    tensor = torch.full((2, nlat, nlon), const, dtype=torch.float32)

    spectrum = compute_zonal_power_spectrum(tensor)

    assert spectrum.shape == torch.Size((batch_size, nlon // 2 + 1))
    assert torch.all(spectrum[:, 0] != 0)
    assert torch.all(spectrum[:, 1:] == 0.0)

    # Check that non 2D input fails
    with pytest.raises(ValueError):
        compute_zonal_power_spectrum(torch.ones(5))


def test_filter_tensor_mapping():
    tensor_mapping = {
        "a": torch.tensor(1.0),
        "b": torch.tensor(2.0),
        "c": torch.tensor(3.0),
        "d": torch.tensor(4.0),
    }
    keys = {"a", "c"}

    result = filter_tensor_mapping(tensor_mapping, keys)

    assert len(result) == 2
    for k in ("a", "c"):
        assert k in result
    for k in ("b", "d"):
        assert k not in result

    # Test with empty keys set
    empty_keys: Set[str] = set()
    result = filter_tensor_mapping(tensor_mapping, empty_keys)
    assert len(result) == 0

    # Test with non-existent keys
    non_existent_keys = {"e", "f"}
    result = filter_tensor_mapping(tensor_mapping, non_existent_keys)
    assert len(result) == 0
