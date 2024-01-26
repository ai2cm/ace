import pytest
import torch

from fme.downscaling.metrics_and_maths import (
    compute_psnr,
    compute_ssim,
    compute_zonal_power_spectrum,
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


@pytest.mark.parametrize("constant", [0, 1, -1])
def test_psnr_between_constants(constant):
    error = 2.0
    target = torch.full((1, 1, 4, 4), constant, dtype=torch.float32)
    pred = torch.full((1, 1, 4, 4), constant) + error
    expected = 80.0  # -10.0 * log10(ε), where ε = 1e-8
    actual = compute_psnr(pred, target)
    assert torch.isclose(actual, torch.tensor(expected))


@pytest.mark.parametrize("shape", [(1, 1, 16, 16), (2, 3, 16, 16)])
@pytest.mark.parametrize("metric", (compute_psnr, compute_ssim))
def test_shapes(shape, metric):
    target = torch.randn(shape)
    pred = torch.randn(shape)
    actual_shape = metric(pred, target).shape
    assert actual_shape == torch.Size([]), f"Expected scalar, got {actual_shape}"


@pytest.mark.parametrize("const", (1.0, 2.0))
def test_compute_zonal_power_spectrum_constant_value(const):
    batch_size, time_steps, nlat, nlon = 2, 1, 8, 16
    tensor = torch.full((2, 1, nlat, nlon), const, dtype=torch.float32)

    lats = torch.linspace(-89.5, 89.5, nlat)
    spectrum = compute_zonal_power_spectrum(tensor, lats)

    assert spectrum.shape == torch.Size((batch_size, time_steps, nlon // 2 + 1))

    spectrum[:, :, 0, ...]

    assert torch.all(spectrum[:, :, 0] != 0)
    assert torch.all(spectrum[:, :, 1:] == 0.0)
