import pytest
import torch

from fme.downscaling.projection import block_mean, project_to_block_mean


def _lat_weights(n_lat: int) -> torch.Tensor:
    lat = torch.linspace(-80.0, 80.0, n_lat, dtype=torch.float64)
    return torch.cos(torch.deg2rad(lat))


def test_block_mean_matches_explicit_weighted_mean():
    torch.manual_seed(0)
    factor = 2
    x = torch.randn(3, 2, 4, 6, dtype=torch.float64)
    lat_weights = _lat_weights(4)

    result = block_mean(x, lat_weights, factor)

    assert result.shape == (3, 2, 2, 3)
    i, j = 1, 2
    block = x[..., i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
    w = lat_weights[i * factor : (i + 1) * factor, None]
    expected = (block * w).sum(dim=(-2, -1)) / (w.sum() * factor)
    torch.testing.assert_close(result[..., i, j], expected)


def test_project_to_block_mean_hits_target_and_keeps_anomalies():
    torch.manual_seed(0)
    factor = 4
    x = torch.randn(2, 3, 8, 16, dtype=torch.float64)
    target = torch.randn(2, 3, 2, 4, dtype=torch.float64)
    lat_weights = _lat_weights(8)

    projected = project_to_block_mean(x, target, lat_weights, factor)

    torch.testing.assert_close(block_mean(projected, lat_weights, factor), target)

    def anomaly(y):
        mean = block_mean(y, lat_weights, factor)
        upsampled = mean.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
        return y - upsampled

    torch.testing.assert_close(anomaly(projected), anomaly(x))
    torch.testing.assert_close(
        project_to_block_mean(projected, target, lat_weights, factor), projected
    )


def test_block_mean_raises_for_indivisible_shape():
    with pytest.raises(ValueError, match="divisible"):
        block_mean(torch.zeros(1, 5, 4), torch.ones(5), 2)
