import pytest
import torch

from fme.core.models.miles_credit.boundary_padding import TensorPadding


@pytest.mark.parametrize("pad_lat", [(1, 0), (0, 2), (0, 0)])
def test_earth_padding_allows_one_sided_or_zero_latitude_padding(
    pad_lat: tuple[int, int],
):
    x = torch.arange(12).reshape(1, 1, 3, 4)
    padding = TensorPadding(mode="earth", pad_lat=pad_lat, pad_lon=(0, 0))

    padded = padding.pad(x)

    expected_lat_size = x.shape[-2] + pad_lat[0] + pad_lat[1]
    assert padded.shape == (*x.shape[:-2], expected_lat_size, x.shape[-1])
    torch.testing.assert_close(padding.unpad(padded), x)
