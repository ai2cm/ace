import numpy as np
import pytest
import torch

from fme.downscaling.data.utils import (
    ClosedInterval,
    adjust_fine_coord_range,
    paired_shuffle,
    scale_slice,
)


def _fine_midpoints(coarse_edges, downscale_factor):
    fine_mids = []
    for start, end in zip(coarse_edges[:-1], coarse_edges[1:]):
        fine_edges = torch.linspace(start, end, downscale_factor + 1)
        mids = (fine_edges[:-1] + fine_edges[1:]) / 2
        fine_mids.append(mids)
    return torch.concatenate(fine_mids)


def test_paired_shuffle():
    a = np.arange(5)
    b = a * 10
    a_shuffled, b_shuffled = paired_shuffle(list(a), list(b))
    for ai, bi in zip(a_shuffled, b_shuffled):
        assert ai * 10 == bi


@pytest.mark.parametrize(
    "downscale_factor, lat_range",
    [
        (2, ClosedInterval(1, 6)),
        (3, ClosedInterval(1, 6)),
        (4, ClosedInterval(0.7, 4)),
        (3, ClosedInterval(0.7, 4)),
    ],
)
def test_adjust_fine_coord_range(downscale_factor, lat_range):
    coarse_edges = torch.linspace(0, 6, 7)
    coarse_lat = _fine_midpoints(coarse_edges, 1)
    fine_lat = _fine_midpoints(coarse_edges, downscale_factor)
    new_lat_range = adjust_fine_coord_range(
        lat_range, full_coarse_coord=coarse_lat, full_fine_coord=fine_lat
    )
    subsel_fine_lat = torch.tensor([lat for lat in fine_lat if lat in new_lat_range])
    subsel_coarse_lat = torch.tensor([lat for lat in coarse_lat if lat in lat_range])
    assert len(subsel_fine_lat) % len(subsel_coarse_lat) == 0
    assert len(subsel_fine_lat) / len(subsel_coarse_lat) == downscale_factor


@pytest.mark.parametrize(
    "input_slice,expected",
    [
        pytest.param(slice(None), slice(None), id="none"),
        pytest.param(slice(None, 5), slice(None, 10), id="start_none"),
        pytest.param(slice(3, None), slice(6, None), id="stop_none"),
        pytest.param(slice(2, 4), slice(4, 8), id="both"),
    ],
)
def test_scale_slice(input_slice, expected):
    scaled = scale_slice(input_slice, scale=2)
    assert scaled.start == expected.start
    assert scaled.stop == expected.stop
