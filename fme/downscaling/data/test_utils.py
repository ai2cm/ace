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


def test_adjust_fine_coord_range_raises_near_domain_boundary():
    downscale_factor = 4  # n_half_fine = 2
    coarse_edges = torch.linspace(0, 6, 7)
    coarse_lat = _fine_midpoints(coarse_edges, 1)
    fine_lat = _fine_midpoints(coarse_edges, downscale_factor)
    # Drop the first fine point so only 1 fine point exists below coarse_min=0.5,
    # but n_half_fine=2 are required — simulating a grid truncated at the domain edge.
    fine_lat_truncated = fine_lat[1:]
    with pytest.raises(ValueError):
        adjust_fine_coord_range(
            ClosedInterval(0, 4),
            full_coarse_coord=coarse_lat,
            full_fine_coord=fine_lat_truncated,
            downscale_factor=downscale_factor,
        )


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


@pytest.mark.parametrize(
    "interval,expected_slice",
    [
        pytest.param(ClosedInterval(2, 4), slice(2, 5), id="middle"),
        pytest.param(ClosedInterval(float("-inf"), 2), slice(0, 3), id="start_inf"),
        pytest.param(ClosedInterval(4, float("inf")), slice(4, 5), id="end_inf"),
        pytest.param(
            ClosedInterval(float("-inf"), float("inf")), slice(0, 5), id="all_inf"
        ),
    ],
)
def test_ClosedInterval_slice_from(interval, expected_slice):
    coords = torch.arange(5)
    result_slice = interval.slice_from(coords)
    assert result_slice == expected_slice


def test_ClosedInterval_fail_on_empty_slice():
    coords = torch.arange(5)
    with pytest.raises(ValueError):
        ClosedInterval(5.5, 7).slice_from(coords)


def test_ClosedInterval_subset_of():
    coords = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    result = ClosedInterval(1.0, 3.0).subset_of(coords)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))
