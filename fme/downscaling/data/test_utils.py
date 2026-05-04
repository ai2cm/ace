import numpy as np
import pytest
import torch

from fme.downscaling.data.utils import (
    ClosedInterval,
    adjust_fine_coord_range,
    compute_lon_roll,
    paired_shuffle,
    roll_lon_coords,
    roll_lon_data,
    scale_slice,
)


def _fine_midpoints(coarse_edges, downscale_factor):
    fine_mids = []
    for start, end in zip(coarse_edges[:-1], coarse_edges[1:]):
        fine_edges = torch.linspace(start, end, downscale_factor + 1)
        mids = (fine_edges[:-1] + fine_edges[1:]) / 2
        fine_mids.append(mids)
    return torch.concatenate(fine_mids)


def _one_deg_lon_coords():
    """0.5, 1.5, ..., 359.5 — standard 1-degree global grid."""
    return torch.arange(0.5, 360.0, 1.0)


@pytest.mark.parametrize(
    "lon_start, expected_roll",
    [
        pytest.param(0.0, 0, id="zero_no_roll"),
        pytest.param(10.0, 0, id="positive_in_range_no_roll"),
        pytest.param(-5.0, 355, id="negative_start_rolls"),
        pytest.param(-0.5, 359, id="just_negative_rolls"),
        pytest.param(-180.0, 180, id="half_period"),
        pytest.param(360.0, 0, id="exactly_360_no_roll"),
    ],
)
def test_compute_lon_roll(lon_start, expected_roll):
    coords = _one_deg_lon_coords()
    assert compute_lon_roll(coords, lon_start) == expected_roll


def test_roll_lon_coords_negative_start():
    """Rolled coords for lon_start=-5 should be monotone and start near -5."""
    coords = _one_deg_lon_coords()
    roll_amount = compute_lon_roll(coords, -5.0)
    rolled = roll_lon_coords(coords, roll_amount, -5.0)

    assert rolled.shape == coords.shape
    # monotonically increasing
    assert torch.all(rolled[1:] > rolled[:-1])
    # first element is the first coord >= 355, shifted to negative convention
    assert rolled[0].item() == pytest.approx(-4.5)
    # last element of the original-convention "low" portion, shifted by -360
    assert rolled[-1].item() == pytest.approx(354.5)


def test_roll_lon_coords_zero_roll_returns_original():
    coords = _one_deg_lon_coords()
    result = roll_lon_coords(coords, 0, 0.0)
    assert torch.equal(result, coords)


def test_roll_lon_data_shifts_correctly():
    """Rolling data by r positions moves index r to index 0."""
    n = 8
    tensor = torch.arange(n, dtype=torch.float).unsqueeze(0)  # shape (1, 8)
    roll_amount = 3
    rolled = roll_lon_data(tensor, roll_amount, lon_dim=-1)
    assert rolled.shape == tensor.shape
    assert rolled[0, 0].item() == pytest.approx(3.0)  # original index 3 → 0
    assert rolled[0, -1].item() == pytest.approx(2.0)  # original index 2 → last


def test_roll_lon_data_zero_roll_returns_original():
    tensor = torch.randn(4, 8)
    result = roll_lon_data(tensor, 0)
    assert torch.equal(result, tensor)


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


def test_adjust_fine_coord_range_wrapping_lon():
    """
    adjust_fine_coord_range handles negative-start lon intervals (wrapping domain).
    """
    downscale_factor = 2  # n_half_fine = 1
    # Coarse 1-degree global grid (0.5 to 359.5), subset and already rolled
    # to shifted convention: coarse domain -2 to 3 (= 358 to 3 in 0-360)
    coarse_lon = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    # Fine 0.5-degree global grid in 0-360 convention (0.25 to 359.75)
    fine_lon = torch.arange(0.25, 360.0, 0.5)
    lon_range = ClosedInterval(-2.5, 3.5)

    result = adjust_fine_coord_range(
        lon_range,
        full_coarse_coord=coarse_lon,
        full_fine_coord=fine_lon,
        downscale_factor=downscale_factor,
    )
    # result should cover from 1 fine step below coarse_min=-2 to 1 fine step above
    # coarse_max=3
    # coarse_min=-2 → 358 in 0-360; fine just below 358 is 357.75 → shifted = -2.25
    # coarse_max=3; fine just above 3 is 3.25
    assert result.start == pytest.approx(-2.25)
    assert result.stop == pytest.approx(3.25)

    # Verify the returned interval properly covers the coarse points with
    # n_half_fine padding
    subsel_coarse = coarse_lon[
        (coarse_lon >= lon_range.start) & (coarse_lon <= lon_range.stop)
    ]
    assert len(subsel_coarse) == len(coarse_lon)  # all 6 coarse points selected

    # The fine interval spans n_coarse * downscale_factor fine points
    # After rolling to match the shifted convention, subset should be contiguous
    from fme.downscaling.data.utils import compute_lon_roll, roll_lon_coords

    fine_roll = compute_lon_roll(fine_lon, result.start)
    rolled_fine = roll_lon_coords(fine_lon, fine_roll, result.start)
    subsel_fine = result.subset_of(rolled_fine)
    assert len(subsel_fine) == len(subsel_coarse) * downscale_factor


def test_adjust_fine_coord_range_negative_lat_not_confused_with_wrapping():
    """Southern hemisphere lat (negative coarse_min) must not trigger lon-wrap logic."""
    downscale_factor = 2
    coarse_lat = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    # Fine lat grid also spans negative values — NOT a 0-360 lon grid
    fine_lat = torch.arange(-3.25, 3.5, 0.5)
    lat_range = ClosedInterval(-3.5, 3.5)
    result = adjust_fine_coord_range(
        lat_range,
        full_coarse_coord=coarse_lat,
        full_fine_coord=fine_lat,
        downscale_factor=downscale_factor,
    )
    # Standard non-wrapping behaviour: fine_min should be below coarse_min=-3
    assert result.start < -3.0
    assert result.stop > 3.0


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


@pytest.mark.parametrize(
    "interval,expected_values",
    [
        pytest.param(ClosedInterval(2, 4), torch.tensor([2, 3, 4]), id="middle"),
        pytest.param(
            ClosedInterval(float("-inf"), 2), torch.tensor([0, 1, 2]), id="start_inf"
        ),
        pytest.param(ClosedInterval(4, float("inf")), torch.tensor([4]), id="end_inf"),
        pytest.param(
            ClosedInterval(float("-inf"), float("inf")),
            torch.arange(5),
            id="all_inf",
        ),
    ],
)
def test_ClosedInterval_subset_of(interval, expected_values):
    coords = torch.arange(5)
    result = interval.subset_of(coords)
    assert torch.equal(result, expected_values)
