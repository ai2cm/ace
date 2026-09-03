import cftime
import numpy as np
import pytest
import xarray as xr

from fme.ace.inference.data_writer.monthly_aggregation import (
    MonthIndexer,
    add_data,
    find_boundary,
    get_valid_times,
)


def _days_since_january(valid_times: np.ndarray, first_year: int, calendar: str):
    """
    Re-express valid times as days since ``first_year``-01-01.

    ``get_valid_times`` returns the 15th of each month as days since
    1970-01-01; this removes both offsets so expectations can be written
    relative to the start of the first year.
    """
    reference_offset = (
        cftime.datetime(first_year, 1, 1, calendar=calendar)
        - cftime.datetime(1970, 1, 1, calendar=calendar)
    ).days
    return valid_times - 14 - reference_offset


@pytest.mark.parametrize("num_years", [2, 500])
@pytest.mark.parametrize("calendar", ["proleptic_gregorian", "noleap"])
def test_get_valid_times(num_years, calendar):
    first_year = 2020
    final_year = first_year + num_years - 1
    years = np.array([i for i in range(first_year, final_year + 1)])
    months = np.zeros((num_years,), dtype=int)
    # For last year set month to 1
    months[-1] = 1
    n_months = 3
    days = _days_since_january(
        get_valid_times(years, months, n_months, calendar), first_year, calendar
    )
    assert days.shape == (num_years, 3)
    # 2020 is a leap year in proleptic_gregorian
    if calendar == "proleptic_gregorian":
        assert days[0, 0] == 0
        assert days[0, 1] == 31
        assert days[0, 2] == 31 + 29
        if num_years == 2:
            assert days[1, 0] == 366 + 31
            assert days[1, 1] == 366 + 31 + 28
            assert days[1, 2] == 366 + 31 + 28 + 31
        if num_years == 500:
            # 121 is number of leap days
            assert days[499, 0] == 182135 + 121 + 31
            assert days[499, 1] == 182135 + 121 + 31 + 28
    if calendar == "noleap":
        assert days[0, 0] == 0
        assert days[0, 1] == 31
        assert days[0, 2] == 31 + 28
        if num_years == 2:
            assert days[1, 0] == 365 + 31
            assert days[1, 1] == 365 + 31 + 28
            assert days[1, 2] == 365 + 31 + 28 + 31
        if num_years == 500:
            assert days[499, 0] == 182135 + 31
            assert days[499, 1] == 182135 + 31 + 28


def test_get_valid_times_with_month_offset():
    calendar = "noleap"
    month_offset = 2
    offset_n_months = 3
    n_months = month_offset + offset_n_months

    years = np.array([2020, 2021])
    months = np.zeros((2,), dtype=int)

    full = get_valid_times(years, months, n_months, calendar)
    expected = full[:, month_offset:]
    result = get_valid_times(
        years,
        months,
        offset_n_months,
        calendar,
        month_offset=month_offset,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "month_array, expected",
    [
        pytest.param([1, 2, 3, 4, 5, 6], 1, id="linear"),
        pytest.param([1, 1, 2], 2, id="after two steps"),
        pytest.param([1], 1, id="one value"),
        pytest.param([1, 1, 1, 1], 4, id="all the same"),
        pytest.param([0] * 50 + [1] * (23), 50, id="long array case"),
    ],
)
def test_find_boundary(month_array, expected):
    assert (
        find_boundary(np.asarray(month_array), start_month=month_array[0]) == expected
    )


def test_add_data_one_first_month():
    target = np.zeros((2, 3))
    target_start_counts = np.zeros((2, 3), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32)
    expected = np.zeros((2, 3))
    expected[:, 0] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(target_start_counts, 0)


def test_add_data_one_first_month_averaging():
    target = np.zeros((2, 3))
    target[0, 0] = 2.0
    target_start_counts = np.zeros((2, 3), dtype=np.int32)
    target_start_counts[0, 0] = 1
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32)
    expected = np.zeros((2, 3))
    expected[0, 0] = (2 + 5) / 6
    expected[1, 0] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def test_add_data_one_later_month():
    target = np.zeros((2, 4))
    target_start_counts = np.zeros((2, 4), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32) + 2
    expected = np.zeros((2, 4))
    expected[:, 2] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def test_add_data_two_later_months():
    target = np.zeros((2, 4))
    target_start_counts = np.zeros((2, 4), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32) + 2
    months_elapsed[0, 2:] = 3
    months_elapsed[1, 3:] = 3
    expected = np.zeros((2, 4))
    expected[0, 2] = 1
    expected[0, 3] = 1
    expected[1, 2] = 1
    expected[1, 3] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def _batch_time(times: list[list[cftime.datetime]]) -> xr.DataArray:
    return xr.DataArray(times, dims=["sample", "time"])


def test_month_indexer_indexes_relative_to_each_samples_first_output_time():
    indexer = MonthIndexer(
        first_output_times=[
            cftime.DatetimeProlepticGregorian(2020, 1, 15),
            cftime.DatetimeProlepticGregorian(2020, 3, 15),
        ]
    )
    np.testing.assert_array_equal(indexer.init_years, [2020, 2020])
    np.testing.assert_array_equal(indexer.init_months, [0, 2])
    first = indexer.month_indices(
        _batch_time(
            [
                [cftime.DatetimeProlepticGregorian(2020, 1, 15)],
                [cftime.DatetimeProlepticGregorian(2020, 3, 15)],
            ]
        )
    )
    np.testing.assert_array_equal(first, [[0], [0]])
    # each sample's origin is its own first output time, so the same calendar
    # month maps to a different index per sample
    later = indexer.month_indices(
        _batch_time(
            [
                [cftime.DatetimeProlepticGregorian(2021, 2, 15)],
                [cftime.DatetimeProlepticGregorian(2021, 2, 15)],
            ]
        )
    )
    np.testing.assert_array_equal(later, [[13], [11]])


def test_month_indexer_indices_are_negative_before_month_zero():
    indexer = MonthIndexer(
        first_output_times=[cftime.DatetimeProlepticGregorian(2020, 3, 1)]
    )
    indices = indexer.month_indices(
        _batch_time([[cftime.DatetimeProlepticGregorian(2020, 1, 15)]])
    )
    np.testing.assert_array_equal(indices, [[-2]])


def test_month_indexer_n_months_through():
    indexer = MonthIndexer(
        first_output_times=[
            cftime.DatetimeProlepticGregorian(2020, 1, 15),
            cftime.DatetimeProlepticGregorian(2020, 3, 15),
        ]
    )
    final_times = [
        cftime.DatetimeProlepticGregorian(2020, 5, 1),
        cftime.DatetimeProlepticGregorian(2020, 8, 1),
    ]
    # sample 0 spans January through May (5), sample 1 March through August (6)
    assert indexer.n_months_through(final_times) == 6
