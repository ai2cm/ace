import datetime
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations

from .enso_coefficient import OVERLAP_THRESHOLD, EnsoCoefficientEvaluatorAggregator


@contextmanager
def change_aggregator_enso_index(
    aggregator: type[EnsoCoefficientEvaluatorAggregator], new_index: xr.DataArray
):
    """
    Temporarily change the class attribute enso index for testing purposes.
    """
    old_index = aggregator._enso_index
    aggregator._enso_index = new_index
    try:
        yield
    finally:
        aggregator._enso_index = old_index


def _data_generator(scale: int, steps: int):
    return torch.linspace(-scale, scale, steps)


def _get_data(
    scale: int,
    n_samples: int,
    n_times: int,
    n_lat: int,
    n_lon: int,
    calendar: str = "julian",
):
    time = xr.date_range(
        start="2000-01-01",
        periods=n_times,
        freq="6h",
        calendar=calendar,
        use_cftime=True,
    )
    enso_index = xr.DataArray(
        _data_generator(scale, n_times), dims=["time"], coords={"time": time}
    )
    target_data = {  # make target data perfectly correlated
        "a": torch.tile(
            _data_generator(scale, n_times)[None, :, None, None],
            [n_samples, 1, n_lat, n_lon],
        ).to(device=get_device()),
    }
    gen_data = {  # make gen data perfectly anti-correlated
        "a": torch.tile(
            _data_generator(-scale, n_times)[None, :, None, None],
            [n_samples, 1, n_lat, n_lon],
        ).to(device=get_device()),
    }
    sample_time = xr.concat(
        [
            xr.DataArray(
                time.values,
                dims=["time"],
            )
            for _ in range(n_samples)
        ],
        dim="sample",
    )
    return enso_index, sample_time, target_data, gen_data


@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0])
def test_enso_coefficient_aggregator_values(scaling):
    """
    Test the EnsoCoefficientEvaluatorAggregator class.

    Check that:
        - The aggregator maintains a zero-mean subset of the enso index for
            each sample.
        - The logged target and gen coefficients are scaled versions of 1.0 and
            -1.0 for perfectly correlated and anti-correlated data,respectively.
        - The logged global-mean coefficient RMSE is scaled from 2.0 (perfect
            correlation minus perfect anti-correlation).
        - The above two are also true via the `get_dataset` method.

    Args:
        scaling: How much to scale up or down the target and generated data;
            a scaling of 1.0 means that the target/gen data have the same scale
            as the enso index so the regression coefficients should be |1.0|.


    """
    n_samples, n_times, n_lat, n_lon = 2, 28, 3, 3
    scale = 3
    area_weights = torch.ones([n_lat, n_lon], device=get_device())
    # get data that doesn't vary in space, but varies in time with the ENSO index
    enso_index, sample_time, target_data, gen_data = _get_data(
        scale, n_samples, n_times, n_lat, n_lon
    )
    with change_aggregator_enso_index(EnsoCoefficientEvaluatorAggregator, enso_index):
        enso_agg = EnsoCoefficientEvaluatorAggregator(
            initial_time=sample_time.isel(time=0),
            n_forward_timesteps=(n_times - 1),
            timestep=datetime.timedelta(hours=6),
            gridded_operations=LatLonOperations(area_weights),
        )
    assert len(enso_agg._sample_index_series) == n_samples
    for index_values in enso_agg._sample_index_series:
        assert index_values is not None
        # check that the index values are zero-mean for each sample
        assert np.isclose(index_values.mean().item(), 0.0)
    target_data["a"] *= scaling
    gen_data["a"] *= scaling
    enso_agg.record_batch(time=sample_time, target_data=target_data, gen_data=gen_data)
    enso_agg.record_batch(time=sample_time, target_data=target_data, gen_data=gen_data)
    coefficients = enso_agg._get_coefficients()
    target_coefficients, gen_coefficients = coefficients
    assert target_coefficients is not None
    assert gen_coefficients is not None
    # check that the target coefficients are 1.0 * scaling (perfectly correlated)
    assert torch.allclose(target_coefficients["a"], torch.tensor(scaling))
    # check that the gen coefficients are -1.0 * scaling (perfectly anti-correlated)
    assert torch.allclose(gen_coefficients["a"], torch.tensor(-1.0 * scaling))
    logs = enso_agg.get_logs("enso_coefficients")
    # check that the global-mean coefficient RMSE is abs(-1.0 - 1.0)*scaling,
    # since the target and gen data are perfectly correlated and anti-correlated, resp.
    np.testing.assert_almost_equal(
        logs["enso_coefficients/rmse/a"], 2.0 * scaling, decimal=5
    )
    enso_dataset = enso_agg.get_dataset()
    # check that the coefficients are as expected in the dataset also
    np.testing.assert_array_almost_equal(
        enso_dataset["a"].sel(source="target").values,
        target_coefficients["a"].cpu().numpy(),
    )
    np.testing.assert_array_almost_equal(
        enso_dataset["a"].sel(source="prediction").values,
        gen_coefficients["a"].cpu().numpy(),
    )


@pytest.mark.parametrize("shift", [1.5, 0.95, 0.05, 0.0])
def test_enso_index_inference_overlap(shift):
    """
    Test the EnsoCoefficientEvaluatorAggregator returns metrics only for appropriate
    overlap of the reference index and the inference period.
    """
    n_samples, n_times, n_lat, n_lon = 2, 28, 3, 3
    data_scale = 3
    area_weights = torch.ones([n_lat, n_lon])
    enso_index, sample_time, target_data, gen_data = _get_data(
        data_scale, n_samples, n_times, n_lat, n_lon
    )
    # shift the sample times so they only partially overlap the reference index
    index_duration = enso_index.time[-1].item() - enso_index.time[0].item()
    offset_seconds = shift * index_duration.total_seconds()
    sample_time += datetime.timedelta(seconds=offset_seconds)
    with change_aggregator_enso_index(EnsoCoefficientEvaluatorAggregator, enso_index):
        enso_agg = EnsoCoefficientEvaluatorAggregator(
            initial_time=sample_time.isel(time=0),
            n_forward_timesteps=(n_times - 1),
            timestep=datetime.timedelta(hours=6),
            gridded_operations=LatLonOperations(area_weights),
        )
    enso_agg.record_batch(time=sample_time, target_data=target_data, gen_data=gen_data)
    target_coefficients, gen_coefficients = enso_agg._get_coefficients()
    overlap = max(1.0 - shift, 0.0)
    if overlap < OVERLAP_THRESHOLD:  # should be empty dict
        assert not target_coefficients
        assert not gen_coefficients
    else:
        assert target_coefficients is not None
        assert gen_coefficients is not None
        assert isinstance(target_coefficients["a"], torch.Tensor)
        assert isinstance(gen_coefficients["a"], torch.Tensor)


@pytest.mark.parametrize(
    "calendar",
    [
        pytest.param("julian", id="both_julian"),
        pytest.param("proleptic_gregorian", id="data_proleptic_gregorian_index_julian"),
    ],
)
def test_enso_agg_calendar(calendar):
    """
    Test the EnsoCoefficientEvaluatorAggregator class with different data calendars.

    The reference ENSO index is always in Julian calendar, but the data may be
    proleptic gregorian. The aggregator should be able to handle this.
    """
    n_samples, n_times, n_lat, n_lon = 2, 28, 3, 3
    data_scale = 3
    area_weights = torch.ones([n_lat, n_lon])
    enso_index, sample_time, target_data, gen_data = _get_data(
        data_scale, n_samples, n_times, n_lat, n_lon, calendar=calendar
    )
    enso_index = enso_index.convert_calendar("julian", dim="time", use_cftime=True)
    with change_aggregator_enso_index(EnsoCoefficientEvaluatorAggregator, enso_index):
        enso_agg = EnsoCoefficientEvaluatorAggregator(
            initial_time=sample_time.isel(time=0),
            n_forward_timesteps=(n_times - 1),
            timestep=datetime.timedelta(hours=6),
            gridded_operations=LatLonOperations(area_weights),
        )
    enso_agg.record_batch(time=sample_time, target_data=target_data, gen_data=gen_data)
    target_coefficients, gen_coefficients = enso_agg._get_coefficients()
    assert (target_coefficients is not None) and (gen_coefficients is not None)
