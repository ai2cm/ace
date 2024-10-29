import datetime
import unittest.mock
from collections import namedtuple
from typing import Iterable, Tuple

import numpy as np
import torch
import xarray as xr

import fme
from fme.ace.inference.loop import Looper
from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.data_typing import LatLonOperations, SigmaCoordinates
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.stepper import SingleModuleStepperConfig

SphericalData = namedtuple("SphericalData", ["data", "area_weights", "sigma_coords"])


def get_data(
    names: Iterable[str], shape: Tuple[int, int, int, int, int]
) -> SphericalData:
    data = {}
    n_lat = shape[2]
    n_lon = shape[3]
    shape_without_z = shape[:-1]
    nz = shape[-1]
    lats = torch.linspace(-89.5, 89.5, n_lat)
    for name in names:
        data[name] = torch.rand(*shape_without_z, device=fme.get_device())
    area_weights = fme.spherical_area_weights(lats, n_lon).to(fme.get_device())
    ak, bk = torch.arange(nz), torch.arange(nz)
    sigma_coords = SigmaCoordinates(ak, bk)
    return SphericalData(data, area_weights, sigma_coords)


def get_scalar_data(names, value):
    return {n: np.array([value], dtype=np.float32) for n in names}


class MockLoader(torch.utils.data.DataLoader):
    def __init__(self, shape: tuple, names: Iterable[str], n_windows: int):
        self._data = {n: torch.rand(*shape) for n in names}
        self._time = xr.DataArray(np.zeros(shape[:2]), dims=["sample", "time"])
        self._n_windows = n_windows
        self._current_window = 0

    def __iter__(self):
        return self

    def __next__(self) -> BatchData:
        if self._current_window < self._n_windows:
            self._current_window += 1
            return BatchData(
                self._data, self._time + self._current_window * len(self._time)
            )
        else:
            raise StopIteration


def _get_stepper():
    class ChannelSum(torch.nn.Module):
        def forward(self, x):
            summed = torch.sum(x, dim=-3, keepdim=True)
            diagnostic = torch.rand_like(summed)
            return torch.concat([x, diagnostic], dim=-3)

    n_samples = 2
    n_time = 5
    n_lat = 2
    n_lon = 4
    nz = 3
    shape = (n_samples, n_time, n_lat, n_lon, nz)
    in_names = ["forcing", "prognostic"]
    out_names = ["prognostic", "diagnostic"]
    all_names = list(set(in_names + out_names))

    spherical_data = get_data(all_names, shape)
    time = xr.DataArray(np.arange(n_time), dims=["time"])

    img_shape = spherical_data.data[in_names[0]].shape[2:]
    gridded_operations = LatLonOperations(spherical_data.area_weights)
    sigma_coordinates = spherical_data.sigma_coords
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": ChannelSum()}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means=get_scalar_data(all_names, 0.0),
            stds=get_scalar_data(all_names, 1.0),
        ),
        loss=WeightedMappingLossConfig(),
    )
    stepper = config.get_stepper(
        img_shape, gridded_operations, sigma_coordinates, datetime.timedelta(seconds=1)
    )
    return stepper, spherical_data, time, in_names, out_names


def test_looper():
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    forcing_names = set(in_names) - set(out_names)
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = {n: spherical_data.data[n][:, :1] for n in spherical_data.data}
    initial_time = time[0:]
    loader = MockLoader(shape, forcing_names, 3)
    looper = Looper(stepper, initial_condition, initial_time, loader)

    expected_output_shape = (shape[0], shape[1] - 1, shape[2], shape[3])
    for prediction_batch, forcing_batch in looper:
        xr.testing.assert_identical(prediction_batch.times, forcing_batch.times)
        assert set(out_names) == set(prediction_batch.data)
        assert set(forcing_names) == set(forcing_batch.data)
        for name in out_names:
            assert prediction_batch.data[name].shape == expected_output_shape
        for name in forcing_names:
            assert forcing_batch.data[name].shape == expected_output_shape


def _mock_compute_derived_quantities(data, sigma_coords, timestep, forcing_data):
    data_name = list(data)[0]
    forcing_name = list(forcing_data)[0]
    derived = {"derived": data[data_name] + forcing_data[forcing_name]}
    return {**data, **derived}


def test_looper_with_derived_variables():
    mock = unittest.mock.MagicMock(side_effect=_mock_compute_derived_quantities)
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    forcing_names = set(in_names) - set(out_names)
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = {n: spherical_data.data[n][:, :1] for n in spherical_data.data}
    initial_time = time[0:]
    loader = MockLoader(shape, forcing_names, 2)
    with unittest.mock.patch("fme.ace.inference.loop.compute_derived_quantities", mock):
        looper = Looper(stepper, initial_condition, initial_time, loader)

        for prediction_batch, forcing_batch in looper:
            assert "derived" in prediction_batch.data
            assert "derived" not in forcing_batch.data


def test_looper_with_target_data():
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    all_names = list(set(in_names + out_names))
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = {n: spherical_data.data[n][:, :1] for n in spherical_data.data}
    initial_time = time[0:]
    loader = MockLoader(shape, all_names, 2)
    looper = Looper(stepper, initial_condition, initial_time, loader)

    for prediction_batch, forcing_batch in looper:
        assert set(out_names) == set(prediction_batch.data)
        assert set(all_names) == set(forcing_batch.data)


def test_looper_with_target_data_and_derived_variables():
    mock = unittest.mock.MagicMock(side_effect=_mock_compute_derived_quantities)
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    all_names = list(set(in_names + out_names))
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = {n: spherical_data.data[n][:, :1] for n in spherical_data.data}
    initial_time = time[0:]
    loader = MockLoader(shape, all_names, 2)
    with unittest.mock.patch("fme.ace.inference.loop.compute_derived_quantities", mock):
        looper = Looper(
            stepper,
            initial_condition,
            initial_time,
            loader,
            compute_derived_for_loaded_data=True,
        )

        for prediction_batch, forcing_batch in looper:
            assert set(out_names + ["derived"]) == set(prediction_batch.data)
            assert set(all_names + ["derived"]) == set(forcing_batch.data)
