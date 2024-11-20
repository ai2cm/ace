import datetime
import unittest.mock
from collections import namedtuple
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.inference.loop import Looper, get_record_to_wandb, run_inference
from fme.ace.inference.timing import GlobalTimer
from fme.core.data_loading.batch_data import (
    BatchData,
    CurrentDevice,
    PairedData,
    PrognosticState,
)
from fme.core.data_loading.data_typing import LatLonOperations, SigmaCoordinates
from fme.core.generics.inference import InferenceStepperABC
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import TensorDict, TensorMapping

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
    def __init__(
        self,
        shape: tuple,
        names: Iterable[str],
        n_windows: int,
        time: Optional[xr.DataArray] = None,
    ):
        device = fme.get_device()
        self._data = {n: torch.rand(*shape, device=device) for n in names}
        if time is None:
            self._time = xr.DataArray(np.zeros(shape[:2]), dims=["sample", "time"])
        elif time.shape != shape[:2]:
            raise ValueError(
                "Time shape must match the first two dimensions of the data."
            )
        else:
            self._time = time
        self._n_windows = n_windows
        self._current_window = 0

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self._n_windows

    def __next__(self) -> BatchData[CurrentDevice]:
        if self._current_window < self._n_windows:
            self._current_window += 1
            return BatchData.new_on_device(
                data=self._data,
                times=self._time
                + (self._current_window - 1) * (self._time.shape[1] - 1),
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
    time = xr.DataArray(
        np.repeat(np.expand_dims(np.arange(n_time), axis=0), n_samples, axis=0),
        dims=["sample", "time"],
    )

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
    initial_condition = BatchData.new_on_device(
        data={n: spherical_data.data[n][:, :1] for n in spherical_data.data},
        times=time[:, 0:1],
    ).get_start(
        prognostic_names=stepper.prognostic_names,
        n_ic_timesteps=1,
    )
    loader = MockLoader(shape, forcing_names, 3, time=time)
    looper = Looper(
        stepper=stepper,
        initial_condition=initial_condition,
        loader=loader,
    )

    expected_output_shape = (shape[0], shape[1] - 1, shape[2], shape[3])
    for prediction_batch, forcing_batch in looper:
        xr.testing.assert_identical(prediction_batch.times, forcing_batch.times)
        assert set(out_names) == set(prediction_batch.data)
        assert set(forcing_names) == set(forcing_batch.data)
        for name in out_names:
            assert prediction_batch.data[name].shape == expected_output_shape
        for name in forcing_names:
            assert forcing_batch.data[name].shape == expected_output_shape


def _mock_compute_derived_quantities(data, forcing_data):
    data_name = list(data)[0]
    forcing_name = list(forcing_data)[0]
    derived = {"derived": data[data_name] + forcing_data[forcing_name]}
    return {**data, **derived}


def test_looper_with_derived_variables():
    mock_derive_func = unittest.mock.MagicMock(
        side_effect=_mock_compute_derived_quantities
    )
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    stepper.derive_func = mock_derive_func
    forcing_names = set(in_names) - set(out_names)
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = BatchData.new_on_device(
        {n: spherical_data.data[n][:, :1] for n in spherical_data.data},
        times=time[:, 0:1],
    ).get_start(
        prognostic_names=stepper.prognostic_names,
        n_ic_timesteps=1,
    )
    loader = MockLoader(shape, forcing_names, 2, time=time)
    looper = Looper(
        stepper=stepper,
        initial_condition=initial_condition,
        loader=loader,
    )

    for prediction_batch, forcing_batch in looper:
        assert "derived" in prediction_batch.data
        assert "derived" not in forcing_batch.data
    mock_derive_func.assert_called()


def test_looper_with_target_data():
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    all_names = list(set(in_names + out_names))
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = BatchData.new_on_device(
        data={n: spherical_data.data[n][:, :1] for n in spherical_data.data},
        times=time[:, 0:1],
    ).get_start(
        prognostic_names=stepper.prognostic_names,
        n_ic_timesteps=1,
    )
    loader = MockLoader(shape, all_names, 2, time=time)
    looper = Looper(
        stepper=stepper,
        initial_condition=initial_condition,
        loader=loader,
    )

    for prediction_batch, forcing_batch in looper:
        assert set(out_names) == set(prediction_batch.data)
        assert set(all_names) == set(forcing_batch.data)


def test_looper_with_target_data_and_derived_variables():
    mock_derive_func = unittest.mock.MagicMock(
        side_effect=_mock_compute_derived_quantities
    )
    stepper, spherical_data, time, in_names, out_names = _get_stepper()
    stepper.derive_func = mock_derive_func
    all_names = list(set(in_names + out_names))
    shape = spherical_data.data[in_names[0]].shape
    initial_condition = BatchData.new_on_device(
        data={n: spherical_data.data[n][:, :1] for n in spherical_data.data},
        times=time[:, 0:1],
    ).get_start(
        prognostic_names=stepper.prognostic_names,
        n_ic_timesteps=1,
    )
    loader = MockLoader(shape, all_names, 2, time=time)
    looper = Looper(
        stepper,
        initial_condition=initial_condition,
        loader=loader,
        compute_derived_for_loaded_data=True,
    )

    for prediction_batch, forcing_batch in looper:
        assert set(out_names + ["derived"]) == set(prediction_batch.data)
        assert set(all_names + ["derived"]) == set(forcing_batch.data)
    mock_derive_func.assert_called()


def get_batch_data(
    start_time,
    n_timesteps,
):
    n_samples = 1
    n_lat = 3
    n_lon = 4
    time_values = torch.arange(
        start_time, start_time + n_timesteps, device=fme.get_device()
    )[None, :, None, None]
    time_axis = torch.broadcast_to(
        start_time + torch.arange(n_timesteps)[None, :], (n_samples, n_timesteps)
    )
    times = xr.DataArray(time_axis, dims=["sample", "time"])
    return BatchData.new_on_device(
        data={
            "var": torch.broadcast_to(
                time_values, (n_samples, n_timesteps, n_lat, n_lon)
            )
        },
        times=times,
    )


class PlusOneStepper(
    InferenceStepperABC[PrognosticState[CurrentDevice], BatchData[CurrentDevice]]
):
    def __init__(
        self,
        n_ic_timesteps: int,
        derive_func: Optional[
            Callable[[TensorMapping, TensorMapping], TensorDict]
        ] = None,
    ):
        self.n_ic_timesteps = n_ic_timesteps
        if derive_func is None:
            self.derive_func: Callable[
                [TensorMapping, TensorMapping], TensorDict
            ] = unittest.mock.MagicMock(side_effect=lambda x, y=None: dict(x))
        else:
            self.derive_func = derive_func

    def predict(
        self,
        initial_condition: PrognosticState[CurrentDevice],
        forcing: BatchData[CurrentDevice],
        compute_derived_variables: bool = False,
    ) -> Tuple[BatchData[CurrentDevice], PrognosticState[CurrentDevice]]:
        ic_state = initial_condition.as_batch_data()
        n_forward_steps = forcing.times.shape[1] - self.n_ic_timesteps
        out_tensor = torch.zeros(
            ic_state.data["var"].shape[0],
            n_forward_steps,
            *ic_state.data["var"].shape[2:],
            device=ic_state.data["var"].device,
            dtype=ic_state.data["var"].dtype,
        )
        out_tensor[:, 0, ...] = ic_state.data["var"][:, -1, ...] + 1
        for i in range(1, n_forward_steps):
            out_tensor[:, i, ...] = out_tensor[:, i - 1, ...] + 1
        data = BatchData.new_on_device(
            data={"var": out_tensor},
            times=forcing.times[:, self.n_ic_timesteps :],
        )
        if compute_derived_variables:
            data = data.compute_derived_variables(
                derive_func=self.derive_func, forcing_data=data
            )
        return data, data.get_end(["var"], self.n_ic_timesteps)

    def get_forward_data(
        self,
        forcing: BatchData[CurrentDevice],
        compute_derived_variables: bool = False,
    ) -> BatchData[CurrentDevice]:
        if compute_derived_variables:
            forcing = forcing.compute_derived_variables(
                derive_func=self.derive_func, forcing_data=forcing
            )
        return forcing.remove_initial_condition(self.n_ic_timesteps)


@pytest.mark.parametrize(
    "compute_derived_for_loaded",
    [True, False],
)
def test_simple_batch_data_looper(compute_derived_for_loaded: bool):
    n_ic_timesteps = 1
    n_forward_steps = 2
    n_iterations = 10
    mock_derive_func = unittest.mock.MagicMock(
        side_effect=lambda batch_data, forcing_data: batch_data
    )
    stepper = PlusOneStepper(
        n_ic_timesteps=n_ic_timesteps, derive_func=mock_derive_func
    )
    initial_condition = get_batch_data(
        0,
        n_timesteps=n_ic_timesteps,
    ).get_start(prognostic_names=["var"], n_ic_timesteps=n_ic_timesteps)
    loader = [
        get_batch_data(
            i,
            n_ic_timesteps + n_forward_steps,
        )
        for i in range(0, n_iterations * n_forward_steps, n_forward_steps)
    ]
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        looper = Looper(
            stepper,
            initial_condition=initial_condition,
            loader=loader,
            compute_derived_for_loaded_data=compute_derived_for_loaded,
        )
        for i, (prediction, forcing) in enumerate(looper):
            for j in range(prediction.times.shape[1]):
                assert torch.allclose(
                    prediction.data["var"][:, j, ...],
                    torch.as_tensor(n_ic_timesteps + i * n_forward_steps + j),
                )
                assert torch.allclose(
                    forcing.data["var"][:, j, ...],
                    torch.as_tensor(n_ic_timesteps + i * n_forward_steps + j),
                )
        times = timer.get_durations()
        assert times["data_loading"] > 0
        assert times["forward_prediction"] > 0
        if compute_derived_for_loaded:
            assert mock_derive_func.call_count == 2 * n_iterations
        else:
            assert mock_derive_func.call_count == n_iterations
        # we mocked out the implicit call that happens in .predict and .get_forward_data
        # if this changed, update the test
        assert "compute_derived_variables" not in times


@pytest.mark.parametrize(
    "n_ic_timesteps, n_forward_steps, n_iterations",
    [
        pytest.param(1, 2, 5, id="n_ic_timesteps=1"),
        pytest.param(2, 2, 5, id="n_ic_timesteps=2"),
    ],
)
def test_simple_run_inference(
    n_ic_timesteps: int, n_forward_steps: int, n_iterations: int
):
    mock_derive_func = unittest.mock.MagicMock(
        side_effect=lambda batch_data, forcing_data: batch_data
    )
    stepper = PlusOneStepper(
        n_ic_timesteps=n_ic_timesteps, derive_func=mock_derive_func
    )
    initial_condition = get_batch_data(
        0,
        n_timesteps=n_ic_timesteps,
    ).get_start(prognostic_names=["var"], n_ic_timesteps=n_ic_timesteps)
    loader = [
        get_batch_data(
            i,
            n_ic_timesteps + n_forward_steps,
        )
        for i in range(0, n_iterations * n_forward_steps, n_forward_steps)
    ]
    mock_writer = unittest.mock.MagicMock()

    # record_batch will start at step n_ic_timesteps
    i = n_ic_timesteps

    def record_batch_side_effect(
        data: PairedData[CurrentDevice],
    ):
        nonlocal i
        ret = [{"step": j} for j in range(i, i + data.times.shape[1])]
        i += data.times.shape[1]
        return ret

    mock_aggregator = unittest.mock.MagicMock()
    mock_aggregator.record_initial_condition = unittest.mock.MagicMock(
        return_value=[{"step": j} for j in range(n_ic_timesteps)]
    )

    def get_summary_logs_side_effect():
        # we expect this gets called _outside_ of run_inference
        raise ValueError("should not be called")

    mock_aggregator.get_summary_logs = unittest.mock.MagicMock(
        side_effect=get_summary_logs_side_effect
    )
    mock_aggregator.record_batch = unittest.mock.MagicMock(
        side_effect=record_batch_side_effect
    )

    with GlobalTimer():
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True)
            record_logs = unittest.mock.MagicMock(
                side_effect=get_record_to_wandb("inference")
            )  # this init must be within mock_wandb context
            run_inference(
                stepper,
                initial_condition,
                loader,
                writer=mock_writer,
                aggregator=mock_aggregator,
                record_logs=record_logs,
            )
            wandb_logs = wandb.get_logs()
        timer = GlobalTimer.get_instance()
        times = timer.get_durations()
        assert times["wandb_logging"] > 0
        assert times["data_writer"] > 0
        assert times["aggregator"] > 0
        assert mock_writer.save_initial_condition.call_count == 1
        assert mock_aggregator.record_initial_condition.call_count == 1
        assert mock_writer.append_batch.call_count == n_iterations
        assert mock_aggregator.record_batch.call_count == n_iterations
        assert len(wandb_logs) == n_ic_timesteps + n_iterations * n_forward_steps
        assert wandb_logs == [
            {"inference/step": i}
            for i in range(n_ic_timesteps + n_iterations * n_forward_steps)
        ]
        assert (
            record_logs.call_count == n_iterations + 1
        )  # +1 for the initial condition
