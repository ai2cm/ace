import logging
import time
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Union

import torch
import xarray as xr

from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.data_loading.data_typing import GriddedData
from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
from fme.core.optimization import NullOptimization
from fme.core.stepper import SteppedData

from .data_writer import NullDataWriter, PairedDataWriter
from .derived_variables import compute_stepped_derived_quantities


class WindowStitcher:
    """
    Handles stitching together the windows of data from the inference loop.

    For example, handles passing in windows to data writers which combine
    them together into a continuous series, and handles storing prognostic
    variables from the end of a window to use as the initial condition for
    the next window.
    """

    def __init__(
        self,
        n_forward_steps: int,
        writer: Union[PairedDataWriter, NullDataWriter],
    ):
        self.i_time = 0
        self.n_forward_steps = n_forward_steps
        self.writer = writer
        # tensors have shape [n_sample, n_lat, n_lon] with no time axis
        self._initial_condition: Optional[Mapping[str, torch.Tensor]] = None

    def append(
        self,
        data: Dict[str, torch.tensor],
        gen_data: Dict[str, torch.tensor],
        batch_times: xr.DataArray,
    ) -> None:
        """
        Appends a time segment of data to the ensemble batch.

        Args:
            data: The reference data for the current time segment, tensors
                should have shape [n_sample, n_time, n_lat, n_lon]
            gen_data: The generated data for the current time segment, tensors
                should have shape [n_sample, n_time, n_lat, n_lon]
            batch_times: Time coordinates for each sample in the batch.
        """
        tensor_shape = next(data.values().__iter__()).shape
        self.writer.append_batch(
            target=data,
            prediction=gen_data,
            start_timestep=self.i_time,
            batch_times=batch_times,
        )
        self.i_time += tensor_shape[1]
        self._initial_condition = {key: value[:, -1] for key, value in data.items()}
        for key, value in gen_data.items():
            self._initial_condition[key] = value[:, -1]
        for key, value in self._initial_condition.items():
            self._initial_condition[key] = value.detach().cpu()

    def apply_initial_condition(self, data: Mapping[str, torch.Tensor]):
        """
        Applies the last recorded state of the batch as the initial condition for
        the next segment of the timeseries.

        Args:
            data: The data to apply the initial condition to, tensors should have
                shape [n_sample, n_time, n_lat, n_lon] and the first value along
                the time axis will be replaced with the last value from the
                previous segment.
        """
        if self.i_time > self.n_forward_steps:
            raise ValueError(
                "Cannot apply initial condition after "
                "the last segment has been appended, currently at "
                f"time index {self.i_time} "
                f"with {self.n_forward_steps} max forward steps."
            )
        if self._initial_condition is not None:
            for key, value in data.items():
                value[:, 0] = self._initial_condition[key].to(value.device)

    def save_initial_condition(
        self,
        ic_data: Dict[str, torch.Tensor],
        ic_time: xr.DataArray,
    ):
        self.writer.save_initial_condition(ic_data, ic_time)


def _inference_internal_loop(
    stepped: SteppedData,
    i_time: int,
    aggregator: InferenceAggregator,
    stitcher: WindowStitcher,
    batch_times: xr.DataArray,
):
    """Do operations that need to be done on each time step of the inference loop.

    This function exists to de-duplicate code between run_inference and
    run_data_inference."""

    # The first data window includes the IC, while subsequent windows don't.
    # The aggregators use the full first window including IC.
    # The data writers exclude the IC from the first window.
    if i_time == 0:
        i_time_aggregator = i_time
        stepped_no_ic = stepped.remove_initial_condition()
        stitcher.save_initial_condition(
            ic_data={k: v[:, 0] for k, v in stepped.target_data.items()},
            ic_time=batch_times.isel(time=0),
        )
        batch_times_no_ic = batch_times.isel(time=slice(1, None))
    else:
        i_time_aggregator = i_time + 1
        stepped_no_ic = stepped
        batch_times_no_ic = batch_times

    # record raw data for the batch, and store the final state
    # for the next segment
    # Do not include the initial condition in the data writers
    stitcher.append(
        stepped_no_ic.target_data, stepped_no_ic.gen_data, batch_times_no_ic
    )

    # record metrics, includes the initial condition
    aggregator.record_batch(
        loss=float(stepped.metrics["loss"]),
        time=batch_times,
        target_data=stepped.target_data,
        gen_data=stepped.gen_data,
        target_data_norm=stepped.target_data_norm,
        gen_data_norm=stepped.gen_data_norm,
        i_time_start=i_time_aggregator,
    )


def _to_device(
    data: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in data.items()}


def run_inference(
    aggregator: InferenceAggregator,
    stepper: SingleModuleStepper,
    data: GriddedData,
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
    n_forward_steps = data.loader.dataset.n_forward_steps
    stitcher = WindowStitcher(n_forward_steps, writer)

    with torch.no_grad():
        # We have data batches with long windows, where all data for a
        # given batch does not fit into memory at once, so we window it in time
        # and run the model on each window in turn.
        #
        # We process each time window and keep track of the
        # final state. We then use this as the initial condition
        # for the next time window.

        timers: Dict[str, float] = defaultdict(float)
        current_time = time.time()
        i_time = 0
        for i, window_batch_data in enumerate(data.loader):
            timers["data_loading"] += time.time() - current_time
            current_time = time.time()
            forward_steps_in_memory = (
                list(window_batch_data.data.values())[0].size(1) - 1
            )
            logging.info(
                f"Inference: starting window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps, "
                f"out of total {n_forward_steps}."
            )
            device = get_device()
            window_data = _to_device(window_batch_data.data, device)

            stitcher.apply_initial_condition(window_data)

            stepped = stepper.run_on_batch(
                window_data,
                NullOptimization(),
                n_forward_steps=forward_steps_in_memory,
            )

            # Prepend initial (pre-first-timestep) output for the first window
            if i == 0:
                (
                    initial_condition,
                    normed_initial_condition,
                ) = stepper.get_initial_condition(window_data)
                stepped = stepped.prepend_initial_condition(
                    initial_condition, normed_initial_condition
                )
                batch_times = window_batch_data.times
            else:
                batch_times = window_batch_data.times.isel(time=slice(1, None))
            stepped = compute_stepped_derived_quantities(
                stepped, data.sigma_coordinates, data.timestep
            )
            timers["run_on_batch"] += time.time() - current_time
            current_time = time.time()
            _inference_internal_loop(stepped, i_time, aggregator, stitcher, batch_times)
            timers["writer_and_aggregator"] += time.time() - current_time
            current_time = time.time()
            i_time += forward_steps_in_memory

        for name, duration in timers.items():
            logging.info(f"{name} duration: {duration:.2f}s")
    return timers


def run_dataset_inference(
    aggregator: InferenceAggregator,
    normalizer: StandardNormalizer,
    prediction_data: GriddedData,
    target_data: GriddedData,
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
    n_forward_steps = target_data.loader.dataset.n_forward_steps
    stitcher = WindowStitcher(n_forward_steps, writer)

    device = get_device()
    # We have data batches with long windows, where all data for a
    # given batch does not fit into memory at once, so we window it in time
    # and run the model on each window in turn.
    #
    # We process each time window and keep track of the
    # final state. We then use this as the initial condition
    # for the next time window.
    timers: Dict[str, float] = defaultdict(float)
    current_time = time.time()
    i_time = 0
    for i, (pred, target) in enumerate(zip(prediction_data.loader, target_data.loader)):
        timers["data_loading"] += time.time() - current_time
        current_time = time.time()
        forward_steps_in_memory = list(pred.data.values())[0].size(1) - 1
        logging.info(
            f"Inference: starting window spanning {i_time}"
            f" to {i_time + forward_steps_in_memory} steps,"
            f" out of total {n_forward_steps}."
        )
        pred_window_data = _to_device(pred.data, device)
        target_window_data = _to_device(target.data, device)
        stepped = SteppedData(
            {"loss": torch.tensor(float("nan"))},
            pred_window_data,
            target_window_data,
            normalizer.normalize(pred_window_data),
            normalizer.normalize(target_window_data),
        )
        stepped = compute_stepped_derived_quantities(
            stepped, target_data.sigma_coordinates, target_data.timestep
        )

        # Windows here all include an initial condition at start.
        # Remove IC and time coord for windows >0 to be consistent with
        # run_on_batch outputs before passing to the shared _inference_internal_loop.
        if i > 0:
            stepped = stepped.remove_initial_condition()
            target_times = target.times.isel(time=slice(1, None))
        else:
            target_times = target.times

        timers["run_on_batch"] += time.time() - current_time
        current_time = time.time()
        _inference_internal_loop(
            stepped,
            i_time,
            aggregator,
            stitcher,
            target_times,
        )
        timers["writer_and_aggregator"] += time.time() - current_time
        current_time = time.time()
        i_time += forward_steps_in_memory
    for name, duration in timers.items():
        logging.info(f"{name} duration: {duration:.2f}s")
    return timers
