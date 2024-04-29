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

from .data_writer import DataWriter, NullDataWriter
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
        writer: Union[DataWriter, NullDataWriter],
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
            start_sample=0,
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

    # for non-initial windows, we want to record only the new data
    # and discard the initial sample of the window
    if i_time > 0:
        batch_times = batch_times.isel(time=slice(1, None))
        i_time_aggregator = i_time + 1
    else:
        i_time_aggregator = i_time
    # record raw data for the batch, and store the final state
    # for the next segment
    stitcher.append(stepped.target_data, stepped.gen_data, batch_times)
    # record metrics
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
    n_forward_steps: int,
    forward_steps_in_memory: int,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
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
        for i, window_batch_data in enumerate(data.loader):
            timers["data_loading"] += time.time() - current_time
            current_time = time.time()
            i_time = i * forward_steps_in_memory
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
            stepped = compute_stepped_derived_quantities(
                stepped, data.sigma_coordinates
            )

            timers["run_on_batch"] += time.time() - current_time
            current_time = time.time()
            _inference_internal_loop(
                stepped,
                i_time,
                aggregator,
                stitcher,
                window_batch_data.times,
            )
            timers["writer_and_aggregator"] += time.time() - current_time
            current_time = time.time()

        for name, duration in timers.items():
            logging.info(f"{name} duration: {duration:.2f}s")
    return timers


def run_dataset_inference(
    aggregator: InferenceAggregator,
    normalizer: StandardNormalizer,
    prediction_data: GriddedData,
    target_data: GriddedData,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
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
    for i, (pred, target) in enumerate(zip(prediction_data.loader, target_data.loader)):
        timers["data_loading"] += time.time() - current_time
        current_time = time.time()
        i_time = i * forward_steps_in_memory
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
            stepped, target_data.sigma_coordinates
        )

        # Windows here all include an initial condition at start.
        # Remove IC for windows >0 to be consistent with run_on_batch
        # outputs before passing to the shared _inference_internal_loop.
        if i > 0:
            stepped = stepped.remove_initial_condition()

        timers["run_on_batch"] += time.time() - current_time
        current_time = time.time()
        _inference_internal_loop(
            stepped,
            i_time,
            aggregator,
            stitcher,
            target.times,
        )
        timers["writer_and_aggregator"] += time.time() - current_time
        current_time = time.time()
    for name, duration in timers.items():
        logging.info(f"{name} duration: {duration:.2f}s")
    return timers
