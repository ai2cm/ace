import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import torch
import xarray as xr

from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import (
    InferenceAggregator,
    InferenceEvaluatorAggregator,
)
from fme.core.data_loading.data_typing import GriddedData
from fme.core.data_loading.utils import BatchData
from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
from fme.core.optimization import NullOptimization
from fme.core.stepper import SteppedData
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB

from .data_writer import DataWriter, NullDataWriter, PairedDataWriter
from .derived_variables import (
    compute_derived_quantities,
    compute_stepped_derived_quantities,
)


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
        snapshot_dims: List[str],
    ):
        self.writer.save_initial_condition(ic_data, ic_time, snapshot_dims)


def _inference_internal_loop(
    stepped: SteppedData,
    i_time: int,
    aggregator: InferenceEvaluatorAggregator,
    stitcher: WindowStitcher,
    batch_times: xr.DataArray,
    snapshot_dims: List[str],
):
    """Do operations that need to be done on each time step of the inference loop.

    This function exists to de-duplicate code between run_inference_evaluator and
    run_dataset_comparison.

    Args:
        stepped: windowed data, which has dim (batch, time, <snapshot_dims>)
        aggregator: the aggregator object
        stitcher: stitches together windows of data from the inference loop.
        batch_times: the times represented in this batch.
        snapshot_dims: represent the dimensions of the time snapshot.
            (B, <spatial dimensions>)

    Note:
        The first data window includes the IC, while subsequent windows don't.
        The aggregators use the full first window including IC.
        The data writers exclude the IC from the first window.
    """
    timers = {}
    current_time = time.time()
    if i_time == 0:
        i_time_aggregator = i_time
        stepped_no_ic = stepped.remove_initial_condition()
        stitcher.save_initial_condition(
            ic_data={k: v[:, 0] for k, v in stepped.target_data.items()},
            ic_time=batch_times.isel(time=0),
            snapshot_dims=snapshot_dims,
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
    timers["writer_and_aggregator"] = time.time() - current_time
    current_time = time.time()

    _log_window_to_wandb(
        aggregator,
        window_slice=slice(
            i_time_aggregator, i_time_aggregator + len(batch_times["time"])
        ),
        label="inference",
    )
    timers["wandb_logging"] = time.time() - current_time

    return timers


def _log_window_to_wandb(
    aggregator: Union[InferenceAggregator, InferenceEvaluatorAggregator],
    window_slice: slice,
    label: str,
):
    if not aggregator.log_time_series:
        return
    wandb = WandB.get_instance()
    if wandb.enabled:
        logging.info(f"Logging inference window to wandb")
        current_time = time.time()
        step_logs = aggregator.get_inference_logs_slice(
            label=label,
            step_slice=window_slice,
        )
        aggregator_duration = time.time() - current_time
        current_time = time.time()
        for j, log in enumerate(step_logs):
            wandb.log(log, step=window_slice.start + j)
        wandb.log(
            {
                "aggregator_get_inference_logs_steps_per_second": len(step_logs)
                / aggregator_duration,
                "wandb_log_steps_per_second": len(step_logs)
                / (time.time() - current_time),
            },
            step=window_slice.start + len(step_logs) - 1,
        )


def _to_device(
    data: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in data.items()}


def run_inference(
    stepper: SingleModuleStepper,
    initial_condition: TensorMapping,
    forcing_data: GriddedData,
    writer: DataWriter,
    aggregator: InferenceAggregator,
) -> Dict[str, float]:
    """Run extended inference loop given initial condition and forcing data.

    Args:
        stepper: The model to run inference with.
        initial_condition: Mapping of prognostic names to initial condition tensors of
            shape (n_sample, n_lat, n_lon).
        forcing_data: GriddedData object which includes a DataLoader which will provide
            windows of forcing data appropriately aligned with the initial condition.
        writer: Data writer for saving the inference results to disk.

    Returns:
        Execution time in seconds for each step of the inference loop.
    """
    with torch.no_grad():
        timers: Dict[str, float] = defaultdict(float)
        current_time = time.time()
        i_time = 0
        window_forcing: BatchData
        diagnostic_ic: Dict[str, torch.Tensor] = {}
        for window_forcing in forcing_data.loader:
            timers["data_loading"] += time.time() - current_time
            current_time = time.time()
            forward_steps_in_memory = list(window_forcing.data.values())[0].size(1) - 1
            logging.info(
                f"Inference: starting window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps."
            )
            window_forcing_data = _to_device(window_forcing.data, get_device())
            prediction = stepper.predict(
                initial_condition, window_forcing_data, forward_steps_in_memory
            )
            timers["run_on_batch"] += time.time() - current_time

            time_dim = 1
            if len(diagnostic_ic) == 0:
                diagnostic_ic = {
                    k: torch.zeros_like(prediction[k].select(time_dim, 0))
                    for k in stepper.diagnostic_names
                }
            prediction_ic = {**initial_condition, **diagnostic_ic}
            prediction_with_ic = {
                k: torch.cat(
                    [prediction_ic[k].unsqueeze(time_dim), prediction[k]],
                    dim=time_dim,
                )
                for k in prediction
            }
            prediction = {
                k: v[:, 1:]
                for k, v in compute_derived_quantities(
                    prediction_with_ic,
                    forcing_data.sigma_coordinates,
                    forcing_data.timestep,
                    forcing_data=window_forcing_data,
                ).items()
            }

            forward_times = window_forcing.times.isel(time=slice(1, None))
            writer.append_batch(prediction, i_time, forward_times)
            aggregator.record_batch(
                time=forward_times, data=prediction, i_time_start=i_time + 1
            )
            timers["writer_and_aggregator"] += time.time() - current_time
            current_time = time.time()
            if i_time == 0:
                _log_window_to_wandb(
                    aggregator,
                    window_slice=slice(0, 1),
                    label="inference",
                )
            _log_window_to_wandb(
                aggregator,
                window_slice=slice(i_time + 1, i_time + len(forward_times["time"]) + 1),
                label="inference",
            )
            timers["wandb_logging"] += time.time() - current_time
            current_time = time.time()

            initial_condition = {
                k: prediction[k][:, -1] for k in stepper.prognostic_names
            }
            diagnostic_ic = {k: prediction[k][:, -1] for k in stepper.diagnostic_names}
            i_time += forward_steps_in_memory

        for name, duration in timers.items():
            logging.info(f"{name} duration: {duration:.2f}s")
    return timers


def run_inference_evaluator(
    aggregator: InferenceEvaluatorAggregator,
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
        target_initial_condition, normed_target_initial_condition = None, None
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

            # Prepend initial generated (pre-first-timestep) output for all windows
            # to calculate derived quantities
            (
                initial_condition,
                normed_initial_condition,
            ) = stepper.get_initial_condition(window_data)
            stepped = stepped.prepend_initial_condition(
                initial_condition,
                normed_initial_condition,
                target_initial_condition,
                normed_target_initial_condition,
            )

            stepped = compute_stepped_derived_quantities(
                stepped,
                data.sigma_coordinates,
                data.timestep,
                # forcing inputs are in target data but not gen_data
                forcing_data=stepped.target_data,
            )
            timers["run_on_batch"] += time.time() - current_time
            current_time = time.time()

            # Remove initial condition for windows >0
            if i > 0:
                stepped = stepped.remove_initial_condition()
                batch_times = window_batch_data.times.isel(time=slice(1, None))
            else:
                batch_times = window_batch_data.times

            timers["writer_and_aggregator"] += time.time() - current_time

            snapshot_dims = ["sample"] + data.horizontal_coordinates.dims
            internal_timers = _inference_internal_loop(
                stepped, i_time, aggregator, stitcher, batch_times, snapshot_dims
            )
            timers["writer_and_aggregator"] += internal_timers["writer_and_aggregator"]
            timers["wandb_logging"] += internal_timers["wandb_logging"]

            current_time = time.time()
            i_time += forward_steps_in_memory

            # Save last target data timestep to use as IC for calculating
            # tendencies in next window
            target_initial_condition, normed_target_initial_condition = (
                {k: v[:, -1] for k, v in stepped.target_data.items()},
                {k: v[:, -1] for k, v in stepped.target_data_norm.items()},
            )

        for name, duration in timers.items():
            logging.info(f"{name} duration: {duration:.2f}s")
    return timers


def run_dataset_comparison(
    aggregator: InferenceEvaluatorAggregator,
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

        timers["run_on_batch"] += time.time() - current_time
        current_time = time.time()

        # Windows here all include an initial condition at start.
        # Remove IC and time coord for windows >0 to be consistent with
        # run_on_batch outputs before passing to the shared _inference_internal_loop.
        if i > 0:
            stepped = stepped.remove_initial_condition()
            target_times = target.times.isel(time=slice(1, None))
        else:
            target_times = target.times

        timers["writer_and_aggregator"] += time.time() - current_time

        snapshot_dims = ["sample"] + target_data.horizontal_coordinates.dims
        internal_timers = _inference_internal_loop(
            stepped,
            i_time,
            aggregator,
            stitcher,
            target_times,
            snapshot_dims,
        )
        timers["writer_and_aggregator"] += internal_timers["writer_and_aggregator"]
        timers["wandb_logging"] += internal_timers["wandb_logging"]

        current_time = time.time()
        i_time += forward_steps_in_memory

    for name, duration in timers.items():
        logging.info(f"{name} duration: {duration:.2f}s")
    return timers


def write_reduced_metrics(
    aggregator: Union[InferenceEvaluatorAggregator, InferenceAggregator],
    data_coords: Mapping[str, np.ndarray],
    path: str,
    excluded: Optional[Iterable[str]] = None,
):
    """
    Write the reduced metrics to disk. Each sub-aggregator will write a netCDF file
    if its `get_dataset` method returns a non-empty dataset.

    Args:
        aggregator: The aggregator to write metrics from.
        data_coords: Coordinates to assign to the datasets.
        path: Path to write the metrics to.
        excluded: Names of metrics to exclude from writing.
    """
    for name, ds in aggregator.get_datasets(excluded_aggregators=excluded).items():
        if len(ds) > 0:
            coords = {k: v for k, v in data_coords.items() if k in ds.dims}
            ds = ds.assign_coords(coords)
            ds.to_netcdf(Path(path) / f"{name}_diagnostics.nc")
