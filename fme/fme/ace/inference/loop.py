import logging
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

from fme.ace.inference.timing import GlobalTimer
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import (
    InferenceAggregator,
    InferenceEvaluatorAggregator,
)
from fme.core.data_loading.data_typing import GriddedData, GriddedDataABC
from fme.core.data_loading.utils import BatchData
from fme.core.device import move_tensordict_to_device
from fme.core.normalizer import StandardNormalizer
from fme.core.stepper import SteppedData
from fme.core.typing_ import TensorDict
from fme.core.wandb import WandB

from .data_writer import DataWriter, NullDataWriter, PairedDataWriter
from .derived_variables import (
    compute_derived_quantities,
    compute_stepped_derived_quantities,
)


def _prepend_timesteps(
    data: TensorDict, timesteps: TensorDict, time_dim: int = 1
) -> TensorDict:
    return {k: torch.cat([timesteps[k], v], dim=time_dim) for k, v in data.items()}


class Looper:
    """
    Class for stepping a model forward arbitarily many times.
    """

    def __init__(
        self,
        stepper: SingleModuleStepper,
        initial_condition: TensorDict,
        initial_times: xr.DataArray,
        loader: Iterable[BatchData],
        compute_derived_for_loaded_data: bool = False,
    ):
        """
        Args:
            stepper: The stepper to use.
            initial_condition: The initial condition data.
            initial_times: The initial times.
            loader: The loader for the forcing, and possibly target, data.
            derived_variables_callable: A callable that computes and inserts derived
                variables into the prediction data.
            compute_derived_for_loaded_data: Whether to compute derived variables for
                the data returned by the loader.
        """
        self._stepper = stepper
        self._n_ic_timesteps = stepper.n_ic_timesteps
        assert set(stepper.prognostic_names).issubset(set(initial_condition))
        self._prognostic_state = move_tensordict_to_device(initial_condition)
        # Insert fill values for diagnostic variables. This is required only because
        # we will later prepend the initial condition to the prediction data to allow
        # for computing tendencies.
        for name in stepper.diagnostic_names:
            self._prognostic_state[name] = torch.full_like(
                self._prognostic_state[stepper.prognostic_names[0]],
                fill_value=torch.nan,
            )
        self._loader = iter(loader)
        self._current_time = initial_times
        self._compute_derived = lambda data, forcing: compute_derived_quantities(
            data, stepper.sigma_coordinates, stepper.timestep, forcing
        )
        self._compute_derived_for_loaded_data = compute_derived_for_loaded_data

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[BatchData, BatchData]:
        """Return predictions for the time period corresponding to the next batch
        of forcing data."""
        timer = GlobalTimer.get_instance()
        try:
            timer.start("data_loading")
            batch_data = next(self._loader)
        except StopIteration:
            timer.stop("data_loading")
            raise StopIteration
        forcing_data = move_tensordict_to_device(batch_data.data)
        times = batch_data.times
        n_forward_steps = times.sizes["time"] - self._n_ic_timesteps
        timer.stop("data_loading")
        timer.start("run_on_batch")
        prediction = self._stepper.predict(
            self._prognostic_state, forcing_data, n_forward_steps
        )
        prediction_with_ic = _prepend_timesteps(prediction, self._prognostic_state)
        prediction_and_derived = self._compute_derived(prediction_with_ic, forcing_data)
        if self._compute_derived_for_loaded_data:
            forcing_data = self._compute_derived(forcing_data, forcing_data)

        # save prognostic state for next iteration
        self._prognostic_state = {
            k: v[:, -self._n_ic_timesteps :] for k, v in prediction.items()
        }
        self._current_time = times.isel(time=slice(-self._n_ic_timesteps, None))

        # drop initial condition time steps
        forward_forcing_data = {
            k: v[:, self._n_ic_timesteps :] for k, v in forcing_data.items()
        }
        forward_prediction_and_derived = {
            k: v[:, self._n_ic_timesteps :] for k, v in prediction_and_derived.items()
        }
        forward_times = times.isel(time=slice(self._n_ic_timesteps, None))
        timer.stop("run_on_batch")

        return (
            BatchData(data=forward_prediction_and_derived, times=forward_times),
            BatchData(data=forward_forcing_data, times=forward_times),
        )


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


def run_inference(
    stepper: SingleModuleStepper,
    initial_condition: TensorDict,
    initial_times: xr.DataArray,
    forcing_data: GriddedData,
    writer: DataWriter,
    aggregator: InferenceAggregator,
):
    """Run extended inference loop given initial condition and forcing data.

    Args:
        stepper: The model to run inference with.
        initial_condition: Mapping of prognostic names to initial condition tensors of
            shape (n_sample, <horizontal dims>).
        initial_times: Time coordinates for the initial condition.
        forcing_data: GriddedData object which includes a DataLoader which will provide
            windows of forcing data appropriately aligned with the initial condition.
        writer: Data writer for saving the inference results to disk.
        aggregator: Aggregator for collecting and reducing metrics.
    """
    timer = GlobalTimer.get_instance()
    with torch.no_grad():
        n_ic_timesteps = stepper.n_ic_timesteps
        if n_ic_timesteps != 1:
            raise NotImplementedError(
                "data loading for n_ic_timesteps != 1 not implemented yet"
            )
        time_dim = 1
        initial_condition = {
            k: v.unsqueeze(time_dim) for k, v in initial_condition.items()
        }
        looper = Looper(stepper, initial_condition, initial_times, forcing_data.loader)
        i_time = 0
        for prediction_batch, _ in looper:
            timer.start("writer_and_aggregator")
            prediction = prediction_batch.data
            times = prediction_batch.times
            forward_steps_in_memory = times.sizes["time"]
            writer.append_batch(prediction, i_time, times)
            if i_time == 0:
                example_tensor = list(initial_condition.values())[0]
                ic_filled = {}
                for name in prediction:
                    if name in initial_condition:
                        ic_filled[name] = initial_condition[name]
                    else:
                        ic_filled[name] = torch.full_like(
                            example_tensor, fill_value=torch.nan
                        )
                aggregator.record_batch(initial_times, ic_filled, i_time)
            aggregator.record_batch(times, prediction, i_time + 1)
            timer.stop("writer_and_aggregator")
            timer.start("wandb_logging")
            if i_time == 0:
                _log_window_to_wandb(
                    aggregator, window_slice=slice(0, 1), label="inference"
                )
            _log_window_to_wandb(
                aggregator,
                window_slice=slice(i_time + 1, i_time + 1 + forward_steps_in_memory),
                label="inference",
            )
            timer.stop("wandb_logging")
            i_time += forward_steps_in_memory


def run_inference_evaluator(
    aggregator: InferenceEvaluatorAggregator,
    stepper: SingleModuleStepper,
    data: GriddedDataABC[BatchData],
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
):
    timer = GlobalTimer.get_instance()
    if writer is None:
        writer = NullDataWriter()

    with torch.no_grad():
        n_ic_timesteps = stepper.n_ic_timesteps
        for batch in data.loader:
            initial_condition = {
                k: v[:, 0:n_ic_timesteps] for k, v in batch.data.items()
            }
            initial_times = batch.times.isel(time=slice(0, n_ic_timesteps))
            break
        initial_condition = move_tensordict_to_device(initial_condition)
        looper = Looper(
            stepper,
            initial_condition,
            initial_times,
            data.loader,
            compute_derived_for_loaded_data=True,
        )
        snapshot_dims = ["sample"] + data.horizontal_coordinates.dims
        writer.save_initial_condition(
            {k: v[:, 0] for k, v in initial_condition.items()},
            initial_times.isel(time=0),
            snapshot_dims,
        )
        i_time = 0
        for prediction_batch, target_batch in looper:
            timer.start("writer_and_aggregator")
            times = prediction_batch.times
            prediction = prediction_batch.data
            target_data = target_batch.data
            forward_steps_in_memory = times.sizes["time"]
            logging.info(
                f"Inference: processing window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps."
            )
            writer.append_batch(
                target=target_data,
                prediction=prediction,
                start_timestep=i_time,
                batch_times=times,
            )
            # filter forcing variables out of the target data
            target_data = {k: v for k, v in target_data.items() if k in prediction}
            if i_time == 0:
                ic_filled = {}
                for name in target_data:
                    if name in initial_condition:
                        ic_filled[name] = initial_condition[name]
                    else:
                        ic_filled[name] = target_data[name][:, 0:1]
                aggregator.record_batch(
                    loss=np.nan,
                    time=initial_times,
                    target_data=ic_filled,
                    gen_data=ic_filled,
                    target_data_norm=stepper.normalizer.normalize(ic_filled),
                    gen_data_norm=stepper.normalizer.normalize(ic_filled),
                    i_time_start=i_time,
                )
            aggregator.record_batch(
                loss=np.nan,
                time=times,
                target_data=target_data,
                gen_data=prediction,
                target_data_norm=stepper.normalizer.normalize(target_data),
                gen_data_norm=stepper.normalizer.normalize(prediction),
                i_time_start=i_time + 1,
            )
            timer.stop("writer_and_aggregator")
            timer.start("wandb_logging")
            if i_time == 0:
                _log_window_to_wandb(
                    aggregator, window_slice=slice(0, 1), label="inference"
                )
            _log_window_to_wandb(
                aggregator,
                window_slice=slice(i_time + 1, i_time + 1 + forward_steps_in_memory),
                label="inference",
            )
            i_time += forward_steps_in_memory
            timer.stop("wandb_logging")


def run_dataset_comparison(
    aggregator: InferenceEvaluatorAggregator,
    normalizer: StandardNormalizer,
    prediction_data: GriddedData,
    target_data: GriddedData,
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
):
    if writer is None:
        writer = NullDataWriter()
    n_forward_steps = target_data.n_forward_steps

    timer = GlobalTimer.get_instance()
    timer.start("data_loading")
    i_time = 0
    for pred, target in zip(prediction_data.loader, target_data.loader):
        timer.stop("data_loading")

        timer.start("run_on_batch")
        forward_steps_in_memory = list(pred.data.values())[0].size(1) - 1
        logging.info(
            f"Inference: starting window spanning {i_time}"
            f" to {i_time + forward_steps_in_memory} steps,"
            f" out of total {n_forward_steps}."
        )
        pred_window_data = dict(pred.device_data)
        target_window_data = dict(target.device_data)
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
        target_times = target.times
        timer.stop("run_on_batch")

        timer.start("writer_and_aggregator")
        stepped_without_ic = stepped.remove_initial_condition(1)
        target_times_without_ic = target_times.isel(time=slice(1, None))
        if i_time == 0:
            i_time_aggregator = i_time
            stepped_for_agg = stepped
            target_times_for_agg = target_times
        else:
            i_time_aggregator = i_time + 1
            stepped_for_agg = stepped_without_ic
            target_times_for_agg = target_times_without_ic

        # Do not include the initial condition in the data writers
        writer.append_batch(
            target=stepped_without_ic.target_data,
            prediction=stepped_without_ic.gen_data,
            start_timestep=i_time,
            batch_times=target_times_without_ic,
        )

        # record metrics, includes the initial condition
        aggregator.record_batch(
            loss=float(stepped_for_agg.metrics["loss"]),
            time=target_times_for_agg,
            target_data=stepped_for_agg.target_data,
            gen_data=stepped_for_agg.gen_data,
            target_data_norm=stepped_for_agg.target_data_norm,
            gen_data_norm=stepped_for_agg.gen_data_norm,
            i_time_start=i_time_aggregator,
        )
        timer.stop("writer_and_aggregator")

        timer.start("wandb_logging")
        _log_window_to_wandb(
            aggregator,
            window_slice=slice(
                i_time_aggregator, i_time_aggregator + len(target_times["time"])
            ),
            label="inference",
        )
        timer.stop("wandb_logging")

        timer.start("data_loading")
        i_time += forward_steps_in_memory

    timer.stop("data_loading")


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
