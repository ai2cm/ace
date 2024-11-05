import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
import torch

from fme.ace.inference.timing import GlobalTimer
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import (
    InferenceAggregator,
    InferenceEvaluatorAggregator,
)
from fme.core.data_loading.batch_data import (
    BatchData,
    GriddedData,
    GriddedDataABC,
    PairedData,
)
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
)
from fme.core.generics.state import PrognosticStateABC
from fme.core.normalizer import StandardNormalizer
from fme.core.stepper import TrainOutput
from fme.core.wandb import WandB

from .data_writer import DataWriter, NullDataWriter, PairedDataWriter


class Looper:
    """
    Class for stepping a model forward arbitarily many times.
    """

    def __init__(
        self,
        stepper: SingleModuleStepper,
        initial_condition: PrognosticStateABC[BatchData],
        loader: Iterable[BatchData],
        compute_derived_for_loaded_data: bool = False,
    ):
        """
        Args:
            stepper: The stepper to use.
            initial_condition: The initial condition data.
            loader: The loader for the forcing, and possibly target, data.
            compute_derived_for_loaded_data: Whether to compute derived variables for
                the data returned by the loader.
        """
        self._stepper = stepper
        self._n_ic_timesteps = stepper.n_ic_timesteps
        self._prognostic_state = initial_condition
        self._loader = iter(loader)
        self._compute_derived_for_loaded_data = compute_derived_for_loaded_data

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[BatchData, BatchData]:
        """Return predictions for the time period corresponding to the next batch
        of forcing data. Also returns the forcing data."""
        timer = GlobalTimer.get_instance()
        try:
            timer.start("data_loading")
            forcing_data = next(self._loader)
        except StopIteration:
            timer.stop("data_loading")
            raise StopIteration
        timer.stop("data_loading")
        timer.start("forward_prediction")
        prediction, new_prognostic_state = self._stepper.predict(
            self._prognostic_state, forcing_data
        )
        timer.stop("forward_prediction")
        timer.start("compute_derived_variables")
        forward_prediction_and_derived = (
            prediction.prepend(self._prognostic_state)
            .compute_derived_variables(forcing_data)
            .remove_initial_condition(self._n_ic_timesteps)
        )
        if self._compute_derived_for_loaded_data:
            forcing_data = forcing_data.compute_derived_variables(forcing_data)

        forcing_data = forcing_data.remove_initial_condition(self._n_ic_timesteps)
        timer.stop("compute_derived_variables")
        self._prognostic_state = new_prognostic_state

        return (
            forward_prediction_and_derived,
            forcing_data,
        )


class HasWandBInferenceLogData(Protocol):
    def get_inference_logs_slice(
        self, label: str, step_slice: slice
    ) -> List[Dict[str, Any]]:
        ...

    @property
    def log_time_series(self) -> bool:
        ...


def _log_window_to_wandb(
    aggregator: HasWandBInferenceLogData,
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
    initial_condition: PrognosticStateABC[BatchData],
    forcing_data: GriddedData,
    writer: DataWriter,
    aggregator: InferenceAggregatorABC[PrognosticStateABC[BatchData], BatchData],
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
        looper = Looper(stepper, initial_condition, forcing_data.loader)
        i_time = 0
        aggregator.record_initial_condition(
            initial_condition=initial_condition,
            normalize=stepper.normalizer.normalize,
        )
        writer.save_initial_condition(
            initial_condition,
        )
        for prediction_batch, _ in looper:
            timer.start("data_writer")
            times = prediction_batch.times
            forward_steps_in_memory = times.sizes["time"]
            logging.info(
                f"Inference: processing window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps."
            )
            writer.append_batch(
                batch=prediction_batch,
                start_timestep=i_time,
            )
            timer.stop("data_writer")
            timer.start("aggregator")
            aggregator.record_batch(
                prediction_batch,
                normalize=stepper.normalizer.normalize,
                i_time_start=i_time + stepper.n_ic_timesteps,
            )
            timer.stop("aggregator")
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
    aggregator: InferenceAggregatorABC[PrognosticStateABC[BatchData], PairedData],
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
            initial_condition = batch.get_start(
                prognostic_names=stepper.prognostic_names,
                n_ic_timesteps=n_ic_timesteps,
            )
            break
        else:
            raise ValueError("No data in data.loader")
        looper = Looper(
            stepper,
            initial_condition,
            data.loader,
            compute_derived_for_loaded_data=True,
        )
        aggregator.record_initial_condition(
            initial_condition=initial_condition,
            normalize=stepper.normalizer.normalize,
        )
        writer.save_initial_condition(
            initial_condition,
        )
        i_time = 0
        for prediction_batch, target_batch in looper:
            timer.start("data_writer")
            times = prediction_batch.times
            forward_steps_in_memory = times.sizes["time"]
            logging.info(
                f"Inference: processing window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps."
            )
            paired_data = PairedData.from_batch_data(prediction_batch, target_batch)
            writer.append_batch(
                batch=paired_data,
                start_timestep=i_time,
            )
            timer.stop("data_writer")
            timer.start("aggregator")
            aggregator.record_batch(
                data=paired_data,
                normalize=stepper.normalizer.normalize,
                i_time_start=i_time + stepper.n_ic_timesteps,
            )
            timer.stop("aggregator")
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
    aggregator: InferenceAggregatorABC[PrognosticStateABC[BatchData], PairedData],
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

        timer.start("forward_prediction")
        forward_steps_in_memory = list(pred.data.values())[0].size(1) - 1
        logging.info(
            f"Inference: starting window spanning {i_time}"
            f" to {i_time + forward_steps_in_memory} steps,"
            f" out of total {n_forward_steps}."
        )
        pred_window_data = dict(pred.device_data)
        target_window_data = dict(target.device_data)
        stepped = TrainOutput(
            {"loss": torch.tensor(float("nan"))},
            pred_window_data,
            target_window_data,
            times=pred.times,
            normalize=normalizer.normalize,
        )
        timer.stop("forward_prediction")
        timer.start("compute_derived_variables")
        stepped = stepped.compute_derived_variables()
        target_times = target.times
        timer.stop("compute_derived_variables")

        timer.start("data_writer")
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
            batch=PairedData(
                target=stepped_without_ic.target_data,
                prediction=stepped_without_ic.gen_data,
                times=target_times_without_ic,
            ),
            start_timestep=i_time,
        )
        timer.stop("data_writer")
        timer.start("aggregator")
        # record metrics, includes the initial condition
        aggregator.record_batch(
            data=PairedData(
                prediction=stepped_for_agg.gen_data,
                target=stepped_for_agg.target_data,
                times=target_times_for_agg,
            ),
            normalize=normalizer.normalize,
            i_time_start=i_time_aggregator,
        )
        timer.stop("aggregator")

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
