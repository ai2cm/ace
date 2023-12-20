import logging
from typing import Dict, Mapping, Optional, Protocol, Union

import torch
import xarray as xr

from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.data_loading.data_typing import GriddedData, SigmaCoordinates
from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
from fme.core.optimization import NullOptimization
from fme.core.stepper import SteppedData

from .data_writer import DataWriter, NullDataWriter
from .derived_variables import (
    compute_derived_quantities,
    compute_stepped_derived_quantities,
)


class EnsembleBatch:
    def __init__(
        self,
        i_batch: int,
        n_forward_steps: int,
        writer: Union[DataWriter, NullDataWriter],
    ):
        self.i_batch = i_batch
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
        batch_size = tensor_shape[0]
        self.writer.append_batch(
            target=data,
            prediction=gen_data,
            start_timestep=self.i_time,
            start_sample=self.i_batch * batch_size,
            batch_times=batch_times,
        )
        self.i_time += tensor_shape[1]
        if self.i_time < self.n_forward_steps:  # only store if needed
            # store the end of the time window as
            # initial condition for the next segment.
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


class DataLoaderFactory(Protocol):
    def __call__(self, window_time_slice: Optional[slice] = None) -> GriddedData:
        ...


def _inference_internal_loop(
    stepped: SteppedData,
    i_time: int,
    sigma_coords: SigmaCoordinates,
    aggregator: InferenceAggregator,
    batch_manager: EnsembleBatch,
    batch_times: xr.DataArray,
):
    """Do operations that need to be done on each time step of the inference loop.

    This function exists to de-duplicate code between run_inference and
    run_data_inference."""

    # for non-initial windows, we want to record only the new data
    # and discard the initial sample of the window
    if i_time > 0:
        stepped = stepped.remove_initial_condition()
        batch_times = batch_times.isel(time=slice(1, None))
        i_time_aggregator = i_time + 1
    else:
        i_time_aggregator = i_time
    # record raw data for the batch, and store the final state
    # for the next segment
    batch_manager.append(stepped.target_data, stepped.gen_data, batch_times)
    # record metrics
    aggregator.record_batch(
        loss=float(stepped.metrics["loss"]),
        target_data=stepped.target_data,
        gen_data=stepped.gen_data,
        target_data_norm=stepped.target_data_norm,
        gen_data_norm=stepped.gen_data_norm,
        i_time_start=i_time_aggregator,
    )


def run_inference(
    aggregator: InferenceAggregator,
    stepper: SingleModuleStepper,
    data_loader_factory: DataLoaderFactory,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
):
    if writer is None:
        writer = NullDataWriter()
    example_valid_data_loader = data_loader_factory(window_time_slice=slice(0, 1))
    batch_managers = [
        EnsembleBatch(i_batch=i_batch, n_forward_steps=n_forward_steps, writer=writer)
        for i_batch in range(len(example_valid_data_loader.loader))
    ]
    if len(batch_managers) == 0:
        raise ValueError("Data loader must have at least one batch")

    with torch.no_grad():
        # We have data batches with long windows, where all data for a
        # given batch does not fit into memory at once, so we window it in time
        # and run the model on each window in turn.
        #
        # All batches for a given time also may not fit in memory.
        #
        # For each time window, we process it for each batch, and keep track of the
        # final state of each batch. We then use this as the initial condition
        # for the next time window.
        device = get_device()

        for i_time in range(0, n_forward_steps, forward_steps_in_memory):
            # need one more timestep for initial condition
            time_slice = slice(i_time, i_time + forward_steps_in_memory + 1)
            logging.info(
                f"Inference: starting window spanning {time_slice.start}"
                f" to {time_slice.stop} steps, out of total {n_forward_steps}."
            )
            # data loader is a sequence of batches, so we need a new one for each
            # time window
            valid_data_loader = data_loader_factory(window_time_slice=time_slice)
            sigma_coords = valid_data_loader.sigma_coordinates
            for batch, batch_manager in zip(valid_data_loader.loader, batch_managers):
                data = {
                    key: value.to(device, dtype=torch.float)
                    for key, value in batch.data.items()
                }
                # must compute derived targets before applying initial condition
                # so that time differencing is still valid
                target_data = compute_derived_quantities(data, sigma_coords)
                # overwrite the first timestep with the last generated timestep
                # from the previous segment
                batch_manager.apply_initial_condition(data)
                stepped = stepper.run_on_batch(
                    data,
                    NullOptimization(),
                    n_forward_steps=forward_steps_in_memory,
                )
                stepped.target_data = target_data
                stepped.gen_data = compute_derived_quantities(
                    stepped.gen_data, sigma_coords
                )
                _inference_internal_loop(
                    stepped,
                    i_time,
                    sigma_coords,
                    aggregator,
                    batch_manager,
                    batch.times,
                )


def remove_initial_condition(
    data: Mapping[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {key: value[:, 1:] for key, value in data.items()}


def run_dataset_inference(
    aggregator: InferenceAggregator,
    normalizer: StandardNormalizer,
    prediction_data_loader_factory: DataLoaderFactory,
    target_data_loader_factory: DataLoaderFactory,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
):
    if writer is None:
        writer = NullDataWriter()
    example_valid_data_loader = target_data_loader_factory(
        window_time_slice=slice(0, 1)
    )
    batch_managers = [
        EnsembleBatch(i_batch, n_forward_steps, writer)
        for i_batch in range(len(example_valid_data_loader.loader))
    ]
    if len(batch_managers) == 0:
        raise ValueError("Data loader must have at least one batch")

    device = get_device()
    with torch.no_grad():
        # We have data batches with long windows, where all data for a
        # given batch does not fit into memory at once, so we window it in time
        # and run the model on each window in turn.
        #
        # All batches for a given time also may not fit in memory.
        #
        # For each time window, we process it for each batch, and keep track of the
        # final state of each batch. We then use this as the initial condition
        # for the next time window.
        for i_time in range(0, n_forward_steps, forward_steps_in_memory):
            # need one more timestep for initial condition
            time_slice = slice(i_time, i_time + forward_steps_in_memory + 1)
            logging.info(
                f"Inference: starting window spanning {time_slice.start}"
                f" to {time_slice.stop} steps, out of total {n_forward_steps}."
            )
            # data loader is a sequence of batches, so we need a new one for each
            # time window
            valid_data_loader = target_data_loader_factory(window_time_slice=time_slice)
            predicted_data_loader = prediction_data_loader_factory(
                window_time_slice=time_slice
            )
            sigma_coords = valid_data_loader.sigma_coordinates
            for valid_batch, predicted_batch, batch_manager in zip(
                valid_data_loader.loader, predicted_data_loader.loader, batch_managers
            ):
                valid_data = {
                    key: value.to(device) for key, value in valid_batch.data.items()
                }
                predicted_data = {
                    key: value.to(device) for key, value in predicted_batch.data.items()
                }
                stepped = SteppedData(
                    {"loss": torch.tensor(float("nan"))},
                    predicted_data,
                    valid_data,
                    normalizer.normalize(predicted_data),
                    normalizer.normalize(valid_data),
                )
                stepped = compute_stepped_derived_quantities(stepped, sigma_coords)
                _inference_internal_loop(
                    stepped,
                    i_time,
                    sigma_coords,
                    aggregator,
                    batch_manager,
                    valid_batch.times,
                )
