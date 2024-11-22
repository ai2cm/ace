import abc
import logging
from pathlib import Path
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

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
    CurrentDevice,
    PairedData,
    PrognosticState,
)
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLogs,
)
from fme.core.generics.inference import InferenceDataABC, InferenceStepperABC
from fme.core.generics.writer import WriterABC
from fme.core.wandb import WandB

from .data_writer import NullDataWriter, PairedDataWriter

PS = TypeVar("PS")
BD = TypeVar("BD")


class Looper(Generic[PS, BD]):
    """
    Class for stepping a model forward arbitarily many times.
    """

    def __init__(
        self,
        stepper: InferenceStepperABC[PS, BD],
        data: InferenceDataABC[PS, BD],
        compute_derived_for_loaded_data: bool = False,
    ):
        """
        Args:
            stepper: The stepper to use.
            data: The data to use.
            compute_derived_for_loaded_data: Whether to compute derived variables for
                the data returned by the loader.
        """
        self._stepper = stepper
        self._prognostic_state = data.initial_condition
        self._len = len(data.loader)
        self._loader = iter(data.loader)
        self._compute_derived_for_loaded_data = compute_derived_for_loaded_data

    def __iter__(self):
        return self

    def __len__(self):
        return self._len

    def __next__(self) -> Tuple[BD, BD]:
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
        prediction, self._prognostic_state = self._stepper.predict(
            self._prognostic_state,
            forcing_data,
            compute_derived_variables=True,
        )
        timer.stop("forward_prediction")
        forcing_data = self._stepper.get_forward_data(
            forcing_data,
            compute_derived_variables=self._compute_derived_for_loaded_data,
        )
        return (
            prediction,
            forcing_data,
        )


def get_record_to_wandb(label: str = "") -> Callable[[InferenceLogs], None]:
    wandb = WandB.get_instance()
    step = 0

    def record_logs(logs: InferenceLogs):
        nonlocal step
        for j, log in enumerate(logs):
            if len(log) > 0:
                if label != "":
                    log = {f"{label}/{k}": v for k, v in log.items()}
                wandb.log(log, step=step + j)
        step += len(logs)

    return record_logs


def run_inference(
    stepper: InferenceStepperABC[PS, BD],
    data: InferenceDataABC[PS, BD],
    writer: WriterABC[PS, BD],
    aggregator: InferenceAggregatorABC[PS, BD],
    record_logs: Optional[Callable[[InferenceLogs], None]] = None,
):
    """Run extended inference loop given initial condition and forcing data.

    Args:
        stepper: The model to run inference with.
        data: Provides an initial condition and appropriately aligned windows of
            forcing data.
        writer: Data writer for saving the inference results to disk.
        aggregator: Aggregator for collecting and reducing metrics.
        record_logs: Function for recording logs. By default, logs are recorded to
            wandb.
    """
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference")
    timer = GlobalTimer.get_instance()
    with torch.no_grad():
        looper = Looper(stepper, data)
        with timer.context("aggregator"):
            logs = aggregator.record_initial_condition(
                initial_condition=data.initial_condition,
            )
        with timer.context("wandb_logging"):
            record_logs(logs)
        with timer.context("data_writer"):
            writer.save_initial_condition(
                data.initial_condition,
            )
        n_windows = len(looper)
        for i, (prediction_batch, _) in enumerate(looper):
            logging.info(
                f"Inference: processing output from window {i + 1} of {n_windows}."
            )
            with timer.context("data_writer"):
                writer.append_batch(
                    batch=prediction_batch,
                )
            with timer.context("aggregator"):
                logs = aggregator.record_batch(
                    data=prediction_batch,
                )
            with timer.context("wandb_logging"):
                record_logs(logs)


def run_inference_evaluator(
    aggregator: InferenceAggregatorABC[
        PrognosticState[CurrentDevice], PairedData[CurrentDevice]
    ],
    stepper: SingleModuleStepper,
    data: InferenceDataABC[PrognosticState[CurrentDevice], BatchData[CurrentDevice]],
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
    record_logs: Optional[Callable[[InferenceLogs], None]] = None,
):
    """
    Run inference evaluator loop.

    Args:
        aggregator: Aggregator for collecting and reducing metrics.
        stepper: The model to run inference with.
        data: Provides an initial condition and appropriately aligned windows of
            forcing and target data.
        writer: Data writer for saving the inference results to disk.
        record_logs: Function for recording logs. By default, logs are recorded to
            wandb.
    """
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference")
    timer = GlobalTimer.get_instance()
    if writer is None:
        writer = NullDataWriter()

    with torch.no_grad():
        looper = Looper(
            stepper,
            data,
            compute_derived_for_loaded_data=True,
        )
        with timer.context("aggregator"):
            logs = aggregator.record_initial_condition(
                initial_condition=data.initial_condition,
            )
        with timer.context("wandb_logging"):
            record_logs(logs)
        with timer.context("data_writer"):
            writer.save_initial_condition(
                data.initial_condition,
            )
        i_time = 0
        for prediction_batch, target_batch in looper:
            times = prediction_batch.times
            forward_steps_in_memory = times.sizes["time"]
            logging.info(
                f"Inference: processing window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} forward steps."
            )
            paired_data = PairedData.from_batch_data(prediction_batch, target_batch)
            with timer.context("data_writer"):
                writer.append_batch(
                    batch=paired_data,
                )
            with timer.context("aggregator"):
                logs = aggregator.record_batch(
                    data=paired_data,
                )
            with timer.context("wandb_logging"):
                record_logs(logs)
            i_time += forward_steps_in_memory


class DeriverABC(abc.ABC):
    """
    Abstract base class for processing data during dataset comparison.
    """

    @abc.abstractmethod
    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData:
        ...

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int:
        ...


def run_dataset_comparison(
    aggregator: InferenceAggregatorABC[PairedData, PairedData],
    prediction_data: InferenceDataABC[
        PrognosticState[CurrentDevice], BatchData[CurrentDevice]
    ],
    target_data: InferenceDataABC[
        PrognosticState[CurrentDevice], BatchData[CurrentDevice]
    ],
    deriver: DeriverABC,
    writer: Optional[Union[PairedDataWriter, NullDataWriter]] = None,
    record_logs: Optional[Callable[[InferenceLogs], None]] = None,
):
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference")
    if writer is None:
        writer = NullDataWriter()

    timer = GlobalTimer.get_instance()
    timer.start("data_loading")
    i_time = 0
    n_windows = min(len(prediction_data.loader), len(target_data.loader))
    for i, (pred, target) in enumerate(zip(prediction_data.loader, target_data.loader)):
        timer.stop("data_loading")
        if i_time == 0:
            with timer.context("aggregator"):
                logs = aggregator.record_initial_condition(
                    initial_condition=PairedData.from_batch_data(
                        prediction=prediction_data.initial_condition.as_batch_data(),
                        target=target_data.initial_condition.as_batch_data(),
                    ),
                )
            with timer.context("wandb_logging"):
                record_logs(logs)

        forward_steps_in_memory = list(pred.data.values())[0].size(1) - 1
        logging.info(
            f"Inference: Processing window {i + 1} of {n_windows}"
            f" spanning {i_time} to {i_time + forward_steps_in_memory} steps."
        )
        pred = deriver.get_forward_data(pred, compute_derived_variables=True)
        target = deriver.get_forward_data(target, compute_derived_variables=True)

        with timer.context("data_writer"):
            # Do not include the initial condition in the data writers
            writer.append_batch(
                batch=PairedData.from_batch_data(pred, target),
            )
        with timer.context("aggregator"):
            # record metrics, includes the initial condition
            logs = aggregator.record_batch(
                data=PairedData.from_batch_data(pred, target),
            )

        with timer.context("wandb_logging"):
            record_logs(logs)

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
