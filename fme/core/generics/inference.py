import logging
from collections.abc import Callable, Iterator
from typing import Generic, Protocol, TypeVar

import torch

from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.writer import NullDataWriter, WriterABC
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

PS = TypeVar("PS")  # prognostic state
FD = TypeVar("FD", contravariant=True)  # forcing data
SD = TypeVar("SD", covariant=True)  # stepped data


class PredictFunction(Protocol, Generic[PS, FD, SD]):
    def __call__(
        self,
        initial_condition: PS,
        forcing: FD,
        compute_derived_variables: bool = False,
    ) -> tuple[SD, PS]: ...


class Looper(Generic[PS, FD, SD]):
    """
    Class for stepping a model forward arbitarily many times.
    """

    def __init__(
        self,
        predict: PredictFunction[PS, FD, SD],
        data: InferenceDataABC[PS, FD],
    ):
        """
        Args:
            predict: The prediction function to use.
            data: The data to use.
        """
        self._predict = predict
        self._prognostic_state = data.initial_condition
        self._len = len(data.loader)
        self._loader = iter(data.loader)

    def __iter__(self) -> Iterator[SD]:
        return self

    def __len__(self) -> int:
        return self._len

    def __next__(self) -> SD:
        """Return predictions for the time period corresponding to the next batch
        of forcing data. Also returns the forcing data.
        """
        timer = GlobalTimer.get_instance()
        with timer.context("data_loading"):
            try:
                forcing_data = next(self._loader)
            except StopIteration:
                raise StopIteration
        output_data, self._prognostic_state = self._predict(
            self._prognostic_state,
            forcing=forcing_data,
            compute_derived_variables=True,
        )
        return output_data

    def get_prognostic_state(self) -> PS:
        return self._prognostic_state


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
    predict: PredictFunction[PS, FD, SD],
    data: InferenceDataABC[PS, FD],
    aggregator: InferenceAggregatorABC[PS, SD],
    writer: WriterABC[PS, SD] | None = None,
    record_logs: Callable[[InferenceLogs], None] | None = None,
):
    """Run extended inference loop given initial condition and forcing data.

    Args:
        predict: The prediction function to use.
        data: Provides an initial condition and appropriately aligned windows of
            forcing data.
        aggregator: Aggregator for collecting and reducing metrics.
        writer: Data writer for saving the inference results to disk.
        record_logs: Function for recording logs. By default, logs are recorded to
            wandb.
    """
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference")
    if writer is None:
        writer = NullDataWriter()
    timer = GlobalTimer.get_instance()
    with torch.no_grad():
        looper = Looper(predict=predict, data=data)
        with timer.context("aggregator"):
            logs = aggregator.record_initial_condition(
                initial_condition=data.initial_condition,
            )
        with timer.context("wandb_logging"):
            record_logs(logs)
        with timer.context("data_writer"):
            writer.write(data.initial_condition, "initial_condition.nc")
        n_windows = len(looper)
        for i, batch in enumerate(looper):
            logging.info(
                f"Inference: processing output from window {i + 1} of {n_windows}."
            )
            with timer.context("data_writer"):
                writer.append_batch(
                    batch=batch,
                )
            with timer.context("aggregator"):
                logs = aggregator.record_batch(
                    data=batch,
                )
            with timer.context("wandb_logging"):
                record_logs(logs)
        with timer.context("data_writer"):
            prognostic_state = looper.get_prognostic_state()
            writer.write(prognostic_state, "restart.nc")
