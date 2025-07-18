import abc
import logging
from collections.abc import Callable

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.inference.data_writer import PairedDataWriter
from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.inference import get_record_to_wandb
from fme.core.generics.writer import NullDataWriter
from fme.core.timing import GlobalTimer


class DeriverABC(abc.ABC):
    """
    Abstract base class for processing data during dataset comparison.
    """

    @abc.abstractmethod
    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData: ...

    @property
    @abc.abstractmethod
    def n_ic_timesteps(self) -> int: ...


def run_dataset_comparison(
    aggregator: InferenceAggregatorABC[PairedData, PairedData],
    prediction_data: InferenceDataABC[PrognosticState, BatchData],
    target_data: InferenceDataABC[PrognosticState, BatchData],
    deriver: DeriverABC,
    writer: PairedDataWriter | NullDataWriter | None = None,
    record_logs: Callable[[InferenceLogs], None] | None = None,
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
        timer.stop()
        if i_time == 0:
            with timer.context("aggregator"):
                logs = aggregator.record_initial_condition(
                    initial_condition=PairedData.from_batch_data(
                        prediction=prediction_data.initial_condition.as_batch_data(),
                        reference=target_data.initial_condition.as_batch_data(),
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
        paired_data = PairedData.from_batch_data(prediction=pred, reference=target)

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

        timer.start("data_loading")
        i_time += forward_steps_in_memory

    timer.stop()
