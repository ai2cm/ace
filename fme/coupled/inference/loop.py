import logging
from collections.abc import Callable
from typing import Protocol

from fme.core.generics.aggregator import InferenceAggregatorABC, InferenceLogs
from fme.core.generics.inference import get_record_to_wandb
from fme.core.generics.writer import NullDataWriter
from fme.core.timing import GlobalTimer
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.inference.data_writer import CoupledPairedDataWriter
from fme.coupled.typing_ import CoupledNames


class CoupledDeriver(Protocol):
    """
    Interface for deriving forward data from CoupledBatchData during
    dataset comparison (remove IC, compute derived variables).
    """

    def get_forward_data(
        self,
        data: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> CoupledBatchData: ...


def run_coupled_dataset_comparison(
    aggregator: InferenceAggregatorABC[
        CoupledPairedData | CoupledPrognosticState,
        CoupledPairedData,
    ],
    prediction_data: InferenceGriddedData,
    target_data: InferenceGriddedData,
    deriver: CoupledDeriver,
    writer: CoupledPairedDataWriter | NullDataWriter | None = None,
    record_logs: Callable[[InferenceLogs], None] | None = None,
    restrict_to_all_names: CoupledNames | None = None,
) -> None:
    if record_logs is None:
        record_logs = get_record_to_wandb(label="inference")
    if writer is None:
        writer = NullDataWriter()

    def _restrict_to_all_names(batch: CoupledBatchData) -> CoupledBatchData:
        if restrict_to_all_names is None:
            return batch
        return CoupledBatchData(
            ocean_data=batch.ocean_data.subset_names(restrict_to_all_names.ocean),
            atmosphere_data=batch.atmosphere_data.subset_names(
                restrict_to_all_names.atmosphere
            ),
        )

    timer = GlobalTimer.get_instance()
    timer.start("data_loading")
    i_time = 0
    n_windows = min(len(prediction_data.loader), len(target_data.loader))
    for i, (pred, target) in enumerate(zip(prediction_data.loader, target_data.loader)):
        timer.stop()
        if i_time == 0:
            with timer.context("aggregator"):
                pred_ic = prediction_data.initial_condition.as_batch_data()
                target_ic = target_data.initial_condition.as_batch_data()
                pred_ic = _restrict_to_all_names(pred_ic)
                target_ic = _restrict_to_all_names(target_ic)
                logs = aggregator.record_initial_condition(
                    initial_condition=CoupledPairedData.from_coupled_batch_data(
                        prediction=pred_ic,
                        reference=target_ic,
                    ),
                )
            with timer.context("wandb_logging"):
                record_logs(logs)

        forward_steps_in_memory = list(pred.ocean_data.data.values())[0].size(1) - 1
        logging.info(
            f"Inference: Processing window {i + 1} of {n_windows}"
            f" spanning {i_time} to {i_time + forward_steps_in_memory} steps."
        )
        pred = deriver.get_forward_data(pred, compute_derived_variables=True)
        target = deriver.get_forward_data(target, compute_derived_variables=True)
        pred = _restrict_to_all_names(pred)
        target = _restrict_to_all_names(target)
        paired_data = CoupledPairedData.from_coupled_batch_data(
            prediction=pred,
            reference=target,
        )

        with timer.context("data_writer"):
            writer.append_batch(batch=paired_data)
        with timer.context("aggregator"):
            logs = aggregator.record_batch(data=paired_data)

        with timer.context("wandb_logging"):
            record_logs(logs)

        timer.start("data_loading")
        i_time += forward_steps_in_memory

    timer.stop()
