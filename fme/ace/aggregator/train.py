import torch

from fme.ace.aggregator.one_step.spectrum import PairedSphericalPowerSpectrumAggregator
from fme.ace.stepper import TrainOutput
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import AggregatorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.tensors import fold_ensemble_dim


class TrainAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self, operations: GriddedOperations):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())
        self._paired_aggregators = [
            PairedSphericalPowerSpectrumAggregator(
                gridded_operations=operations,
                report_plot=False,
            )
        ]

    @torch.no_grad()
    def record_batch(self, batch: TrainOutput):
        self._loss += batch.metrics["loss"]
        self._n_batches += 1
        target_data, _ = fold_ensemble_dim(batch.target_data)
        gen_data, _ = fold_ensemble_dim(batch.gen_data)
        for aggregator in self._paired_aggregators:
            aggregator.record_batch(target_data=target_data, gen_data=gen_data)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        if self._n_batches > 0:
            logs[f"{label}/mean/loss"] = self._loss / self._n_batches
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        for aggregator in self._paired_aggregators:
            logs.update(
                {f"{label}/{k}": v for k, v in aggregator.get_logs(label).items()}
            )
        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
