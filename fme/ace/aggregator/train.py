import torch

from fme.ace.stepper import TrainOutput
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import AggregatorABC


class TrainAggregator(AggregatorABC[TrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())

    @torch.no_grad()
    def record_batch(self, batch: TrainOutput):
        self._loss += batch.metrics["loss"]
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, torch.Tensor]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {f"{label}/mean/loss": self._loss / self._n_batches}
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None) -> None:
        pass
