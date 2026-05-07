import torch

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorDict, TensorMapping


class TensorDictAccumulator:
    """Accumulates per-batch TensorDict values into a running sum.

    The set of keys is determined by the first ``add`` call; subsequent calls
    must provide the same keys.

    Two flavors of readers are exposed:

    * ``get_sum`` / ``get_mean`` return the local running sum / mean, or
      ``None`` if nothing has been added yet. They do not reduce across ranks.
    * ``get_distributed_sum`` / ``get_distributed_mean`` reduce across ranks
      and return a key-sorted dict. They raise ``ValueError`` if nothing has
      been added yet.
    """

    def __init__(self, device: torch.device | None = None):
        self._device = device
        self._sum: TensorDict | None = None
        self._count: int = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, values: TensorMapping) -> None:
        if self._sum is None:
            self._sum = {
                name: torch.zeros_like(tensor, device=self._device)
                for name, tensor in values.items()
            }
        elif set(self._sum) != set(values):
            raise ValueError(
                "keys changed between add() calls: "
                f"previous={sorted(self._sum)}, current={sorted(values)}"
            )
        for name, tensor in values.items():
            self._sum[name] += tensor
        self._count += 1

    def get_sum(self) -> TensorDict | None:
        if self._sum is None:
            return None
        return dict(self._sum)

    def get_mean(self) -> TensorDict | None:
        if self._sum is None or self._count == 0:
            return None
        return {name: tensor / self._count for name, tensor in self._sum.items()}

    def get_distributed_sum(self) -> TensorDict:
        if self._sum is None:
            raise ValueError("No values have been added to the accumulator")
        dist = Distributed.get_instance()
        return {k: dist.reduce_sum(self._sum[k]) for k in sorted(self._sum)}

    def get_distributed_mean(self) -> TensorDict:
        if self._sum is None or self._count == 0:
            raise ValueError("No values have been added to the accumulator")
        dist = Distributed.get_instance()
        return {
            k: dist.reduce_mean(self._sum[k] / self._count) for k in sorted(self._sum)
        }
