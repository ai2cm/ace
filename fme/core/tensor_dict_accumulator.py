import torch

from fme.core.typing_ import TensorDict, TensorMapping


class TensorDictAccumulator:
    """Accumulates per-batch TensorDict values into a running sum.

    The set of keys is determined by the first ``add`` call; subsequent calls
    must provide the same keys. Distributed reduction and normalization are
    not performed here and are the caller's responsibility.
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
