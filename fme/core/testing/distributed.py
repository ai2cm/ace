import contextlib

import torch

from fme.core import distributed


class MockDistributed:
    def __init__(self, fill_value: float, world_size: int):
        self.world_size = world_size
        self.fill_value = fill_value
        self.reduce_called = False

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.fill_(self.fill_value)
        self.reduce_called = True
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 1

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.fill_(self.fill_value)
        self.reduce_called = True
        return tensor

    def is_root(self) -> bool:
        return True

    def is_distributed(self) -> bool:
        return True

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        return [tensor for i in range(self.world_size)]

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        """
        Note this uses the actual implementation but mocks the underlying
        distributed calls.
        """
        return distributed.gather_irregular(
            tensor, self.reduce_max, self.gather, is_distributed=self.is_distributed()
        )


@contextlib.contextmanager
def mock_distributed(fill_value: float = 0.0, world_size: int = 1):
    """
    Mock the distributed singleton to return a MockDistributed object.

    This is useful for testing that metrics are reduced across processes.

    It will make it so that when any tensor is reduced, it is filled with
    the given fill_value, which can be checked for in tests.
    """
    original = distributed.singleton
    distributed.singleton = MockDistributed(
        fill_value=fill_value, world_size=world_size
    )  # type: ignore
    try:
        yield distributed.singleton
    finally:
        distributed.singleton = original
