import torch
import contextlib
from fme.core import distributed


class MockDistributed:
    def __init__(self, fill_value: float):
        self.fill_value = fill_value
        self.reduce_called = False

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.fill_(self.fill_value)
        self.reduce_called = True
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.fill_(self.fill_value)
        self.reduce_called = True
        return tensor

    def is_root(self) -> bool:
        return True

    def is_distributed(self) -> bool:
        return True


@contextlib.contextmanager
def mock_distributed(fill_value: float):
    """
    Mock the distributed singleton to return a MockDistributed object.

    This is useful for testing that metrics are reduced across processes.

    It will make it so that when any tensor is reduced, it is filled with
    the given fill_value, which can be checked for in tests.
    """
    original = distributed.singleton
    distributed.singleton = MockDistributed(fill_value=fill_value)  # type: ignore
    try:
        yield distributed.singleton
    finally:
        distributed.singleton = original
