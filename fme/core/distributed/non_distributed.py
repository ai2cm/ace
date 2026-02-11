import torch

from .base import DistributedBackend


class DummyWrapper(torch.nn.Module):
    """
    Wrapper class for a single pytorch module, which does nothing.

    Exists so we have an identical module structure to the case where we use
    a DistributedDataParallel wrapper.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class NonDistributed(DistributedBackend):
    """A non-distributed backend implementation."""

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return 0

    @property
    def total_ranks(self) -> int:
        """Total number of processes."""
        return 1

    def get_local_slices(self, tensor_shape):
        return tuple(slice(None, None) for _ in tensor_shape)

    def get_local_rank(self):
        return 0

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        # reduction is across processes, so no-op here
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        return [tensor]

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        return [tensor]

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return DummyWrapper(module)

    def barrier(self):
        return

    def shutdown(self):
        return

    def comm_get_size(self, key: str):
        return 1

    def comm_get_group(self, key: str):
        return 1

    def comm_get_rank(self, key: str):
        return 0
