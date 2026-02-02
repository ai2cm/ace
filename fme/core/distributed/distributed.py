import contextlib
import logging
from collections.abc import Iterator

import torch.distributed

from .base import DistributedBackend
from .non_distributed import NonDistributed
from .torch_distributed import TorchDistributed

logger = logging.getLogger(__name__)


class Distributed:
    """
    A class to represent the distributed concerns for FME training.

    This should generally be initialized first, before any pytorch objects.
    This is important because it sets global variables such as the CUDA
    device for the local rank, which is used when initializing pytorch objects.

    This class uses the
    [Singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern) and should
    be initialized through get_instance. This pattern allows easy access to global
    variables without having to pass them around, and lets us put the initialization
    for this global state in the same place as the routines that use it.
    """

    def __init__(self, force_non_distributed: bool = False):
        if TorchDistributed.is_available() and not force_non_distributed:
            self._distributed: DistributedBackend = TorchDistributed()
        else:
            self._distributed = NonDistributed()
        self._seed = 0
        self._force_non_distributed = force_non_distributed  # for debugging

    @classmethod
    def get_instance(cls) -> "Distributed":
        """
        Get the singleton instance of the Distributed class.
        """
        global singleton
        if singleton is None:
            singleton = cls()
        return singleton

    @classmethod
    @contextlib.contextmanager
    def force_non_distributed(cls) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        try:
            with cls.replace_backend(NonDistributed()) as instance:
                original_force_non_distributed = instance._force_non_distributed
                instance._force_non_distributed = True
                yield instance
        finally:
            instance._force_non_distributed = original_force_non_distributed

    @classmethod
    @contextlib.contextmanager
    def replace_backend(cls, backend: DistributedBackend) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        instance = cls.get_instance()
        original_backend = instance._distributed
        try:
            instance._distributed = backend
            yield instance
        finally:
            instance._distributed = original_backend

    @property
    def rank(self) -> int:
        """
        Get the global rank of this process.
        """
        return self._distributed.rank

    @property
    def world_size(self) -> int:
        """
        Get the total number of processes.
        """
        return self._distributed.total_ranks

    def get_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool,
        drop_last: bool = False,
    ) -> torch.utils.data.Sampler:
        return torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=self._distributed.total_ranks,
            rank=self._distributed.rank,
            seed=self._seed,
            drop_last=drop_last,
        )

    def local_batch_size(self, batch_size: int) -> int:
        """
        Get the local batch size for the current process.
        """
        return self._distributed.local_batch_size(batch_size)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a mean across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_mean(tensor)

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a sum across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_sum(tensor)

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a min across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_min(tensor)

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a max across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_max(tensor)

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process.

        Note: tensor shape is assumed to be equal across all processes; data will
            reshaped/filled/dropped to coerce non-root tensors to the shape
            of the root tensor if not. To avoid this behavior, use
            "gather_irregular" instead.

        Args:
            tensor: The tensor to gather.

        Returns:
            A list of tensors, where the i-th element is the tensor
                from the i-th process.
        """
        return self._distributed.gather(tensor)

    def gather_irregular(
        self,
        tensor: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process. The rank tensors
        may have diferent dimension lengths, but must have the same number of
        dimensions.

        Args:
            tensor: The tensor to gather.

        Returns:
            A list of tensors of consistent shape, where the i-th element is the tensor
                from the i-th process.
        """
        return self._distributed.gather_irregular(tensor)

    def is_root(self) -> bool:
        """
        Returns True if this process is the root process.
        """
        return self._distributed.rank == 0

    def is_distributed(self) -> bool:
        """
        Returns True if this process is running in a distributed context
        with more than 1 worker.
        """
        return self._distributed.total_ranks > 1

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a model with DistributedDataParallel if running in a distributed context.
        """
        return self._distributed.wrap_module(module)

    def barrier(self):
        """
        Wait for all processes to reach this point.
        """
        return self._distributed.barrier()

    def set_seed(self, seed: int):
        """
        Set the random seed.
        """
        self._seed = seed

    def get_seed(self) -> int:
        """
        Get the random seed.
        """
        return self._seed

    def shutdown(self):
        return self._distributed.shutdown()


singleton: Distributed | None = None
