import os
from typing import List, Optional

import torch.distributed

from fme.core.device import using_gpu

singleton: Optional["Distributed"] = None


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

    Attributes:
        world_size: The number of processes in the distributed training job.
        rank: The global rank of the current process.
        local_rank: The node-local rank of the current process.
    """

    @classmethod
    def get_instance(cls) -> "Distributed":
        """
        Get the singleton instance of the Distributed class.
        """
        global singleton
        if singleton is None:
            singleton = cls()
        return singleton

    def __init__(self):
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            self._distributed = self._init_distributed()
        else:
            self._distributed = False

    def _init_distributed(self):
        if "RANK" in os.environ:  # we were executed with torchrun
            if using_gpu():
                torch.distributed.init_process_group(
                    backend="nccl", init_method="env://"
                )
            else:
                torch.distributed.init_process_group(
                    backend="gloo", init_method="env://"
                )
            self.world_size = torch.distributed.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = torch.distributed.get_rank()
            if using_gpu():
                torch.cuda.set_device(self.local_rank)
            distributed = True
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            distributed = False
        return distributed

    def local_batch_size(self, batch_size: int) -> int:
        """
        Get the local batch size for the current process.
        """
        return batch_size // self.world_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a mean across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        if self._distributed:
            torch.distributed.all_reduce(tensor)
        return tensor / self.world_size

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a sum across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        if self._distributed:
            torch.distributed.all_reduce(tensor)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a min across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        if self._distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a max across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        if self._distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return tensor

    def gather(self, tensor: torch.Tensor) -> Optional[List[torch.Tensor]]:
        """
        Gather a tensor from all processes to the root process.

        Modifies the input tensor in-place as a side effect.

        Args:
            tensor: The tensor to gather.

        Returns:
            A list of tensors, where the i-th element is the tensor
                from the i-th process.
        """
        if self.rank == 0:
            gather_list: Optional[List[torch.Tensor]] = [
                torch.empty_like(tensor) for _ in range(self.world_size)
            ]
        else:
            gather_list = None
        if self._distributed:
            torch.distributed.gather(tensor, gather_list)
        return gather_list

    def is_root(self) -> bool:
        """
        Returns True if this process is the root process.
        """
        return self.rank == 0

    def is_distributed(self) -> bool:
        """
        Returns True if this process is running in a distributed context
        with more than 1 worker.
        """
        return self._distributed and self.world_size > 1
