import logging
import os
from collections.abc import Callable

import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.functional import pad
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import get_device, using_gpu, using_srun
from fme.ace.utils import comm

logger = logging.getLogger(__name__)


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

    Parameters:
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
        self._seed = 0

    def _init_distributed(self):
        #NOTE: I am commenting this out for now to make testing easier. 
        #We can review this block of code once spatial parallelism 
        #is functioning correctly in a full test.
        #if "RANK" in os.environ and not using_srun():  # we were executed with torchrun
        #    if using_gpu():
        #        torch.distributed.init_process_group(
        #            backend="nccl", init_method="env://"
        #        )
        #    else:
        #        torch.distributed.init_process_group(
        #            backend="gloo", init_method="env://"
        #        )
        #    self.world_size = torch.distributed.get_world_size()
        #    self.local_rank = int(os.environ["LOCAL_RANK"])
        #    self.rank = torch.distributed.get_rank()
        #    if using_gpu():
        #        self._device_id = self.local_rank
        #        torch.cuda.set_device(self._device_id)
        #    distributed = True
        #elif using_srun():  # executing with srun
        #    shared_dist_file = os.environ["SRUN_DIST_FILE_PATH"]
        #    self.rank = int(os.environ["SLURM_PROCID"])
        #    self.world_size = int(os.environ["SLURM_NTASKS"])
        #    self.local_rank = int(os.environ["SLURM_LOCALID"])
        #    backend = "nccl" if using_gpu() else "gloo"
        #    torch.distributed.init_process_group(
        #        backend=backend,
        #        init_method=f"file://{shared_dist_file}",
        #        rank=self.rank,
        #        world_size=self.world_size,
        #    )
        #    if using_gpu():
        #        # this assumes one GPU per process in the SLURM setting
        #        # --gpus-per-task=1 --gpu-bind=closest
        #        self._device_id = 0
        #        torch.cuda.set_device(self._device_id)
        #    distributed = True
        #else:
        #    self.world_size = 1
        #    self.rank = 0
        #    self.local_rank = 0
        #    distributed = False
        #TODO: Pass dist inputs instead of hard-coding them.
        fin_parallel_size=1#args.fin_parallel_size
        fout_parallel_size=1#args.fout_parallel_size
        h_parallel_size=1#args.h_parallel_size
        w_parallel_size=1#args.w_parallel_size
        params={}
        params["fin_parallel_size"] = fin_parallel_size
        params["fout_parallel_size"] = fout_parallel_size
        params["h_parallel_size"] = h_parallel_size
        params["w_parallel_size"] = w_parallel_size

        params["model_parallel_sizes"] = [h_parallel_size, w_parallel_size, fin_parallel_size, fout_parallel_size]
        params["model_parallel_names"] = ["h", "w", "fin", "fout"]

        comm.init(model_parallel_sizes=params["model_parallel_sizes"], model_parallel_names=params["model_parallel_names"], verbose=False)

        self.world_size = comm.get_world_size()
        self.rank = comm.get_world_rank()
        self.local_rank = comm.get_local_rank()
        distributed = True
        torch.cuda.set_device(comm.get_local_rank())
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return distributed

    def get_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool,
        drop_last: bool = False,
    ) -> torch.utils.data.Sampler:
        return torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=self._seed,
            drop_last=drop_last,
        )

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
        gather_list: list[torch.Tensor] | None = None
        if self.rank == 0:
            gather_list = [tensor] + [
                torch.empty_like(tensor) for _ in range(self.world_size - 1)
            ]
        if self._distributed:
            torch.distributed.gather(tensor, gather_list)
        return gather_list

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
        return gather_irregular(
            tensor,
            self.reduce_max,
            self.gather,
            is_distributed=self.is_distributed(),
        )

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

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a model with DistributedDataParallel if running in a distributed context.
        """
        if self.is_distributed() and any(p.requires_grad for p in module.parameters()):
            if using_gpu():
                device_ids = [self._device_id]
                output_device = [self._device_id]
            else:
                device_ids = None
                output_device = None
            return DistributedDataParallel(
                SyncBatchNorm.convert_sync_batchnorm(module),
                device_ids=device_ids,
                output_device=output_device,
            )
        else:
            return DummyWrapper(module)

    def barrier(self):
        """
        Wait for all processes to reach this point.
        """
        if self._distributed:
            logger.debug(f"Barrier on rank {self.rank}")
            torch.distributed.barrier()

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
        self.barrier()
        if self._distributed:
            logger.debug(f"Shutting down rank {self.rank}")
            comm.cleanup()
        #    torch.distributed.destroy_process_group()


singleton: Distributed | None = None


def gather_irregular(
    tensor: torch.Tensor,
    reduce_max: Callable[[torch.Tensor], torch.tensor],
    gather: Callable[[torch.Tensor], list[torch.Tensor] | None],
    is_distributed: bool = False,
    fill_value: float | int = 0.0,
) -> list[torch.Tensor] | None:
    """
    Gather a tensor from all processes to the root process. The rank tensors
    may have different dimension lengths, but must have the same number of dimensions.

    To accomplish this, the tensor is temporarily padded with `fill_value` where
    its dimension length is smaller than the maximum dimension length for the purpose of
    communication, and the padding is removed prior to returning the gathered tensors.

    Args:
        tensor: The tensor to gather.
        reduce_max: The reduction function to use for each dimension length.
        gather: The gather function to use.
        is_distributed: Whether the current process is distributed.
        fill_value: The value to fill each tensor with.

    Returns:
        A list of tensors, where the i-th element is the tensor from the i-th process.
    """
    output_tensor_size = []
    tensor_size = list(tensor.size())
    for dim_len in tensor_size:
        if is_distributed:
            dimension_length = torch.tensor(
                dim_len, dtype=torch.int32, device=get_device()
            )
            reduce_max(dimension_length)
            output_tensor_size.append(int(dimension_length.item()))
        else:
            output_tensor_size.append(int(dim_len))
    dimension_difference = [
        output - input for output, input in zip(output_tensor_size, tensor_size)
    ]
    regular_tensor = pad_tensor_at_end(tensor, dimension_difference, fill_value)
    gathered_regular_tensors = gather(regular_tensor)
    gathered_dimension_differences = gather(
        torch.tensor(dimension_difference, device=get_device())
    )
    if gathered_regular_tensors is None or gathered_dimension_differences is None:
        return None
    else:
        return [
            unpad_tensor_at_end(regular_tensor, dimension_difference)
            for regular_tensor, dimension_difference in zip(
                gathered_regular_tensors, gathered_dimension_differences
            )
        ]


def pad_tensor_at_end(
    tensor: torch.Tensor,
    dimension_difference: list[int],
    fill_value: float | int = 0.0,
):
    """Pad tensor by specified amount at end of each dimension.
    Note that `pad` format is in reverse dimension order.

    Args:
        tensor: The tensor to pad
        dimension_difference: The amount to pad each dimension
        fill_value: The value to fill the padding with

    Returns:
        The padded tensor
    """
    assert len(dimension_difference) == len(tensor.size()), "Dimension mismatch"
    pad_dimensions = tuple(
        [
            val
            for pair in zip(
                [0 for _ in tensor.size()],
                [diff for diff in dimension_difference[::-1]],
            )
            for val in pair
        ]
    )
    padded_tensor = pad(tensor, pad_dimensions, mode="constant", value=fill_value).to(
        get_device()
    )
    return padded_tensor


def unpad_tensor_at_end(
    tensor: torch.Tensor, dimension_difference: torch.Tensor
) -> torch.Tensor:
    """Remove padding from tensor.

    Args:
        tensor: The tensor to remove padding from
        dimension_difference: The amount of padding to remove from each dimension

    Returns:
        The tensor with padding removed
    """
    assert len(dimension_difference) == len(tensor.size()), "Dimension mismatch"
    slice_dimensions = tuple(
        [
            slice(0, tensor.size()[i] - dimension_difference[i])
            for i in range(len(tensor.size()))
        ]
    )
    return tensor[slice_dimensions]
