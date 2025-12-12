import logging
import os
from collections.abc import Callable
from datetime import timedelta

import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.functional import pad
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import get_device, using_gpu, using_srun

from .base import DistributedBackend
from .non_distributed import DummyWrapper

logger = logging.getLogger(__name__)


class TorchDistributed(DistributedBackend):
    """A non-distributed backend implementation."""

    def __init__(self):
        if "RANK" in os.environ and not using_srun():  # we were executed with torchrun
            if not torch.distributed.is_initialized():
                if using_gpu():
                    torch.distributed.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        timeout=timedelta(minutes=20),
                    )
                else:
                    torch.distributed.init_process_group(
                        backend="gloo", init_method="env://"
                    )
            self.world_size = torch.distributed.get_world_size()
            local_rank = int(os.environ["LOCAL_RANK"])
            self._rank = torch.distributed.get_rank()
            if using_gpu():
                self._device_id = local_rank
                torch.cuda.set_device(self._device_id)
        elif using_srun():  # executing with srun
            shared_dist_file = os.environ["SRUN_DIST_FILE_PATH"]
            self._rank = int(os.environ["SLURM_PROCID"])
            self.world_size = int(os.environ["SLURM_NTASKS"])
            backend = "nccl" if using_gpu() else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                init_method=f"file://{shared_dist_file}",
                rank=self.rank,
                world_size=self.world_size,
            )
            if using_gpu():
                # this assumes one GPU per process in the SLURM setting
                # --gpus-per-task=1 --gpu-bind=closest
                self._device_id = 0
                torch.cuda.set_device(self._device_id)
        else:
            raise ValueError(
                "Distributed backend initialized without torchrun or srun."
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if torch distributed is available."""
        return torch.distributed.is_available() and "RANK" in os.environ or using_srun()

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def total_ranks(self) -> int:
        """Total number of processes."""
        return self.world_size

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size // self.total_ranks

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor)
        return tensor / self.total_ranks

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return tensor

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        gather_list: list[torch.Tensor] | None = None
        if self.rank == 0:
            gather_list = [tensor] + [
                torch.empty_like(tensor) for _ in range(self.world_size - 1)
            ]
        torch.distributed.gather(tensor, gather_list)
        return gather_list

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        return _gather_irregular(
            tensor,
            self.reduce_max,
            self.gather,
        )

    @property
    def _device_ids(self) -> list[int] | None:
        if using_gpu():
            return [self._device_id]
        else:
            return None

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if any(p.requires_grad for p in module.parameters()):
            if using_gpu():
                output_device = [self._device_id]
            else:
                output_device = None
            return DistributedDataParallel(
                SyncBatchNorm.convert_sync_batchnorm(module),
                device_ids=self._device_ids,
                output_device=output_device,
            )
        return DummyWrapper(module)

    def barrier(self):
        logger.debug(f"Barrier on rank {self.rank}")
        torch.distributed.barrier(device_ids=self._device_ids)

    def shutdown(self):
        self.barrier()
        logger.debug(f"Shutting down rank {self.rank}")
        torch.distributed.destroy_process_group()


def _gather_irregular(
    tensor: torch.Tensor,
    reduce_max: Callable[[torch.Tensor], torch.tensor],
    gather: Callable[[torch.Tensor], list[torch.Tensor] | None],
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
        fill_value: The value to fill each tensor with.

    Returns:
        A list of tensors, where the i-th element is the tensor from the i-th process.
    """
    output_tensor_size = []
    tensor_size = list(tensor.size())
    for dim_len in tensor_size:
        dimension_length = torch.tensor(dim_len, dtype=torch.int32, device=get_device())
        reduce_max(dimension_length)
        output_tensor_size.append(int(dimension_length.item()))
    dimension_difference = [
        output - input for output, input in zip(output_tensor_size, tensor_size)
    ]
    regular_tensor = _pad_tensor_at_end(tensor, dimension_difference, fill_value)
    gathered_regular_tensors = gather(regular_tensor)
    gathered_dimension_differences = gather(
        torch.tensor(dimension_difference, device=get_device())
    )
    if gathered_regular_tensors is None or gathered_dimension_differences is None:
        return None
    else:
        return [
            _unpad_tensor_at_end(regular_tensor, dimension_difference)
            for regular_tensor, dimension_difference in zip(
                gathered_regular_tensors, gathered_dimension_differences
            )
        ]


def _pad_tensor_at_end(
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


def _unpad_tensor_at_end(
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
