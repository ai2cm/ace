import logging
import os

import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import using_gpu
from fme.core.distributed.base import DistributedBackend
from fme.core.distributed.model_torch_distributed import comm
from fme.core.distributed.non_distributed import DummyWrapper

try:
    from physicsnemo import distributed as pnd
except ImportError:
    pnd = None


logger = logging.getLogger(__name__)


class ModelTorchDistributed(DistributedBackend):
    """A Spatial distributed backend implementation."""

    def __init__(self):
        h_parallel_size = int(os.environ.get("H_PARALLEL_SIZE", 1))
        w_parallel_size = int(os.environ.get("W_PARALLEL_SIZE", 1))
        logger.debug(f" Spatial parallelism dimension in h {h_parallel_size}")
        logger.debug(f" Spatial parallelism dimension in w {w_parallel_size}")
        if (h_parallel_size > 1) or (w_parallel_size > 1):
            logger.debug(" Spatial parallelism enable.")
            model_parallel_sizes = [h_parallel_size, w_parallel_size, 1, 1]
            model_parallel_names = ["h", "w", "fin", "fout"]
            comm.init(
                model_parallel_sizes=model_parallel_sizes,
                model_parallel_names=model_parallel_names,
                verbose=False,
            )
            self.world_size = comm.get_world_size()
            self._rank = comm.get_world_rank()
            self._device_id = comm.get_local_rank()
            torch.cuda.set_device(self._device_id)
        else:
            raise ValueError(
                "Spatially distributed backend: "
                "h_parallel_size and w_parallel_size are both <=1."
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if Torch distributed is available and "
        "if spatial parallelism is enabled.
        """
        return torch.distributed.is_available() and _spatial_parallelism_enabled()

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def total_ranks(self) -> int:
        """Total number of processes."""
        return self.world_size

    def get_local_rank(self) -> int:
        return self._device_id

    def get_local_slices(self, tensor_shape):
        return _get_local_slices(tensor_shape)

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size // comm.get_size("data")

    def reduce_mean(self, tensor: torch.Tensor, group=None) -> torch.Tensor | None:
        torch.distributed.all_reduce(
            tensor, group=comm.get_group(group), op=torch.distributed.ReduceOp.AVG
        )
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return tensor

    def comm_get_size(self, key: str):
        return comm.get_size(key)

    def comm_get_group(self, key: str):
        return comm.get_group(key)

    def comm_get_rank(self, key: str):
        return comm.get_rank(key)

    def gather(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        raise NotImplementedError()

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        raise NotImplementedError()

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


def _spatial_parallelism_enabled() -> bool:
    h_parallel_size = int(os.environ.get("H_PARALLEL_SIZE", 1))
    w_parallel_size = int(os.environ.get("W_PARALLEL_SIZE", 1))
    if (h_parallel_size > 1) or (w_parallel_size > 1):
        return True
    else:
        return False


def _get_local_slices(tensor_shape):
    tensor_offset = (0, 0)
    local_shape_h = tensor_shape[0]
    local_offset_h = tensor_offset[0]
    local_shape_w = tensor_shape[1]
    local_offset_w = tensor_offset[1]
    if comm.get_size("h") > 1:
        shapes_h = pnd.utils.compute_split_shapes(tensor_shape[0], comm.get_size("h"))
        local_shape_h = shapes_h[comm.get_rank("h")]
        local_offset_h = tensor_offset[0] + sum(shapes_h[: comm.get_rank("h")])
    if comm.get_size("w") > 1:
        shapes_w = pnd.utils.compute_split_shapes(tensor_shape[1], comm.get_size("w"))
        local_shape_w = shapes_w[comm.get_rank("w")]
        local_offset_w = tensor_offset[1] + sum(shapes_w[: comm.get_rank("w")])
    return (
        slice(local_offset_h, local_offset_h + local_shape_h),
        slice(local_offset_w, local_offset_w + local_shape_w),
    )
