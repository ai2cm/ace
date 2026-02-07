import logging
import os

import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import using_gpu
from fme.core.distributed import comm

from .base import DistributedBackend
from .non_distributed import DummyWrapper
from .torch_distributed import _gather_irregular

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

    def get_local_slices(self, crop_shape):
        crop_offset = (0, 0)
        local_shape_h = crop_shape[0]
        local_offset_h = crop_offset[0]
        local_shape_w = crop_shape[1]
        local_offset_w = crop_offset[1]
        if comm.get_size("h") > 1:
            shapes_h = pnd.utils.compute_split_shapes(crop_shape[0], comm.get_size("h"))
            local_shape_h = shapes_h[comm.get_rank("h")]
            local_offset_h = crop_offset[0] + sum(shapes_h[: comm.get_rank("h")])
        if comm.get_size("w") > 1:
            shapes_w = pnd.utils.compute_split_shapes(crop_shape[1], comm.get_size("w"))
            local_shape_w = shapes_w[comm.get_rank("w")]
            local_offset_w = crop_offset[1] + sum(shapes_w[: comm.get_rank("w")])
        return (
            slice(local_offset_h, local_offset_h + local_shape_h),
            slice(local_offset_w, local_offset_w + local_shape_w),
        )

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size // comm.get_size("data")

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


def _spatial_parallelism_enabled() -> bool:
    h_parallel_size = int(os.environ.get("H_PARALLEL_SIZE", 1))
    w_parallel_size = int(os.environ.get("W_PARALLEL_SIZE", 1))
    if (h_parallel_size > 1) or (w_parallel_size > 1):
        return True
    else:
        return False
