import logging
import os

import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import using_gpu
from fme.core.distributed.base import DistributedBackend
from fme.core.distributed.non_distributed import DummyWrapper

# NOTE: physicsnemo is optional, so carefully guard
try:
    from physicsnemo import distributed as pnd

    # model_torch_distributed_comm relies physicsnemo too
    from fme.core.distributed import model_torch_distributed_comm as comm
except ImportError:
    pnd = None

logger = logging.getLogger(__name__)


class ModelTorchDistributed(DistributedBackend):
    """Distributed backend for spatial (H/W) parallelism via physicsnemo.

    Splits the spatial dimensions (latitude/height and longitude/width)
    across multiple GPUs using NVIDIA PhysicsNeMo's distributed manager.
    Falls back to TorchDistributed when H_PARALLEL_SIZE and W_PARALLEL_SIZE
    are both <= 1 or further to NonDistributed if needed.
    """

    def __init__(self):
        if pnd is None:
            raise ImportError(
                "The physicsnemo package is required for spatial parallelism."
            )
        h_parallel_size = int(os.environ.get("H_PARALLEL_SIZE", 1))
        w_parallel_size = int(os.environ.get("W_PARALLEL_SIZE", 1))
        if (h_parallel_size > 1) or (w_parallel_size > 1):
            # NOTE: last two aren't used (yet)
            model_parallel_sizes = [h_parallel_size, w_parallel_size, 1, 1]
            model_parallel_names = ["h", "w", "fin", "fout"]
            # If comm is already initialized with different sizes, reset it
            if comm.is_distributed("world"):
                if (
                    comm.get_size("h") != h_parallel_size
                    or comm.get_size("w") != w_parallel_size
                ):
                    _reset_comm(comm)

            if not comm.is_distributed("world"):
                comm.init(
                    model_parallel_sizes=model_parallel_sizes,
                    model_parallel_names=model_parallel_names,
                    verbose=False,
                )
            assert comm.get_size("h") == h_parallel_size
            assert comm.get_size("w") == w_parallel_size

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
        """Check if distributed spatial parallelism is enabled."""
        return (
            pnd is not None
            and torch.distributed.is_available()
            and _spatial_parallelism_enabled()
        )

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def data_parallel_rank(self) -> int:
        return comm.get_rank("data")

    @property
    def total_ranks(self) -> int:
        """Total number of processes."""
        return self.world_size

    @property
    def total_data_parallel_ranks(self) -> int:
        return comm.get_size("data")

    def get_local_rank(self) -> int:
        return self._device_id

    def get_local_slices(
        self,
        tensor_shape,
        rank: int | None = None,
        data_parallel_dim: int | None = None,
    ):
        # Under spatial parallelism, spatial slicing is determined by comm groups.
        # Batch (data-parallel) partitioning is handled separately via
        # local_batch_size and data_parallel_rank, not through get_local_slices.
        return _get_local_slices(tensor_shape)

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size // comm.get_size("data")

    def reduce_mean(self, tensor: torch.Tensor, group=None) -> torch.Tensor | None:
        torch.distributed.all_reduce(
            tensor, group=comm.get_group(group), op=torch.distributed.ReduceOp.AVG
        )
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None:
        # Intentionally uses the default (world) process group.
        # Unlike reduce_mean, reduce_sum does not accept a group parameter
        # because current callers always need the global sum.
        torch.distributed.all_reduce(tensor)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None:
        # Uses the default (world) process group; see reduce_sum for rationale.
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None:
        # Uses the default (world) process group; see reduce_sum for rationale.
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return tensor

    def comm_get_size(self, key: str):
        return comm.get_size(key)

    def comm_get_group(self, key: str):
        return comm.get_group(key)

    def comm_get_rank(self, key: str):
        return comm.get_rank(key)

    def gather(
        self, tensor: torch.Tensor, gather_list: list[torch.Tensor] | None = None
    ) -> list[torch.Tensor] | None:
        raise NotImplementedError(
            "gather is not yet implemented for spatial parallelism"
        )

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        raise NotImplementedError(
            "gather_irregular is not yet implemented for spatial parallelism"
        )

    @property
    def _device_ids(self) -> list[int] | None:
        if using_gpu():
            return [self._device_id]
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
        comm.cleanup()
        torch.distributed.destroy_process_group()


def _spatial_parallelism_enabled() -> bool:
    h_parallel_size = int(os.environ.get("H_PARALLEL_SIZE", 1))
    w_parallel_size = int(os.environ.get("W_PARALLEL_SIZE", 1))
    if (h_parallel_size > 1) or (w_parallel_size > 1):
        return True
    return False


def _get_local_slices(tensor_shape):
    """Compute local (H, W) slices for this rank under spatial parallelism.

    Args:
        tensor_shape: A 2-tuple (height, width) of the global spatial shape.
    """
    assert len(tensor_shape) == 2, f"Expected 2D shape (H, W), got {tensor_shape}"

    if pnd is None:
        raise ImportError("physicsnemo is required for spatial parallel local slices")

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


# NOTE: the following is a hack to enable easier testing
def _reset_comm(comm_module):
    """Reset comm state so init() can be called again with new group sizes.

    This accesses private globals in the comm module to perform a partial
    cleanup. Unlike comm.cleanup(), this does NOT destroy the underlying
    torch.distributed process group, but does destroy all physicsnemo-managed
    subgroups to free nccl resources and clears internal registries to avoid conflicts.

    This is a hack meant to facilitate testing different layouts/decomps in the
    same torchrun session. We place it here because we want to keep the comm
    code identical to upstream so that syncing it would be easy. Put differently,
    something like this code likely belongs upstream, not here...

    NOTE: physicsnemo/distributed/manager.py:392: UserWarning:
        Distributed manager is already intialized
        warn("Distributed manager is already intialized")
    """
    if comm_module._DM is not None:
        # Properly destroy all created process groups to free NCCL resources
        # without tearing down the default group.
        if hasattr(comm_module._DM, "_groups"):
            import torch.distributed as dist

            # Destroy child groups first though order shouldn't matter
            # for independent groups; we iterate over a copy of values
            # since we aren't modifying the dict in-place here
            for group in comm_module._DM._groups.values():
                dist.destroy_process_group(group)
            comm_module._DM._groups.clear()

        # Clear PhysicsNeMo internal state registries so subsequent init()
        # calls can re-create groups without "Group already exists" errors.
        if hasattr(comm_module._DM, "_group_ranks"):
            comm_module._DM._group_ranks.clear()
        if hasattr(comm_module._DM, "_group_names"):
            comm_module._DM._group_names.clear()

        # Also clear mesh-related state if present
        if hasattr(comm_module._DM, "_mesh_groups"):
            comm_module._DM._mesh_groups.clear()

        if hasattr(comm_module._DM, "_mesh_dims"):
            comm_module._DM._mesh_dims.clear()
        if hasattr(comm_module._DM, "_global_mesh"):
            comm_module._DM._global_mesh = None

        comm_module._DM = None
    comm_module._COMM_ROOTS = {}
