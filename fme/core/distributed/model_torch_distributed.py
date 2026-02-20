"""
Distributed backend using PhysicsNeMo's DistributedManager for
spatial (h, w) model parallelism.

Provides the same DistributedBackend interface as TorchDistributed
while splitting ranks across two spatial dimensions (h, w).
Any remaining ranks are used for data parallelism.

Process group hierarchy::

    world
    ├── spatial  (h * w ranks: model parallelism)
    │   ├── h
    │   └── w
    └── data     (residual ranks: data parallelism)
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from fme.core.device import using_gpu, using_srun

from .base import DistributedBackend
from .external.pnd_manager import DistributedManager
from .non_distributed import DummyWrapper
from .torch_distributed import _gather_irregular

logger = logging.getLogger(__name__)


class ModelTorchDistributed(DistributedBackend):
    """Distributed backend with spatial model parallelism.

    Splits ranks across two spatial dimensions (h, w).
    Residual ranks become data-parallel.

    Parameters
    ----------
    h_size : int
        Number of ranks along the *h* spatial dimension.
    w_size : int
        Number of ranks along the *w* spatial dimension.
    verbose : bool
        Print group layout at init time.
    """

    def __init__(
        self,
        h_size: int = 1,
        w_size: int = 1,
        verbose: bool = False,
    ):
        # Initialise PhysicsNeMo DistributedManager.
        DistributedManager.initialize()
        self._dm = DistributedManager()

        # Build a 3-D (data, h, w) DeviceMesh; data=-1 auto-sizes the
        # data-parallel dimension to absorb all remaining ranks.
        spatial_size = h_size * w_size
        if self._dm.world_size % spatial_size != 0:
            raise ValueError(
                f"world_size must be divisible by h_size * w_size, "
                f"got world_size={self._dm.world_size} and h*w={spatial_size}"
            )
        mesh = self._dm.initialize_mesh(
            mesh_shape=(-1, h_size, w_size),
            mesh_dim_names=("data", "h", "w"),
        )
        if verbose and self._dm.rank == 0:
            logger.info(
                "DeviceMesh initialized: data=%d h=%d w=%d",
                self._dm.world_size // spatial_size,
                h_size,
                w_size,
            )

        # Cache frequently-used values.
        self._rank = self._dm.rank
        self._local_rank = self._dm.local_rank
        self._world_size = self._dm.world_size

        # Derive per-axis process groups and sizes from the mesh.
        self._data_group = self._dm.get_mesh_group(mesh["data"])
        self._h_group = self._dm.get_mesh_group(mesh["h"])
        self._w_group = self._dm.get_mesh_group(mesh["w"])

        self._data_size = torch.distributed.get_world_size(group=self._data_group)
        self._data_rank = torch.distributed.get_rank(group=self._data_group)
        self._h_size = torch.distributed.get_world_size(group=self._h_group)
        self._h_rank = torch.distributed.get_rank(group=self._h_group)
        self._w_size = torch.distributed.get_world_size(group=self._w_group)
        self._w_rank = torch.distributed.get_rank(group=self._w_group)

        # Keep a reference to the spatial sub-mesh for the combined group.
        # The (h, w) 2-D sub-mesh is obtained by slicing the global mesh
        # over the "data" dimension for this rank's data slice.
        self._mesh = mesh

        if using_gpu():
            self._device_id = self._local_rank
            torch.cuda.set_device(self._device_id)

        logger.info(
            "ModelTorchDistributed initialized: "
            "rank=%d/%d, data_rank=%d/%d, "
            "spatial h=%d w=%d",
            self._rank,
            self._world_size,
            self._data_rank,
            self._data_size,
            self._h_size,
            self._w_size,
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if the environment supports this backend."""
        return torch.distributed.is_available() and (
            "RANK" in os.environ or using_srun()
        )

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return self._rank

    @property
    def data_parallel_rank(self) -> int:
        """Rank within the data-parallel group."""
        return self._data_rank

    @property
    def total_ranks(self) -> int:
        """Total number of processes (world size)."""
        return self._world_size

    @property
    def total_data_parallel_ranks(self) -> int:
        """Number of data-parallel ranks."""
        return self._data_size

    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None):
        """Return index slices for the data-parallel chunk."""
        return_list = [slice(None, None) for _ in tensor_shape]
        if data_parallel_dim is not None:
            n_dp = self.total_data_parallel_ranks
            if tensor_shape[data_parallel_dim] % n_dp != 0:
                raise ValueError(
                    f"expected global data parallel dim to be "
                    f"divisible by data parallel ranks, got "
                    f"global shape {tensor_shape} with "
                    f"{n_dp} data parallel ranks"
                )
            per_rank = tensor_shape[data_parallel_dim] // n_dp
            return_list[data_parallel_dim] = slice(
                self._data_rank * per_rank, (self._data_rank + 1) * per_rank
            )
        return tuple(return_list)

    def local_batch_size(self, batch_size: int) -> int:
        """Divide global batch among data-parallel ranks."""
        return batch_size // self.total_data_parallel_ranks

    # NOTE: reductions are performed over the data-parallel group only, since
    # the spatial groups are meant for model parallelism, and for now we assume
    # that any tensors being reduced are replicated across the spatial groups.
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, group=self._data_group)
        return tensor / self.total_data_parallel_ranks

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(tensor, group=self._data_group)
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=self._data_group
        )
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None:
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=self._data_group
        )
        return tensor

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor] | None:
        # NOTE: gather is performed over the data-parallel group only, like reductions
        # NOTE: dst must be a *global* rank that belongs to the data group.
        # data_rank=0 corresponds to the first entry in the group's rank list.
        root_global_rank = torch.distributed.get_process_group_ranks(self._data_group)[
            0
        ]
        if gather_list is None and self._data_rank == 0:
            gather_list = [tensor] + [
                torch.empty_like(tensor) for _ in range(self._data_size - 1)
            ]
        torch.distributed.gather(
            tensor,
            gather_list,
            dst=root_global_rank,
            group=self._data_group,
        )
        return gather_list if self._rank == 0 else None

    def gather_object(self, obj: object) -> list[object] | None:
        """Gather a picklable object over the data-parallel group."""
        root_global_rank = torch.distributed.get_process_group_ranks(self._data_group)[
            0
        ]
        gather_list: list[object] | None = (
            [None for _ in range(self._data_size)] if self._data_rank == 0 else None
        )
        torch.distributed.gather_object(
            obj, gather_list, dst=root_global_rank, group=self._data_group
        )
        return gather_list if self._rank == 0 else None

    # For now, let's just borrow the same gather_irregular implementation
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
        return None

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Wrap with DDP over the **data** process group.

        For now, we assume spatial communication is expected to be handled
        inside the model layers themselves. If we need to change course, we
        can revisit...
        """
        if any(p.requires_grad for p in module.parameters()):
            if using_gpu():
                output_device = [self._device_id]
            else:
                output_device = None
            return DistributedDataParallel(
                SyncBatchNorm.convert_sync_batchnorm(module),
                device_ids=self._device_ids,
                output_device=output_device,
                process_group=self._data_group,
            )
        return DummyWrapper(module)

    def barrier(self):
        """Global barrier across all ranks."""
        logger.debug("Barrier on rank %d", self._rank)
        torch.distributed.barrier(device_ids=self._device_ids)

    def shutdown(self):
        self.barrier()
        logger.debug("Shutting down rank %d", self._rank)
        DistributedManager.cleanup()
