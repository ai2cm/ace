from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
import torch.nn as nn

from .stop_agreement import StopAgreement

T = TypeVar("T")


class DistributedBackend(ABC):
    """
    Interface that TorchDistributed / NonDistributed must implement.
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """Global rank of this process."""
        ...

    @property
    @abstractmethod
    def data_parallel_rank(self) -> int: ...

    @property
    @abstractmethod
    def total_ranks(self) -> int:
        """Total number of processes."""
        ...

    @property
    @abstractmethod
    def total_data_parallel_ranks(self) -> int:
        """
        Total number of rank splits along the data parallel dimension.

        For example, 8 ranks using 2 ranks of model parallelism would have
        only 4 ranks of data paralellism.
        """

    @abstractmethod
    def local_batch_size(self, batch_size: int) -> int: ...

    @abstractmethod
    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None = None): ...

    @abstractmethod
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def stop_agreement(self) -> StopAgreement:
        """The group ranks use to agree on leaving a loop together.

        It must be **world-wide**, which is why it is not `reduce_max`: that is
        data-parallel-only under a spatially-parallel backend, while the teardown
        is world-wide, so reusing it would let a stop originating on a spatial
        co-rank never reach its data-parallel peers.

        Created with the backend and **not** destroyed with it. ``shutdown()`` is
        a no-op on a gloo group and its destructor is unbounded, so the group is
        held for the life of the process and disposed of by the process's exit.
        """
        ...

    @abstractmethod
    def abort(self) -> None:
        """Abort this rank's communicators, for the teardown watchdog.

        Called only when the collective teardown has overrun its deadline, in
        place of dropping the communicators by a hard exit. It targets the
        **default** group and never the agreement group: gloo's ``abort()`` is an
        empty default, so this is NCCL-only in effect, and aborting the agreement
        group would achieve nothing while the design needs it kept anyway.
        """
        ...

    @abstractmethod
    def gather(
        self, tensor: torch.Tensor, gather_list: list[torch.Tensor] | None
    ) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process.

        Note: tensor shape is assumed to be equal across all processes; data will
            reshaped/filled/dropped to coerce non-root tensors to the shape
            of the root tensor if not. To avoid this behavior, use
            "gather_irregular" instead.

        Args:
            tensor: The tensor to gather.
            gather_list: A list of tensor buffers to gather into,
                one for each rank.

        Returns:
            A list of tensors, where the i-th element is the tensor
                from the i-th process.
        """
        ...

    @abstractmethod
    def gather_object(self, obj: T) -> list[T] | None: ...

    @abstractmethod
    def scatter_object(self, obj: T) -> T: ...

    @abstractmethod
    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
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
        ...

    @abstractmethod
    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a module in for distributed training, if required.

        The wrapped module must follow the module structure of DistributedDataParallel,
        with the passed module's state contained under "module".
        """
        ...

    @abstractmethod
    def barrier(self): ...

    @abstractmethod
    def shutdown(self): ...

    @abstractmethod
    def get_sht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        """Create a forward SHT (possibly distributed)."""
        ...

    @abstractmethod
    def get_isht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        """Create an inverse SHT (possibly distributed)."""
        ...

    @abstractmethod
    def get_disco_conv_s2(self, *args, **kwargs) -> nn.Module:
        """Create a disco conv S2 instance (possibly distributed)."""
        ...

    @abstractmethod
    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce sum across spatial (h, w) ranks. Identity for non-spatial."""
        ...

    @abstractmethod
    def broadcast_spatial(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a tensor from the spatial-group root to spatial co-ranks.

        Used to make a non-spatial quantity (e.g. a per-sample dropout mask)
        identical across tiles of the same sample, while leaving data-parallel
        ranks free to hold distinct values. Identity when there is no spatial
        parallelism.
        """
        ...

    @abstractmethod
    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Compute a weighted mean, correctly handling spatial parallelism."""
        ...

    @abstractmethod
    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the zonal mean (mean over longitude dimension)."""
        ...
