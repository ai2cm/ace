from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
import torch.nn as nn

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

    def gather_spatial_tensor(
        self, tensor: torch.Tensor, img_shape: tuple[int, int]
    ) -> torch.Tensor:
        """Reassemble a spatially-sharded tensor on every rank via all-reduce.

        Args:
            tensor: Local spatial shard.
            img_shape: Global ``(H, W)`` spatial dimensions.
        """
        if img_shape == tensor.shape[-2:]:
            return tensor
        global_shape = (*tensor.shape[:-2], *img_shape)
        slices = self.get_local_slices(img_shape)
        buf = torch.zeros(global_shape, dtype=tensor.dtype, device=tensor.device)
        buf[(..., *slices)] = tensor
        return self.spatial_reduce_sum(buf)

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

    @abstractmethod
    def gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
    ) -> torch.Tensor:
        """Compute percent difference of weighted mean gradient magnitude."""
        ...
