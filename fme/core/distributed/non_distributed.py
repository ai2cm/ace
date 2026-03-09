from typing import TypeVar

import torch
import torch.nn as nn
import torch_harmonics as th

from fme.core import metrics

from .base import DistributedBackend

T = TypeVar("T")


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


class NonDistributed(DistributedBackend):
    """A non-distributed backend implementation."""

    @property
    def rank(self) -> int:
        """Global rank of this process."""
        return 0

    @property
    def data_parallel_rank(self) -> int:
        return self.rank  # no model parallelism

    @property
    def total_ranks(self) -> int:
        """Total number of processes."""
        return 1

    @property
    def total_data_parallel_ranks(self) -> int:
        return self.total_ranks  # no model parallelism

    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None = None):
        return tuple(slice(None, None) for _ in tensor_shape)

    def local_batch_size(self, batch_size: int) -> int:
        return batch_size

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        # reduction is across processes, so no-op here
        return tensor

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def gather(
        self, tensor: torch.Tensor, gather_list: list[torch.Tensor] | None = None
    ) -> list[torch.Tensor] | None:
        if gather_list is not None:
            if len(gather_list) != 1:
                raise ValueError(
                    f"expected 1 element in gather_list, got {len(gather_list)}"
                )
            gather_list[0][:] = tensor
            return gather_list
        return [tensor]

    def gather_object(self, obj: T) -> list[T] | None:
        return [obj]

    def scatter_object(self, obj: T) -> T:
        return obj

    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        return [tensor]

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return DummyWrapper(module)

    def barrier(self):
        return

    def get_sht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        return th.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()

    def get_isht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        return th.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()

    def get_disco_conv_s2(self, *args, **kwargs) -> nn.Module:
        return th.DiscreteContinuousConvS2(*args, **kwargs).float()

    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        return metrics.weighted_mean(data, weights, dim=dim, keepdim=keepdim)

    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        return data.nanmean(dim=-1)

    def gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
    ) -> torch.Tensor:
        return metrics.gradient_magnitude_percent_diff(
            truth, predicted, weights=weights, dim=dim
        )

    def shutdown(self):
        return
