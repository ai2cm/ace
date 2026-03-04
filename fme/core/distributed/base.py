from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.distributed import ProcessGroup


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
    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None): ...

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
    def gather_object(self, obj: object) -> list[object] | None: ...

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

    @property
    def h_size(self) -> int:
        return 1

    @property
    def w_size(self) -> int:
        return 1

    @property
    def h_rank(self) -> int:
        return 0

    @property
    def w_rank(self) -> int:
        return 0

    @property
    def h_group(self) -> ProcessGroup | None:
        return None

    @property
    def w_group(self) -> ProcessGroup | None:
        return None

    @property
    def is_spatial_parallel(self) -> bool:
        return False

    def get_spatial_slices(self, h: int, w: int) -> tuple[slice, slice]:
        """Return ``(h_slice, w_slice)`` for the local spatial chunk.

        Parameters
        ----------
        h, w : int
            Global spatial dimensions.

        Returns:
        -------
        tuple[slice, slice]
            Slices into the global ``[..., h, w]`` tensor for this rank.
            Non-spatial backends return ``(slice(None), slice(None))``.
        """
        return slice(None), slice(None)

    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce sum across spatial (h, w) ranks. Identity for non-spatial."""
        return tensor
