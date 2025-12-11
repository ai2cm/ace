from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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
    def total_ranks(self) -> int:
        """Total number of processes."""
        ...

    @abstractmethod
    def local_batch_size(self, batch_size: int) -> int: ...

    @abstractmethod
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
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
        ...

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
