import contextlib
import logging
import os
from collections.abc import Generator, Iterator
from typing import TypeVar

import torch.distributed

from .base import DistributedBackend
from .model_torch_distributed import ModelTorchDistributed
from .non_distributed import NonDistributed
from .torch_distributed import TorchDistributed

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Distributed:
    """
    A class to represent the distributed concerns for FME training.

    This should generally be initialized first, before any pytorch objects.
    This is important because it sets global variables such as the CUDA
    device for the local rank, which is used when initializing pytorch objects.

    This class uses the
    [Singleton pattern](https://en.wikipedia.org/wiki/Singleton_pattern) and should
    be initialized through get_instance. This pattern allows easy access to global
    variables without having to pass them around, and lets us put the initialization
    for this global state in the same place as the routines that use it.
    """

    _entered: bool = False

    def __init__(
        self,
        force_non_distributed: bool = False,
        spatial_parallelism: tuple[int, int] | None = None,
        verbose: bool = False,
    ):
        if force_non_distributed:
            self._distributed: DistributedBackend = NonDistributed()
        elif spatial_parallelism is not None:
            h_size, w_size = spatial_parallelism
            self._distributed = ModelTorchDistributed(
                h_size=h_size,
                w_size=w_size,
                verbose=verbose,
            )
        elif TorchDistributed.is_available():
            self._distributed = TorchDistributed()
        else:
            self._distributed = NonDistributed()
        self._seed = 0
        self._force_non_distributed = force_non_distributed

    @classmethod
    @contextlib.contextmanager
    def context(cls) -> Generator[None, None, None]:
        """
        Context manager for initializing and shutting down the distributed backend.

        This should generally be used at the top level of the training script to
        wrap the entire training process, to ensure proper initialization and
        shutdown of the distributed backend.
        """
        if cls._entered:
            raise RuntimeError("Nested Distributed.context() is not supported.")
        cls._entered = True
        instance = cls.get_instance()
        try:
            yield
        except BaseException:
            # exit immediately to avoid hanging other ranks
            # the OS should clean up resources based on the non-zero exit
            raise  # re-raise the exception to avoid masking it
        else:  # if no exception is raised, let root finish cleanup
            instance.shutdown()
        finally:
            cls._entered = False

    @classmethod
    def get_instance(cls) -> "Distributed":
        """
        Get the singleton instance of the Distributed class.

        The backend can be overridden at the session level via environment
        variables, which is useful for switching between backends

        - ``FME_DISTRIBUTED_BACKEND=torch`` to force :class:`TorchDistributed`
          (the default when ``torch.distributed`` is available)
        - ``FME_DISTRIBUTED_BACKEND=model`` to force
          :class:`ModelTorchDistributed`. Requires ``FME_DISTRIBUTED_H`` and
          ``FME_DISTRIBUTED_W`` to also be set.
        - ``FME_DISTRIBUTED_BACKEND=none`` to force
          :class:`NonDistributed`
        """
        if not cls._entered:
            raise RuntimeError(
                "Distributed.get_instance() called before entering context. "
                "Please use Distributed.context() to wrap the training process, "
                "or ensure that get_instance() is only called within the context."
            )
        global singleton
        if singleton is None:
            backend_env = os.environ.get("FME_DISTRIBUTED_BACKEND")
            if backend_env == "model":
                h = int(os.environ["FME_DISTRIBUTED_H"])
                w = int(os.environ["FME_DISTRIBUTED_W"])
                singleton = cls(spatial_parallelism=(h, w))
            elif backend_env == "none":
                singleton = cls(force_non_distributed=True)
            elif backend_env == "torch":
                singleton = cls()
            elif backend_env is not None:
                raise ValueError(
                    f"Unknown FME_DISTRIBUTED_BACKEND value '{backend_env}'. "
                    "Valid values: 'torch', 'model', 'none'."
                )
            else:
                singleton = cls()
        return singleton

    @classmethod
    @contextlib.contextmanager
    def force_non_distributed(cls) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        try:
            with cls.replace_backend(NonDistributed()) as instance:
                original_force_non_distributed = instance._force_non_distributed
                instance._force_non_distributed = True
                yield instance
        finally:
            instance._force_non_distributed = original_force_non_distributed

    @classmethod
    @contextlib.contextmanager
    def replace_backend(cls, backend: DistributedBackend) -> Iterator["Distributed"]:
        """
        Force the distributed singleton to be in non-distributed mode.
        """
        instance = cls.get_instance()
        original_backend = instance._distributed
        try:
            instance._distributed = backend
            yield instance
        finally:
            instance._distributed = original_backend

    @property
    def rank(self) -> int:
        """
        Get the global rank of this process.
        """
        return self._distributed.rank

    @property
    def data_parallel_rank(self) -> int:
        """
        Get the data parallel rank of this process.

        In the context of distributed learning, this is the "batch"
        rank of this process.
        """
        return self._distributed.data_parallel_rank

    @property
    def total_data_parallel_ranks(self) -> int:
        """
        Get the total number of data parallel ranks.

        This is the number of parallel splits along the "batch" dimension.
        """
        return self._distributed.total_data_parallel_ranks

    @property
    def world_size(self) -> int:
        """
        Get the total number of processes.
        """
        return self._distributed.total_ranks

    def get_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool,
        drop_last: bool = False,
    ) -> torch.utils.data.Sampler:
        return torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=self._distributed.total_data_parallel_ranks,
            rank=self._distributed.data_parallel_rank,
            seed=self._seed,
            drop_last=drop_last,
        )

    def local_batch_size(self, batch_size: int) -> int:
        """
        Get the local batch size for the current process.
        """
        return self._distributed.local_batch_size(batch_size)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a mean across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_mean(tensor)

    def get_local_slices(
        self,
        tensor_shape,
        data_parallel_dim: int | None = None,
    ):
        """
        Gets the slice corresponding to the current rank within a global tensor_shape.

        Args:
            tensor_shape: the shape of the global tensor, which may or may not contain
                a data parallel (batch) dimension. Assume last dims are (H, W).
            data_parallel_dim: the index of the data parallel dimension, if it exists.
                by default, assumes the tensor does not have a data parallel dimension.
        """
        if data_parallel_dim is not None and (
            tensor_shape[data_parallel_dim] % self.total_data_parallel_ranks != 0
        ):
            raise ValueError(
                "expected global data parallel dim to be divisible by data parallel "
                f"ranks, got global shape {tensor_shape} with "
                f"{self.total_data_parallel_ranks} data parallel ranks"
            )
        return self._distributed.get_local_slices(
            tensor_shape, data_parallel_dim=data_parallel_dim
        )

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a sum across all processes.

        Whether the tensor represents a mean is important because to reduce a mean,
        we must divide by the number of processes. To reduce a sum, we must not.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_sum(tensor)

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a min across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_min(tensor)

    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce a tensor representing a max across all processes.

        Modifies the input tensor in-place as a side effect.
        """
        return self._distributed.reduce_max(tensor)

    def gather_object(self, obj: object) -> list[object] | None:
        """
        Gather a picklable object from all processes to the root process.

        Args:
            obj: The object to gather.

        Returns:
            A list of objects, where the i-th element is the object
                from the i-th process.
        """
        return self._distributed.gather_object(obj)

    def gather(self, tensor: T, gather_list: list[T] | None = None) -> list[T] | None:
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
        return self._distributed.gather(tensor, gather_list=gather_list)

    def gather_global(
        self, tensor: torch.Tensor, global_shape, data_parallel_dim: int = 0
    ) -> torch.Tensor | None:
        """
        Gathers tensor data into a single tensor with the data from all ranks.

        Each rank places its local portion (determined by :meth:`get_local_slices`)
        into a zero tensor of ``global_shape``.  Spatial chunks are combined via
        :meth:`spatial_reduce_sum`, then the data-parallel portions are gathered
        to root via :meth:`gather`.

        Args:
            tensor: the tensor data to gather
            global_shape: the shape of the tensor containing data from all ranks
            data_parallel_dim: the dimension in global_shape corresponding to the
                data parallel (or "batch") dimension
        """
        if global_shape[data_parallel_dim] % self.total_data_parallel_ranks != 0:
            raise ValueError(
                "expected global data parallel dim to be divisible by data parallel "
                f"ranks, got global_shape {global_shape} with "
                f"{self.total_data_parallel_ranks} data parallel ranks"
            )
        local_slices = self.get_local_slices(
            global_shape, data_parallel_dim=data_parallel_dim
        )
        # Place local data into global-shaped zeros, then combine spatial chunks.
        global_tensor = torch.zeros(
            *global_shape, dtype=tensor.dtype, device=tensor.device
        )
        global_tensor[local_slices] = tensor
        global_tensor = self.spatial_reduce_sum(global_tensor)
        # Each dp rank now has the full spatial extent for its batch portion.
        # Extract this rank's dp slice and gather across dp ranks to root.
        dp_slices = self.get_local_slices(
            global_shape, data_parallel_dim=data_parallel_dim
        )
        # Build dp-only slices (spatial dims are already reconstructed).
        dp_only_list = list(dp_slices)
        for i in range(len(dp_only_list)):
            if i != data_parallel_dim:
                dp_only_list[i] = slice(None)
        dp_only = tuple(dp_only_list)
        dp_local = global_tensor[dp_only]
        gathered_dp_slices = self.gather_object(dp_only)
        if self.is_root():
            if gathered_dp_slices is None:
                raise RuntimeError("gather_object returned None on root process")
            gathered_global = torch.zeros(
                *global_shape, dtype=tensor.dtype, device=tensor.device
            )
            gather_list = []
            for i in range(self.total_data_parallel_ranks):
                gather_list.append(gathered_global[gathered_dp_slices[i]])
        else:
            gather_list = None
            gathered_global = None
        self.gather(dp_local, gather_list=gather_list)
        return gathered_global

    def gather_irregular(
        self,
        tensor: torch.Tensor,
    ) -> list[torch.Tensor] | None:
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
        return self._distributed.gather_irregular(tensor)

    def is_root(self) -> bool:
        """
        Returns True if this process is the root process.
        """
        return self._distributed.rank == 0

    def is_distributed(self) -> bool:
        """
        Returns True if this process is running in a distributed context
        with more than 1 worker.
        """
        return self._distributed.total_ranks > 1

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a model with DistributedDataParallel if running in a distributed context.
        """
        return self._distributed.wrap_module(module)

    def barrier(self):
        """
        Wait for all processes to reach this point.
        """
        return self._distributed.barrier()

    def set_seed(self, seed: int):
        """
        Set the random seed.
        """
        self._seed = seed

    def get_seed(self) -> int:
        """
        Get the random seed.
        """
        return self._seed

    def get_sht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        return self._distributed.get_sht(nlat, nlon, lmax, mmax, grid)

    def get_isht(self, nlat, nlon, lmax=None, mmax=None, grid="legendre-gauss"):
        return self._distributed.get_isht(nlat, nlon, lmax, mmax, grid)

    def get_disco_conv_s2(self, *args, **kwargs):
        return self._distributed.get_disco_conv_s2(*args, **kwargs)

    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._distributed.spatial_reduce_sum(tensor)

    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        return self._distributed.weighted_mean(data, weights, dim=dim, keepdim=keepdim)

    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        return self._distributed.zonal_mean(data)

    def gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
    ) -> torch.Tensor:
        return self._distributed.gradient_magnitude_percent_diff(
            truth, predicted, weights, dim
        )

    def scatter_spatial(
        self, data: dict[str, torch.Tensor], img_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Slice global tensors to the local spatial chunk for this rank."""
        slices = self.get_local_slices(img_shape)
        return {k: v[(..., *slices)].contiguous() for k, v in data.items()}

    def gather_spatial(
        self, data: dict[str, torch.Tensor], img_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Gather local spatial chunks back to global tensors via all-reduce."""
        slices = self.get_local_slices(img_shape)
        result = {}
        for k, v in data.items():
            global_shape = (*v.shape[:-2], *img_shape)
            global_tensor = torch.zeros(global_shape, dtype=v.dtype, device=v.device)
            global_tensor[(..., *slices)] = v
            result[k] = self.spatial_reduce_sum(global_tensor)
        return result

    def shutdown(self):
        return self._distributed.shutdown()


singleton: Distributed | None = None
