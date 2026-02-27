"""Vendored PhysicsNeMo distributed (pnd) manager.

Based on the following file:
https://github.com/NVIDIA/physicsnemo/blob/62adbe43da94615b3843dbee866bd7af8939bc91/physicsnemo/distributed/manager.py

We make some edits (formatting, removal of version checks and decorators,
and removal of unused functions) to fit the current needs of FME.
We also require a minimum of torch 2.4.0 for FME as a result.
"""

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
from warnings import warn

import numpy as np
import torch
import torch.distributed as dist


class PhysicsNeMoUninitializedDistributedManagerWarning(Warning):
    """Warning to indicate usage of an uninitialized DistributedManager."""

    def __init__(self):
        message = (
            "A DistributedManager object is being instantiated before "
            + "this singleton class has been initialized. "
            + "Instantiating a manager before "
            + "initialization can lead to unexpected results where processes fail "
            + "to communicate. Initialize the distributed manager via "
            + "DistributedManager.initialize() before instantiating."
        )
        super().__init__(message)


class DistributedManager:
    """Distributed Manager for setting up distributed training environment.

    This is a singleton that creates a persistence class instance for storing parallel
    environment information throughout the lifetime of the program. This should be
    used to help set up Distributed Data Parallel and parallel datapipes.

    Note:
    ----
    One should call `DistributedManager.initialize()` prior to constructing a manager
    object

    Example:
    -------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """

    _shared_state: dict[str, bool] = {}

    # Instance attribute type declarations. Attributes are shared across all
    # instances via ``obj.__dict__ = cls._shared_state`` in ``__new__``.
    _rank: int
    _world_size: int
    _local_rank: int
    _distributed: bool
    _device: torch.device
    _cuda: bool
    _initialization_method: str
    _is_initialized: bool
    _global_mesh: "torch.distributed.DeviceMesh | None"
    _mesh_dims: "dict[str, int]"
    _mesh_groups: "dict[int, dist.ProcessGroup]"

    def __new__(cls):
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not hasattr(obj, "_cuda"):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, "_initialization_method"):
            obj._initialization_method = "None"
        if not hasattr(obj, "_is_initialized"):
            obj._is_initialized = False
        if not hasattr(obj, "_global_mesh"):
            obj._global_mesh = None  # Lazy initialized right when it's first needed
        if not hasattr(obj, "_mesh_dims"):
            obj._mesh_dims = {}  # Dictionary mapping axis names to sizes

        return obj

    def __init__(self):
        if not self._is_initialized:
            raise PhysicsNeMoUninitializedDistributedManagerWarning()
        super().__init__()

    @property
    def rank(self):
        """Process rank."""
        return self._rank

    @property
    def local_rank(self):
        """Process rank on local machine."""
        return self._local_rank

    @property
    def world_size(self):
        """Number of processes in distributed environment."""
        return self._world_size

    @property
    def device(self):
        """Process device."""
        return self._device

    @property
    def distributed(self):
        """Distributed environment."""
        return self._distributed

    @property
    def cuda(self):
        """If cuda is available."""
        return self._cuda

    def __str__(self):
        output = (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self._initialization_method}'. Device set to {str(self.device)}"
        )
        return output

    @classmethod
    def is_initialized(cls) -> bool:
        """If manager singleton has been initialized."""
        return cls._shared_state.get("_is_initialized", False)

    @staticmethod
    def get_available_backend():
        """Get communication backend."""
        force_cpu = os.environ.get("FME_FORCE_CPU", "0") == "1"
        if (
            not force_cpu
            and torch.cuda.is_available()
            and torch.distributed.is_nccl_available()
        ):
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        """Setup method using generic initialization."""
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if "LOCAL_RANK" in os.environ:
            local_rank_str = os.environ.get("LOCAL_RANK")
            if local_rank_str is not None:
                local_rank = int(local_rank_str)
            else:
                local_rank = rank % torch.cuda.device_count()

        else:
            local_rank = rank % torch.cuda.device_count()

        # Read env variables
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        """Setup method using OpenMPI initialization."""
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        """Setup method using SLURM initialization."""
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        """Initialize the distributed manager.

        Current supported initialization methods are:
            `ENV`: PyTorch environment variable initialization
                https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            `SLURM`: Initialization on SLURM systems.
                Uses `SLURM_PROCID`, `SLURM_NPROCS`, `SLURM_LOCALID` and
                `SLURM_LAUNCH_NODE_IPADDR` environment variables.
            `OPENMPI`: Initialization for OpenMPI launchers.
                Uses `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE` and
                `OMPI_COMM_WORLD_LOCAL_RANK` environment variables.

        Initialization by default is done using the first valid method in the
        order listed above. Initialization method can also be explicitly
        controlled using the
        `PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD` environment variable
        and setting it to one of the options above.
        """
        if DistributedManager.is_initialized():
            warn("Distributed manager is already initialized")
            return

        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        # https://pytorch.org/docs/master/notes/cuda.html#id5
        # was changed in version 2.2
        if torch.__version__ < (2, 2):
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        else:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        initialization_method = os.getenv(
            "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD"
        )
        if initialization_method is None:
            try:
                DistributedManager.initialize_env()
            except TypeError:
                if "SLURM_PROCID" in os.environ:
                    DistributedManager.initialize_slurm(port)
                elif "OMPI_COMM_WORLD_RANK" in os.environ:
                    DistributedManager.initialize_open_mpi(addr, port)
                else:
                    warn(
                        "Could not initialize using ENV, SLURM or OPENMPI "
                        "methods. Assuming this is a single process job"
                    )
                    DistributedManager._shared_state["_is_initialized"] = True
        elif initialization_method == "ENV":
            DistributedManager.initialize_env()
        elif initialization_method == "SLURM":
            DistributedManager.initialize_slurm(port)
        elif initialization_method == "OPENMPI":
            DistributedManager.initialize_open_mpi(addr, port)
        else:
            raise RuntimeError(
                "Unknown initialization method "
                f"{initialization_method}. "
                "Supported values for "
                "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD are "
                "ENV, SLURM and OPENMPI"
            )

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

    def initialize_mesh(
        self, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> "torch.distributed.DeviceMesh":
        """
        Initialize a global device mesh over the entire distributed job.

        Creates a multi-dimensional mesh of processes that can be used for distributed
        operations. The mesh shape must multiply to equal the total world size, with
        one dimension optionally being flexible (-1).

        Parameters
        ----------
        mesh_shape : Tuple[int, ...]
            Tuple of ints describing the size of each mesh dimension. Product must equal
            world_size. One dimension can be -1 to be automatically calculated.

        mesh_dim_names : Tuple[str, ...]
            Names for each mesh dimension. Must match length of mesh_shape.

        Returns:
        -------
        torch.distributed.DeviceMesh
            The initialized device mesh

        Raises:
        ------
        RuntimeError
            If mesh dimensions are invalid or don't match world size
        AssertionError
            If distributed environment is not available
        """
        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError(
                "torch.distributed is unavailable. "
                "Check pytorch build to ensure the distributed package is available. "
                "If building PyTorch from source, set `USE_DISTRIBUTED=1` "
                "to enable the distributed package"
            )

        # Assert basic properties:
        if len(mesh_shape) == 0:
            raise RuntimeError(
                "Device Mesh requires at least one mesh dimension in `mesh_shape`"
            )
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                "mesh_shape and mesh_dim_names must have the same length, but found "
                f"{len(mesh_shape)} and {len(mesh_dim_names)} respectively."
            )
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError("Mesh dimension names must be unique")

        # Check against the total mesh shape vs. world size:
        total_mesh_shape = np.prod(mesh_shape)

        # Allow one shape to be -1
        if -1 in mesh_shape:
            residual_shape = int(self.world_size / (-1 * total_mesh_shape))
            mesh_shape = tuple(residual_shape if m == -1 else m for m in mesh_shape)
            total_mesh_shape = np.prod(mesh_shape)

        if total_mesh_shape != self.world_size:
            raise RuntimeError(
                "Device Mesh num elements must equal world size of "
                f"{total_mesh_shape} but was configured by user with "
                f"global size of {self.world_size}."
            )

        # Actually create the mesh:
        force_cpu = os.environ.get("FME_FORCE_CPU", "0") == "1"
        device_type = "cuda" if (self.cuda and not force_cpu) else "cpu"
        self._global_mesh = dist.init_device_mesh(
            device_type,
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

        # Finally, upon success, cache the mesh dimensions:
        self._mesh_dims = {key: val for key, val in zip(mesh_dim_names, mesh_shape)}

        return self._global_mesh

    def get_mesh_group(self, mesh: "dist.DeviceMesh") -> dist.ProcessGroup:
        """
        Get the process group for a given mesh.

        Creating a group is an expensive operation, so we cache the result manually.

        We hash the mesh and use that as the key.
        """
        key = hash(mesh)

        # Initialize a cache for the groups
        if not hasattr(self, "_mesh_groups"):
            self._mesh_groups: dict[int, dist.ProcessGroup] = {}

        if key in self._mesh_groups.keys():
            return self._mesh_groups[key]
        else:
            if mesh.ndim != 1:
                # We need to get all ranks in this mesh and spawn a group.
                # The mesh.mesh object is a GPU tensor and using it will block.
                ranks = mesh.mesh.cpu()
                ranks = list(ranks.flatten().tolist())
                group = dist.new_group(ranks=ranks, use_local_synchronization=True)
                self._mesh_groups[key] = group
                return group

            else:
                self._mesh_groups[key] = mesh.get_group()
                return mesh.get_group()

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        """Set up PyTorch distributed process group and update manager attributes."""
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        DistributedManager._shared_state["_is_initialized"] = True
        manager = DistributedManager()

        manager._distributed = torch.distributed.is_available()
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank

        force_cpu = os.environ.get("FME_FORCE_CPU", "0") == "1"
        manager._device = torch.device(
            f"cuda:{manager.local_rank}"
            if (torch.cuda.is_available() and not force_cpu)
            else "cpu"
        )

        if manager._distributed:
            # Setup distributed process group.
            # device_id (introduced in PyTorch 2.3) only accepts CUDA devices.
            if manager.device.type == "cuda":
                try:
                    dist.init_process_group(
                        backend,
                        rank=manager.rank,
                        world_size=manager.world_size,
                        device_id=manager.device,
                    )
                except TypeError:
                    # device_id only introduced in PyTorch 2.3
                    dist.init_process_group(
                        backend,
                        rank=manager.rank,
                        world_size=manager.world_size,
                    )
            else:
                dist.init_process_group(
                    backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                )

        if torch.cuda.is_available() and os.environ.get("FME_FORCE_CPU", "0") != "1":
            # Set device for this process and empty cache to optimize memory usage
            torch.cuda.set_device(manager.device)
            torch.cuda.device(manager.device)
            torch.cuda.empty_cache()

        manager._initialization_method = method

    @atexit.register
    @staticmethod
    def cleanup(barrier: bool = False):
        """Clean up distributed group and singleton.

        Parameters
        ----------
        barrier : bool, optional
            Whether to use a global barrier before destroying the
            process group, by default False.
        """
        # Destroying group.WORLD is enough for all process groups to get destroyed
        if (
            "_is_initialized" in DistributedManager._shared_state
            and DistributedManager._shared_state["_is_initialized"]
            and "_distributed" in DistributedManager._shared_state
            and DistributedManager._shared_state["_distributed"]
        ):
            if barrier:
                force_cpu = os.environ.get("FME_FORCE_CPU", "0") == "1"
                if torch.cuda.is_available() and not force_cpu:
                    dist.barrier(device_ids=[DistributedManager().local_rank])
                else:
                    dist.barrier()
            dist.destroy_process_group()
        DistributedManager._shared_state = {}
