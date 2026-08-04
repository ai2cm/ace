import os
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.utils.data

from fme import get_device
from fme.core.distributed import Distributed, model_torch_distributed, torch_distributed
from fme.core.distributed.external.pnd_manager import DistributedManager
from fme.core.distributed.model_torch_distributed import ModelTorchDistributed
from fme.core.distributed.stop_agreement import SoloStopAgreement
from fme.core.distributed.torch_distributed import (
    TorchDistributed,
    _gather_irregular,
    _pad_tensor_at_end,
    _unpad_tensor_at_end,
)


@pytest.mark.medium_duration
def test_context_tears_down_the_backend_on_sigterm():
    """Every entrypoint wraps its work in `Distributed.context()`.

    Handling the signal here rather than in the Trainer is what puts a graceful
    teardown on the inference and evaluation paths, and on the startup phase of
    training before the Trainer exists.

    Runs in a subprocess because the test session is itself already inside a
    `Distributed.context()`, which refuses to nest.
    """
    program = textwrap.dedent(
        """
        import signal
        from fme.core.distributed import Distributed
        from fme.core.distributed.shutdown import add_post_shutdown_callback

        Distributed.get_instance().shutdown = lambda: print("shutdown")
        with Distributed.context():
            add_post_shutdown_callback(lambda: print("callback"))
            signal.raise_signal(signal.SIGTERM)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, timeout=120, text=True
    )

    assert result.stdout.split() == ["shutdown", "callback"]
    assert result.returncode == 128 + signal.SIGTERM


@pytest.mark.medium_duration
def test_context_leaves_signals_alone_when_asked_not_to_handle_them():
    """The test suite wraps the whole session in a context and needs Ctrl-C.

    A handler installed for the session's lifetime turns the first Ctrl-C into a
    caught teardown -- a test failure, with the rest of the run left without a
    process group -- instead of stopping pytest. So the suite opts out, and this
    covers the opt-out: the process keeps whatever disposition it had.

    A subprocess for the same reason as the test above: the session is already
    inside a context, which refuses to nest.
    """
    program = textwrap.dedent(
        """
        import signal
        from fme.core.distributed import Distributed

        before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))
        with Distributed.context(handle_signals=False):
            during = (
                signal.getsignal(signal.SIGTERM),
                signal.getsignal(signal.SIGINT),
            )
        print(before == during)
        # the default disposition survives, so Ctrl-C still raises here
        print(signal.getsignal(signal.SIGINT) is signal.default_int_handler)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, timeout=120, text=True
    )

    assert result.stdout.split() == ["True", "True"], result.stderr
    assert result.returncode == 0


def test_torch_shutdown_is_a_noop_when_the_process_group_is_gone(monkeypatch):
    """A termination signal can arrive during the normal end-of-run teardown.

    `destroy_process_group` raises when there is no group left, which would turn
    a second teardown into a logged exception on the way out.
    """
    destroyed = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", lambda *args: destroyed.append(args)
    )

    TorchDistributed.shutdown(SimpleNamespace(rank=0))  # type: ignore[arg-type]

    assert destroyed == []


def test_model_shutdown_still_cleans_up_when_the_process_group_is_gone(monkeypatch):
    """The spatial backend has manager state to reset even with no group left.

    `DistributedManager.cleanup` skips the collective by itself when the manager
    is not initialized, and clears its shared state either way, so skipping the
    call would strand that state and leave the manager claiming to be up.
    """
    cleaned = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        model_torch_distributed.DistributedManager,
        "cleanup",
        classmethod(lambda cls: cleaned.append("cleanup")),
    )

    ModelTorchDistributed.shutdown(SimpleNamespace(_rank=0))  # type: ignore[arg-type]

    assert cleaned == ["cleanup"]


@pytest.mark.parametrize(
    ["padding", "fill_value"],
    [
        pytest.param([0, 0, 0], None, id="no_padding"),
        pytest.param([1, 1, 1], 0.0, id="padding_1"),
        pytest.param([1, 1, 1], 1.0, id="padding_1_fill_one"),
    ],
)
def test_pad_tensor_at_end(padding, fill_value):
    tensor = torch.ones(2, 3, 4)
    padded_tensor = _pad_tensor_at_end(tensor, padding, fill_value)
    assert padded_tensor.size() == (2 + padding[0], 3 + padding[1], 4 + padding[2])
    for dim, pad in enumerate(padding):
        if pad > 0:
            assert torch.allclose(
                padded_tensor.select(dim=dim, index=padded_tensor.size(dim) - 1),
                torch.tensor(fill_value),
            )


def test_force_non_distributed():
    assert not Distributed.get_instance()._force_non_distributed
    with Distributed.force_non_distributed():
        assert Distributed.get_instance()._force_non_distributed


@pytest.mark.parametrize(
    ["padding"],
    [
        pytest.param([0, 0, 0], id="no_padding"),
        pytest.param([1, 1, 1], id="padding_1"),
    ],
)
def test_pad_unpad_rountrip(padding):
    tensor = torch.ones(2, 3, 4, device=get_device())
    padded_tensor = _pad_tensor_at_end(tensor, padding)
    unpadded_tensor = _unpad_tensor_at_end(padded_tensor, padding)
    assert unpadded_tensor.size() == tensor.size()
    assert torch.allclose(unpadded_tensor, tensor)


def run_gather_test(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(worldsize)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_SHM_DISABLE"] = "1"

    dist = Distributed()
    assert dist.is_distributed()
    assert dist.rank == rank, f"Failed with Distrubuted rank {dist.rank} and arg {rank}"

    tensor = torch.ones(2, 5, device=get_device()) * dist.rank
    gathered = dist.gather(tensor)
    if dist.rank == 0:
        assert gathered is not None, "Gathered tensor are none instead of List"
        assert len(gathered) == dist.world_size
        for i in range(dist.world_size):
            assert torch.allclose(gathered[i].cpu(), torch.ones(2, 5) * i)


@pytest.mark.serial
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
def test_distributed_gather():
    world_size = 2
    mp.spawn(run_gather_test, args=(world_size,), nprocs=world_size, join=True)


@pytest.mark.parallel
def test_scatter_object():
    dist = Distributed()
    if dist.is_root():
        obj = {"key": "value"}
    else:
        obj = None
    scattered = dist.scatter_object(obj)
    assert scattered == {"key": "value"}


def test_non_distributed_gather():
    dist = Distributed()
    assert not dist.is_distributed()
    tensor = torch.ones(2, 5, device=get_device()) * 5
    gathered = dist.gather(tensor)
    assert gathered is not None, "Gathered tensor are none instead of List"
    assert len(gathered) == 1
    assert torch.allclose(gathered[0], tensor)


def test_gather_irregular():
    tensors = [torch.randn(10 + i) for i in range(8)]
    tensor_lengths = set(len(t) for t in tensors)
    max_length = torch.tensor(max(tensor_lengths))
    reduce_max_seen = []

    def reduce_max(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 0
        reduce_max_seen.append(int(tensor))
        tensor.fill_(max_length)
        return tensor

    gather_nonscalar_seen: list[torch.Tensor] = []
    gather_scalar_seen: list[torch.Tensor] = []

    def gather(tensor: torch.Tensor) -> list[torch.Tensor]:
        if tensor.ndim == 0 or len(tensor) == 1:
            gather_scalar_seen.append(tensor)
            return gather_scalar_seen  # not correct, but it's OK for this test
        else:
            gather_nonscalar_seen.append(tensor)
            return gather_nonscalar_seen

    gathered = []
    for tensor in tensors:
        gathered_item = _gather_irregular(tensor, reduce_max, gather, fill_value=0.0)
        assert isinstance(gathered_item, list)
        assert all(isinstance(item, torch.Tensor) for item in gathered_item)
        gathered.append(gathered_item)
    final_gathered = gathered_item
    assert final_gathered is not None
    assert set(reduce_max_seen) == tensor_lengths
    assert set(tensor.shape[0] for tensor in final_gathered) == tensor_lengths, (
        "Final gathered shapes are "
        f"{set(tensor.shape[0] for tensor in final_gathered)} "
        f"but expected {tensor_lengths}, "
        "did the tensors get un-padded back to their original lengths?"
    )
    assert gathered is not None
    assert len(gathered) == 8
    gathered_nonscalar_shapes = [t.shape for t in gather_nonscalar_seen]
    assert all(
        shape == gathered_nonscalar_shapes[0] for shape in gathered_nonscalar_shapes
    )


def _set_torchrun_env(monkeypatch, rank: int, world_size: int, local_rank: int):
    monkeypatch.delenv("FME_USE_SRUN", raising=False)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))


def _set_srun_env(monkeypatch, rank: int, world_size: int, local_rank: int):
    monkeypatch.setenv("FME_USE_SRUN", "1")
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    monkeypatch.setenv("SLURM_NTASKS", str(world_size))
    monkeypatch.setenv("SLURM_LOCALID", str(local_rank))
    monkeypatch.setenv(
        "SRUN_DIST_FILE_PATH", "/tmp/unused-init-process-group-is-stubbed"
    )


def _forbid_cuda_and_process_group_init(monkeypatch):
    """Make every CUDA and process group entry point a worker must avoid fail."""

    def fail(*args, **kwargs):
        raise AssertionError("must not be called in a DataLoader worker")

    monkeypatch.setattr(torch.distributed, "init_process_group", fail)
    monkeypatch.setattr(torch.cuda, "set_device", fail)
    # get_device() calls current_device(), which lazily initializes CUDA
    monkeypatch.setattr(torch.cuda, "current_device", fail)


def _pretend_dataloader_worker(monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())


LAUNCHER_ENVS = [
    pytest.param(_set_torchrun_env, id="torchrun"),
    pytest.param(_set_srun_env, id="srun"),
]

LAUNCHER_ENVS_AND_DEVICE_IDS = [
    pytest.param(_set_torchrun_env, 1, id="torchrun"),
    pytest.param(_set_srun_env, 0, id="srun"),
]


@pytest.mark.parametrize("set_launcher_env", LAUNCHER_ENVS)
def test_dataloader_worker_skips_cuda_and_process_group_init(
    monkeypatch, set_launcher_env
):
    """A DataLoader worker gets usable rank metadata without initializing CUDA.

    Each CUDA context costs hundreds of MiB of GPU memory, and workers have no
    use for one, so a worker must not set the CUDA device or join the process
    group.
    """
    set_launcher_env(monkeypatch, rank=3, world_size=8, local_rank=1)
    _pretend_dataloader_worker(monkeypatch)
    _forbid_cuda_and_process_group_init(monkeypatch)

    dist = torch_distributed.TorchDistributed()

    assert dist.rank == 3
    assert dist.total_ranks == 8


@pytest.mark.parametrize("set_launcher_env", LAUNCHER_ENVS)
def test_spatial_parallel_dataloader_worker_skips_cuda_and_process_group_init(
    monkeypatch, set_launcher_env
):
    """A worker under spatial parallelism also gets metadata CUDA-free.

    The data-parallel index and size are derived from the row-major
    (data, h, w) mesh layout rather than from the DeviceMesh, which cannot be
    built without the process group the worker skips.
    """
    set_launcher_env(monkeypatch, rank=3, world_size=8, local_rank=1)
    _pretend_dataloader_worker(monkeypatch)
    _forbid_cuda_and_process_group_init(monkeypatch)

    def fail_initialize():
        raise AssertionError("must not be called in a DataLoader worker")

    monkeypatch.setattr(DistributedManager, "initialize", fail_initialize)

    dist = ModelTorchDistributed(h_size=2, w_size=1)

    assert dist.rank == 3
    assert dist.total_ranks == 8
    # rank 3 is the second (h, w) group of the second data-parallel slice
    assert dist.data_parallel_rank == 1
    assert dist.total_data_parallel_ranks == 4


@pytest.mark.parametrize(
    "set_launcher_env,expected_device_id", LAUNCHER_ENVS_AND_DEVICE_IDS
)
def test_dataloader_worker_guard_does_not_affect_main_process(
    monkeypatch, set_launcher_env, expected_device_id
):
    """Outside a worker the CUDA device is still set, as training requires."""
    set_launcher_env(monkeypatch, rank=3, world_size=8, local_rank=1)
    monkeypatch.setattr(torch_distributed, "using_gpu", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    # The constructor joins a world-wide gloo agreement group immediately after
    # `init_process_group`, which this test skips by answering `is_initialized()`
    # `True`. `new_group` is a real collective and there is no default group here
    # for it to run on, so it is stubbed alongside `init_process_group` for the
    # same reason. This test asserts nothing about the agreement group.
    monkeypatch.setattr(
        torch_distributed, "new_stop_agreement", lambda world_size: SoloStopAgreement()
    )
    monkeypatch.setattr(
        torch.distributed.distributed_c10d, "_get_default_group", lambda: None
    )
    set_devices: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)

    dist = torch_distributed.TorchDistributed()

    assert dist.rank == 3
    assert dist.total_ranks == 8
    assert set_devices == [expected_device_id]


def test_dataloader_worker_without_launcher_env_raises(monkeypatch):
    """A worker with no launcher environment reports why, not a KeyError."""
    monkeypatch.delenv("FME_USE_SRUN", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    _pretend_dataloader_worker(monkeypatch)

    with pytest.raises(ValueError, match="without torchrun or srun"):
        torch_distributed.TorchDistributed()
