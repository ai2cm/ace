import pytest
import torch

from fme.core.distributed.nccl_abort_probe import run_probe

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two CUDA devices",
)


@pytest.mark.slow
@pytest.mark.serial
@requires_two_gpus
def test_abort_on_sigterm_during_live_collectives(tmp_path):
    assert run_probe(nproc=2, work_dir=tmp_path, scenarios=("healthy",))


@pytest.mark.slow
@pytest.mark.serial
@requires_two_gpus
def test_abort_on_sigterm_releases_wedged_rank(tmp_path):
    assert run_probe(nproc=2, work_dir=tmp_path, scenarios=("wedge",))
