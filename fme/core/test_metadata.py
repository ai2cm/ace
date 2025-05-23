from .metadata import get_job_id
from .testing import mock_distributed, mock_wandb
from .wandb import WandB


def test_get_job_id_wandb():
    with mock_wandb(), mock_distributed():
        wandb = WandB.get_instance()
        assert get_job_id() is None, "wandb is not enabled/configured"
        wandb.configure(log_to_wandb=True)
        wandb.init()
        job_id = get_job_id()
        assert wandb.get_id() == job_id
