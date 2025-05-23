import subprocess

from fme.core.wandb import WandB


def get_job_env() -> dict[str, str | None]:
    return {
        "git_sha": git_revision_short_hash(),
        "job_id": get_job_id(),
    }


def git_revision_short_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def get_job_id() -> str | None:
    """
    Get the wandb ID of the current job.
    """
    wandb = WandB.get_instance()
    if wandb.enabled and wandb.configured:
        return wandb.get_id()
    else:
        return None
