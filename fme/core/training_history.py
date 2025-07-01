import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import overload

from fme.core.wandb import WandB


@dataclass
class TrainingJob:
    """
    Metadata of a stepper training job.

    Parameters:
        git_sha: git sha of the training job
        job_id: wandb run id of the training job
    """

    git_sha: str | None = None
    job_id: str | None = None

    @classmethod
    def from_env(cls) -> "TrainingJob":
        return cls(
            git_sha=git_revision_short_hash(),
            job_id=get_job_id(),
        )


class TrainingHistory(Sequence):
    """Metadata of a stepper's training history.

    Parameters:
        training_jobs: list of training jobs
    """

    def __init__(self, training_jobs: list[TrainingJob] | None = None):
        self._training_jobs: list[TrainingJob] = training_jobs or []

    @overload
    def __getitem__(self, idx: int) -> TrainingJob:
        pass

    @overload
    def __getitem__(self, idx: slice) -> list[TrainingJob]:
        pass

    def __getitem__(self, idx):
        return self._training_jobs[idx]

    def __len__(self) -> int:
        return len(self._training_jobs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingHistory):
            return False
        else:
            return self._training_jobs == other._training_jobs

    def _keys(self) -> set[str]:
        return {field.name for field in fields(TrainingJob)}

    def append(self, job: TrainingJob) -> None:
        """Append a training job to the metadata."""
        self._training_jobs.append(job)

    def extend(self, other_history: "TrainingHistory") -> None:
        """Extend the metadata with another training history."""
        self._training_jobs.extend(other_history)

    def get_history_by_key(self) -> dict[str, list[str]]:
        """Get the history of the training jobs organized by key."""
        return {key: self._get_key_history(key) for key in self._keys()}

    def _get_key_history(self, key: str) -> list[str]:
        return [getattr(job, key) for job in self._training_jobs]

    def get_state(self) -> list[dict[str, str]]:
        return [asdict(job) for job in self._training_jobs]

    @classmethod
    def from_state(cls, state: list[dict[str, str]]) -> "TrainingHistory":
        return cls([TrainingJob(**job) for job in state])


def get_job_id() -> str | None:
    """
    Get the wandb ID of the current job.
    """
    wandb = WandB.get_instance()
    if wandb.enabled and wandb.configured:
        return wandb.get_id()
    else:
        return None


def git_revision_short_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return None


def get_job_env() -> dict[str, str | None]:
    """
    Get the environment of the current job, including git SHA and wandb job ID.
    """
    return dict(git_sha=git_revision_short_hash(), job_id=get_job_id())
