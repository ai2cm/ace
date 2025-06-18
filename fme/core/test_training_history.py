from .testing import mock_distributed, mock_wandb
from .training_history import TrainingHistory, TrainingJob, get_job_id
from .wandb import WandB


def test_training_history_append_extend():
    training_history = TrainingHistory()

    job1 = TrainingJob(
        git_sha="abc123",
        job_id="run_456",
    )

    # Append the job to the history
    training_history.append(job1)

    # Check if the job is appended correctly
    assert len(training_history) == 1
    assert training_history[0].git_sha == "abc123"
    assert training_history[0].job_id == "run_456"

    # Extend the history with another TrainingHistory instances
    job2 = TrainingJob(
        git_sha="def456",
        job_id="run_789",
    )
    TrainingHistory2 = TrainingHistory([job2])
    training_history.extend(TrainingHistory2)

    # Check if the job is appended correctly
    assert len(training_history) == 2
    assert training_history[1].git_sha == "def456"
    assert training_history[1].job_id == "run_789"

    # Get the history of a specific key
    training_history_by_key = training_history.get_history_by_key()
    assert training_history_by_key["git_sha"] == ["abc123", "def456"]
    assert training_history_by_key["job_id"] == ["run_456", "run_789"]


def test_stepper_training_metadata_roundtrip():
    history = TrainingHistory()
    job1 = TrainingJob(
        git_sha="abc123",
        job_id="run_456",
    )
    history.append(job1)

    # roundtrip the training history
    new_history = TrainingHistory.from_state(history.get_state())
    assert new_history == history


def test_get_job_id_wandb():
    with mock_wandb(), mock_distributed():
        wandb = WandB.get_instance()
        assert get_job_id() is None, "wandb is not enabled/configured"
        wandb.configure(log_to_wandb=True)
        wandb.init()
        job_id = get_job_id()
        assert wandb.get_id() == job_id
