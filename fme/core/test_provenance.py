import json

import pytest
import yaml

import fme.core.wandb
from fme.core.provenance import validate_output_dir, write_provenance
from fme.core.testing.wandb import mock_wandb


def test_write_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAKER_EXPERIMENT_ID", "beaker-id")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config_data = {"experiment_dir": "job", "nested": {"a": 1}}
    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        wandb.init()
        run_id = wandb.get_id()
        write_provenance(str(output_dir), config_data, experiment_dir="job")
    assert run_id is not None
    with open(output_dir / "config.yaml") as f:
        assert yaml.safe_load(f) == config_data
    with open(output_dir / "provenance.json") as f:
        provenance = json.load(f)
    assert provenance["experiment_dir"] == "job"
    assert provenance["output_dir"] == str(output_dir)
    assert provenance["beaker_experiment_id"] == "beaker-id"
    assert provenance["wandb_run_id"] == run_id
    assert provenance["environment"]["BEAKER_EXPERIMENT_ID"] == "beaker-id"


def test_write_provenance_without_wandb(tmp_path, monkeypatch):
    # a fresh, never-initialized WandB has no run id
    monkeypatch.setattr(fme.core.wandb, "singleton", fme.core.wandb.WandB())
    write_provenance(str(tmp_path), {"a": 1}, experiment_dir="job")
    with open(tmp_path / "provenance.json") as f:
        assert json.load(f)["wandb_run_id"] is None


@pytest.mark.parametrize(
    "experiment_dir, output_dir, raises",
    [
        ("/local/job", None, False),
        ("gs://bucket/job", None, False),
        ("/local/job", "gs://bucket/output", False),
        ("gs://bucket/job", "gs://bucket/output", True),
        ("gs://bucket/job", "/local/output", True),
    ],
)
def test_validate_output_dir(experiment_dir, output_dir, raises):
    if raises:
        with pytest.raises(ValueError, match="experiment_dir must be on a local"):
            validate_output_dir(experiment_dir, output_dir)
    else:
        validate_output_dir(experiment_dir, output_dir)
