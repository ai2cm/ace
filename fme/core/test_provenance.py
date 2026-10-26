import dataclasses
import json
import os

import fsspec
import pytest
import yaml

import fme
from fme.core.distributed import Distributed
from fme.core.provenance import write_provenance
from fme.core.testing.output_dir import OUTPUT_DIR_KINDS, output_dir_of_kind


@dataclasses.dataclass
class _Logging:
    entity: str = "my-entity"
    project: str = "my-project"


@dataclasses.dataclass
class _Config:
    experiment_dir: str
    logging: _Logging = dataclasses.field(default_factory=_Logging)


def _read_outputs(output_dir: str) -> tuple[dict, dict]:
    with fsspec.open(os.path.join(output_dir, "config.yaml")) as f:
        config_data = yaml.safe_load(f)
    with fsspec.open(os.path.join(output_dir, "provenance.json")) as f:
        provenance = json.load(f)
    return config_data, provenance


@pytest.mark.parametrize("kind", OUTPUT_DIR_KINDS)
def test_write_provenance(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("BEAKER_EXPERIMENT_ID", "exp-id")
    monkeypatch.delenv("BEAKER_JOB_ID", raising=False)
    config = _Config(experiment_dir=str(tmp_path / "job"))
    with output_dir_of_kind(kind, tmp_path) as output_dir:
        write_provenance(output_dir, config, config.experiment_dir)
        config_data, provenance = _read_outputs(output_dir)
    assert config_data == dataclasses.asdict(config)
    written_at = provenance.pop("written_at")
    assert written_at.endswith("+00:00")
    assert provenance == {
        "experiment_dir": config.experiment_dir,
        "output_dir": output_dir,
        "beaker_experiment_id": "exp-id",
        "beaker_job_id": None,
        "wandb": {"entity": "my-entity", "project": "my-project", "run_id": None},
        "fme_version": fme.__version__,
    }


def test_write_provenance_non_root_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(Distributed.get_instance(), "is_root", lambda: False)
    output_dir = tmp_path / "output"
    config = _Config(experiment_dir=str(tmp_path / "job"))
    write_provenance(str(output_dir), config, config.experiment_dir)
    assert not output_dir.exists()
