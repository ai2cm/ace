import os
import sys
import types

import pytest
import yaml
from create_coupled_datasets import CreateCoupledDatasetsConfig
from upload_coupled_stats import UploadSpec, _upload, _upload_spec
from upload_stats import STATS_FILENAMES

DIRNAME = os.path.abspath(os.path.dirname(__file__))
SINGLE_CONFIG = os.path.join(
    DIRNAME, "configs", "CM4-piControl-coupled-1deg-1daily-200yr.yaml"
)
ENSEMBLE_CONFIG = os.path.join(
    DIRNAME, "configs", "CM4-like-AM4-random-CO2-ensemble-coupled.yaml"
)


def _config(
    path: str, tmp_path, beaker_dataset: str | None = "coupled-stats"
) -> CreateCoupledDatasetsConfig:
    with open(path) as f:
        config_data = yaml.safe_load(f)
    config_data.setdefault("stats", {})["beaker_dataset"] = beaker_dataset
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)
    return CreateCoupledDatasetsConfig.from_file(str(config_path))


class _BeakerDatasetNotFound(Exception):
    pass


class _FakeDatasetClient:
    def __init__(self, existing: tuple[str, ...] = ()):
        self.existing = set(existing)
        self.created: dict[str, list[str]] = {}

    def get(self, name: str):
        if name not in self.existing:
            raise _BeakerDatasetNotFound(name)

    def create(self, name: str, source: str, workspace: str, description: str):
        self.created[name] = sorted(
            os.path.relpath(os.path.join(root, f), source)
            for root, _, files in os.walk(source)
            for f in files
        )


def _fake_beaker(monkeypatch, existing: tuple[str, ...] = ()):
    module = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(BeakerDatasetNotFound=_BeakerDatasetNotFound)
    )
    monkeypatch.setitem(sys.modules, "beaker", module)
    return types.SimpleNamespace(dataset=_FakeDatasetClient(existing))


def _write_merged_stats(directory, categories: tuple[str, ...]):
    for category in categories:
        os.makedirs(directory / category)
        for filename in STATS_FILENAMES:
            (directory / category / filename).write_bytes(b"stats")


def test_upload_spec_single_dataset(tmp_path):
    config = _config(SINGLE_CONFIG, tmp_path)
    spec = _upload_spec(config)
    assert spec is not None
    assert spec.beaker_dataset == "coupled-stats"
    assert spec.merged_stats_directory == config.coupled_stats_directory
    assert f"{config.version}-{config.family_name}" in spec.description


def test_upload_spec_ensemble_uses_combined_stats(tmp_path):
    config = _config(ENSEMBLE_CONFIG, tmp_path)
    spec = _upload_spec(config)
    assert spec is not None
    assert spec.merged_stats_directory == os.path.join(
        config.coupled_stats_directory, "combined"
    )


def test_upload_spec_none_without_beaker_dataset(tmp_path):
    config = _config(SINGLE_CONFIG, tmp_path, beaker_dataset=None)
    assert config.stats.beaker_dataset is None
    assert _upload_spec(config) is None


def test_upload_creates_one_subdirectory_per_category(tmp_path, monkeypatch):
    _write_merged_stats(tmp_path, ("uncoupled_atmosphere", "ocean"))
    client = _fake_beaker(monkeypatch)
    _upload(client, UploadSpec("coupled-stats", str(tmp_path), "description"))
    assert client.dataset.created == {
        "coupled-stats": sorted(
            os.path.join(category, filename)
            for category in ("ocean", "uncoupled_atmosphere")
            for filename in STATS_FILENAMES
        )
    }


def test_upload_skips_existing_dataset(tmp_path, monkeypatch):
    client = _fake_beaker(monkeypatch, existing=("coupled-stats",))
    _upload(client, UploadSpec("coupled-stats", str(tmp_path), "description"))
    assert client.dataset.created == {}


def test_upload_raises_without_merged_stats(tmp_path, monkeypatch):
    client = _fake_beaker(monkeypatch)
    with pytest.raises(FileNotFoundError):
        _upload(client, UploadSpec("coupled-stats", str(tmp_path), "description"))
