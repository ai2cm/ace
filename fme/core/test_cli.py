import os

from .cli import prepare_config, prepare_directory


def test_prepare_config(tmp_path):
    path = tmp_path / "file.yaml"
    with open(path, "w") as f:
        f.write("a: 2")
    output = prepare_config(path)
    assert output == {"a": 2}


def test_prepare_config_override(tmp_path):
    path = tmp_path / "file.yaml"
    with open(path, "w") as f:
        f.write("a: 2")
    output = prepare_config(path, ["a=3"])
    assert output == {"a": 3}


def test_prepare_directory(tmp_path):
    path = tmp_path / "test"
    config_data = {"a": 2}
    prepare_directory(path, config_data)
    assert os.path.exists(path / "config.yaml")
