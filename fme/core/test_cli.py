import os

from .cli import prepare_config, prepare_directory, remove_stale_tmp_checkpoints


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


def test_remove_stale_tmp_checkpoints_removes_multiple_tmp_files(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    tmp1 = ckpt_dir / ".uuid-1.tmp"
    tmp2 = ckpt_dir / ".uuid-2.tmp"
    tmp1.write_bytes(b"data1")
    tmp2.write_bytes(b"data2")
    valid_ckpt = ckpt_dir / "latest.ckpt"
    valid_ckpt.write_bytes(b"valid")

    remove_stale_tmp_checkpoints(str(ckpt_dir))

    assert not tmp1.exists()
    assert not tmp2.exists()
    assert valid_ckpt.exists()


def test_remove_stale_tmp_checkpoints_nonexistent_dir(tmp_path):
    remove_stale_tmp_checkpoints(str(tmp_path / "does_not_exist"))


def test_remove_stale_tmp_checkpoints_no_tmp_files(tmp_path):
    ckpt_dir = tmp_path / "training_checkpoints"
    ckpt_dir.mkdir()
    valid_ckpt = ckpt_dir / "ckpt.tar"
    valid_ckpt.write_bytes(b"valid checkpoint")

    remove_stale_tmp_checkpoints(str(ckpt_dir))

    assert valid_ckpt.exists()
