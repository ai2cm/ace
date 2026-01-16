from pathlib import Path

import pytest

from fme.core.cloud import is_local, mkdirs


@pytest.mark.parametrize(
    ("path, expected"),
    [
        ("/absolute/path/somefile", True),
        ("relative/path/somefile", True),
        ("file://path/somefile", True),
        ("local://path/somefile", True),
        pytest.param(Path("/absolute/path/somefile"), True, id="Path object"),
        ("gs://mybucket/somefile", False),
    ],
)
def test_is_local(path: str | Path, expected: bool):
    assert is_local(path) == expected


@pytest.mark.parametrize("use_str_input", [True, False])
def test_mkdirs(tmp_path: Path, use_str_input: bool):
    path = tmp_path / "test" / "mkdirs"
    assert not path.exists()
    input_path = str(path) if use_str_input else path

    mkdirs(input_path)
    assert path.exists()
    assert path.is_dir()

    mkdirs(input_path, exist_ok=True)
    assert path.exists()
    assert path.is_dir()

    with pytest.raises(FileExistsError):
        mkdirs(input_path)
