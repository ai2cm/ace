from pathlib import Path

import pytest

from fme.core.cloud import is_local


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
