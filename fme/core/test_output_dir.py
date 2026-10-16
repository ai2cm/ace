import pytest

from fme.core.output_dir import (
    is_separate_output_dir,
    resolve_output_dir,
    validate_output_dir,
)


@pytest.mark.parametrize(
    "experiment_dir, output_dir, expected",
    [
        ("/job", None, "/job"),
        ("/job", "/output", "/output"),
        ("/job", "memory://bucket/output", "memory://bucket/output"),
        ("memory://bucket/job", None, "memory://bucket/job"),
    ],
)
def test_valid_output_dir(experiment_dir, output_dir, expected):
    validate_output_dir(experiment_dir, output_dir)
    assert resolve_output_dir(experiment_dir, output_dir) == expected


@pytest.mark.parametrize("output_dir", ["/output", "memory://bucket/output"])
def test_output_dir_requires_local_experiment_dir(output_dir):
    with pytest.raises(ValueError, match="experiment_dir must be on a local"):
        validate_output_dir("memory://bucket/job", output_dir)


@pytest.mark.parametrize(
    "output_dir, expected",
    [
        (None, False),
        ("/job/", False),
        ("/job/../job", False),
        ("/output", True),
        ("memory://bucket/output", True),
    ],
)
def test_is_separate_output_dir(output_dir, expected):
    assert is_separate_output_dir("/job", output_dir) == expected
