"""Helpers for testing entrypoints run with a separate ``output_dir``."""

import contextlib
import json
import os
import pathlib
import uuid
from collections.abc import Iterator, Sequence

import fsspec

OUTPUT_DIR_KINDS = ["local", "memory"]


@contextlib.contextmanager
def output_dir_of_kind(kind: str, tmp_path: pathlib.Path) -> Iterator[str]:
    """Yield a fresh output directory: a local subdirectory of ``tmp_path`` for
    ``"local"``, or a unique ``memory://`` path (removed on exit) standing in
    for a remote store for ``"memory"``.
    """
    if kind == "local":
        yield str(tmp_path / "output")
        return
    output_dir = f"memory://{uuid.uuid4().hex}/output"
    try:
        yield output_dir
    finally:
        fs, _ = fsspec.url_to_fs(output_dir)
        if fs.exists(output_dir):
            fs.rm(output_dir, recursive=True)


def assert_outputs_in_output_dir(
    experiment_dir: str, output_dir: str, output_paths: Sequence[str]
) -> None:
    """Assert the data outputs are in ``output_dir`` and not ``experiment_dir``,
    which keeps the job's config and log file, and that ``output_dir`` has a
    provenance record pointing back to ``experiment_dir``.
    """
    fs, _ = fsspec.url_to_fs(output_dir)
    for path in output_paths:
        assert fs.exists(os.path.join(output_dir, path)), path
        assert not os.path.exists(os.path.join(experiment_dir, path)), path
    for path in ["config.yaml", "inference_out.log"]:
        assert os.path.exists(os.path.join(experiment_dir, path)), path
    assert_provenance(experiment_dir, output_dir)


def assert_provenance(experiment_dir: str, output_dir: str) -> None:
    """Assert ``output_dir`` has a config copy and a ``provenance.json``
    pointing back to ``experiment_dir``.
    """
    fs, _ = fsspec.url_to_fs(output_dir)
    assert fs.exists(os.path.join(output_dir, "config.yaml"))
    with fsspec.open(os.path.join(output_dir, "provenance.json")) as f:
        provenance = json.load(f)
    assert provenance["experiment_dir"] == experiment_dir
