import pathlib

EXCLUDED_FILES = {
    "downscaling/modules/swinir.py",
    "core/cuhpx/data/convert_fits_to_npy.py",
}


def _iter_python_files(root: pathlib.Path):
    for path in root.rglob("*.py"):
        if path.name in EXCLUDED_FILES:
            continue
        if path.name == pathlib.Path(__file__).name:
            continue
        yield path


def _has_main_guard(source: str) -> bool:
    return 'if __name__ == "__main__":' in source


def _has_distributed_context(source: str) -> bool:
    return "Distributed.context()" in source


def test_main_guard_requires_distributed_context():
    root = pathlib.Path(__file__).parent

    failures = []

    for path in _iter_python_files(root):
        source = path.read_text(encoding="utf-8")

        if _has_main_guard(source) and not _has_distributed_context(source):
            failure_path = str(path.relative_to(root))
            if failure_path not in EXCLUDED_FILES:
                failures.append(failure_path)

    if failures:
        raise AssertionError(
            "Files with main guard missing Distributed.context():\n"
            + "\n".join(failures)
        )
