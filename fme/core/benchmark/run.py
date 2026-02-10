import argparse
import os
import pathlib
import subprocess

import torch

from fme.core.benchmark.benchmark import get_benchmarks

RESULTS_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "results"

_GIT_COMMIT: str | None = None


def get_git_commit() -> str:
    global _GIT_COMMIT
    if _GIT_COMMIT is None:
        args = ["git", "rev-parse", "--short", "HEAD"]
        _GIT_COMMIT = (
            subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        )
    return _GIT_COMMIT


def main(names: list[str] | None, iters: int, child: str | None = None) -> None:
    RESULTS_PATH.mkdir(exist_ok=True)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_properties(0).name
    else:
        device_name = "CPU"

    print(f"Running benchmarks on device: {device_name}")
    benchmarks = get_benchmarks()
    if names is not None:
        if any(name not in benchmarks for name in names):
            print("Some specified benchmarks not found. Available benchmarks:")
            for name in benchmarks:
                print(f"  - {name}")
            return
        benchmarks_to_run = {name: benchmarks[name] for name in names}
    else:
        benchmarks_to_run = benchmarks

    def get_label(name):
        return f"{name} on {device_name} at commit {get_git_commit()}"

    def get_filename(name) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        safe_device_name = device_name.replace(" ", "_").replace("/", "_").lower()
        return RESULTS_PATH / f"{safe_name}_{safe_device_name}_{get_git_commit()}.png"

    for name, cls in benchmarks_to_run.items():
        print(f"Running benchmark: {name}")
        result = cls.run_benchmark(iters=iters)
        result.to_png(get_filename(name), label=get_label(name))
        if child is not None:
            child_name = f"{name}.{child}"
            child_label = get_label(child_name)
            print(f"  Generating report for child timer: {child_label}")
            result.to_png(get_filename(child_name), label=child_label, child=child)
        print(f"  Result: {result}")


def get_benchmark_label(name):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_properties(0).name
    else:
        device_name = "CPU"
    return f"{name} on {device_name} at commit {get_git_commit()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run registered benchmarks.")
    parser.add_argument(
        "benchmark",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Name of the benchmark to run. If not provided, "
            "all benchmarks will be run."
        ),
    )
    parser.add_argument(
        "--child",
        type=str,
        default=None,
        help=(
            "If provided, the child timer to generate a report for. "
            "This should be a dot-separated path to a child timer, "
            "e.g. 'forward' or 'forward.linear'."
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Number of iterations to run each benchmark for.",
    )
    args = parser.parse_args()

    main(
        names=[args.benchmark] if args.benchmark else None,
        iters=args.iters,
        child=args.child,
    )
