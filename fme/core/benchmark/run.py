import argparse
import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import sys

import torch

from fme.core.benchmark.benchmark import get_benchmarks

RESULTS_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "results"

_GIT_COMMIT: str | None = None


def get_git_commit() -> str:
    global _GIT_COMMIT
    if _GIT_COMMIT is None:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        # Non-empty output means repo is dirty
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        if dirty:
            commit = f"{commit}-dirty"

        _GIT_COMMIT = commit

    return _GIT_COMMIT


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return "CPU"


def main(
    name: str | None, iters: int, output_dir: pathlib.Path, child: str | None = None
) -> int:
    output_dir.mkdir(exist_ok=True)
    device_name = get_device_name()

    logging.info(f"Running benchmarks on device: {device_name}")
    benchmarks = get_benchmarks()
    if name is not None:
        if name not in benchmarks:
            logging.error(
                f"Specified benchmark {name} not found. "
                f"Available benchmarks: {', '.join(benchmarks.keys())}"
            )
            return 1
        benchmarks_to_run = {name: benchmarks[name]}
    else:
        benchmarks_to_run = benchmarks

    def get_label(name):
        return f"{name} on {device_name} at commit {get_git_commit()}"

    def get_filename(name, extension) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        safe_device_name = device_name.replace(" ", "_").replace("/", "_").lower()
        return (
            output_dir
            / f"{safe_name}_{safe_device_name}_{get_git_commit()}.{extension}"
        )

    for name, cls in benchmarks_to_run.items():
        logging.info(f"Running benchmark: {name}")
        result = cls.run_benchmark(iters=iters)
        png_filename = get_filename(name, "png")
        logging.info(f"Saving result image to {png_filename}")
        result.to_png(png_filename, label=get_label(name))
        result_data = json.dumps(dataclasses.asdict(result), indent=2)
        logging.info(f"Result: {result_data}")
        with open(get_filename(name, "json"), "w") as f:
            logging.info(f"Saving result json to {f.name}")
            f.write(result_data)
        if child is not None:
            child_name = f"{name}.{child}"
            child_label = get_label(child_name)
            logging.info(f"Generating benchmark result for child timer: {child_label}")
            png_filename = get_filename(child_name, "png")
            logging.info(f"Saving child result image to {png_filename}")
            result.to_png(png_filename, label=child_label, child=child)
    return 0


def get_benchmark_label(name):
    device_name = get_device_name()
    return f"{name} on {device_name} at commit {get_git_commit()}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run registered benchmarks.")
    parser.add_argument(
        "--name",
        type=str,
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save benchmark results in. If not provided, "
            "results will be saved in a 'results' directory next to this script."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = RESULTS_PATH

    sys.exit(
        main(
            name=args.name,
            iters=args.iters,
            child=args.child,
            output_dir=output_dir,
        )
    )
