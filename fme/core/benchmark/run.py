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
from fme.core.distributed.distributed import Distributed
from fme.core.wandb import WandB

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
    benchmark_name: str | None,
    iters: int,
    output_dir: pathlib.Path,
    child: str | None = None,
    wandb_project: str | None = None,
) -> int:
    output_dir.mkdir(exist_ok=True)
    device_name = get_device_name()
    safe_device_name = device_name.replace(" ", "_").replace("/", "_").lower()

    logging.info(f"Running benchmarks on device: {device_name}")
    benchmarks = get_benchmarks()
    if benchmark_name is not None:
        if benchmark_name not in benchmarks:
            logging.error(
                f"Specified benchmark {benchmark_name} not found. "
                f"Available benchmarks: {', '.join(benchmarks.keys())}"
            )
            return 1
        benchmarks_to_run = {benchmark_name: benchmarks[benchmark_name]}
    else:
        benchmarks_to_run = benchmarks

    def get_label(name):
        return f"{name} on {device_name} at commit {get_git_commit()}"

    def get_filename(name, extension) -> pathlib.Path:
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        return (
            output_dir
            / f"{safe_name}_{safe_device_name}_{get_git_commit()}.{extension}"
        )

    wandb_logs = {}
    for benchmark_name, cls in benchmarks_to_run.items():
        logging.info(f"Running benchmark: {benchmark_name}")
        result = cls.run_benchmark(iters=iters)
        wandb_logs.update(
            {
                f"{benchmark_name}/{log_name}": value
                for log_name, value in result.get_logs(max_depth=1).items()
            }
        )
        png_filename = get_filename(benchmark_name, "png")
        logging.info(f"Saving result image to {png_filename}")
        result.to_png(png_filename, label=get_label(benchmark_name))
        result_data = json.dumps(dataclasses.asdict(result), indent=2)
        logging.info(f"Result: {result_data}")
        with open(get_filename(benchmark_name, "json"), "w") as f:
            logging.info(f"Saving result json to {f.name}")
            f.write(result_data)
        if child is not None:
            child_name = f"{benchmark_name}.{child}"
            child_label = get_label(child_name)
            logging.info(f"Generating benchmark result for child timer: {child_label}")
            png_filename = get_filename(child_name, "png")
            logging.info(f"Saving child result image to {png_filename}")
            result.to_png(png_filename, label=child_label, child=child)

    if wandb_project is not None:
        entity, project = wandb_project.split("/")
        wandb = WandB.get_instance()
        wandb.configure(log_to_wandb=True)
        wandb_name = f"{get_git_commit()}-{safe_device_name}"
        wandb.init(
            resumable=False,
            project=project,
            entity=entity,
            name=wandb_name,
        )
        wandb.log(wandb_logs, commit=True)
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
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=(
            "Weights & Biases entity and project, in the format <entity>/<project>. "
            "By default, logging to wandb is disabled."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = RESULTS_PATH

    with Distributed.context():
        return_code = main(
            benchmark_name=args.name,
            iters=args.iters,
            child=args.child,
            output_dir=output_dir,
            wandb_project=args.wandb_project,
        )
    sys.exit(return_code)
