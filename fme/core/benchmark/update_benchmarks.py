import json
import logging
import os
import pathlib
import subprocess

import torch

from fme.core.benchmark.benchmark import BenchmarkResult, get_benchmarks

RESULTS_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "testdata"

_GIT_COMMIT: str | None = None


def get_git_commit() -> str:
    global _GIT_COMMIT
    if _GIT_COMMIT is None:
        args = ["git", "rev-parse", "--short", "HEAD"]
        _GIT_COMMIT = (
            subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        )
    return _GIT_COMMIT


def get_device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).name
    else:
        return "CPU"


def validate_and_update_benchmark_result(
    x: BenchmarkResult,
    filename_root: str,
    name: str,
    rtol: float,
    children_rtol: float,
) -> bool:
    updated: bool = False
    device_name = get_device_name().replace(" ", "_").replace("/", "_").lower()
    json_path = RESULTS_PATH / f"{filename_root}-{device_name}.json"
    png_path = RESULTS_PATH / f"{filename_root}-{device_name}.png"
    label = get_benchmark_label(name)
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump(x.asdict(), f, indent=4)
        x.to_png(png_path, label=label)
        updated = True
    else:
        with open(json_path) as f:
            d = json.load(f)
        y = BenchmarkResult.from_dict(d)
        try:
            x.assert_close(y, rtol=rtol, children_rtol=children_rtol)
        except AssertionError as e:
            logging.warning(
                f"Benchmark result for {name} on {device_name} at commit "
                f"{get_git_commit()} differ from regression result: {e}, "
                f"updating regression result."
            )
            with open(json_path, "w") as f:
                json.dump(x.asdict(), f, indent=4)
            x.to_png(png_path, label=label)
            updated = True
    return updated


def main() -> None:
    RESULTS_PATH.mkdir(exist_ok=True)
    device_name = get_device_name()

    logging.info(f"Running benchmarks on device: {device_name}")
    benchmarks = get_benchmarks()

    updated = False
    for name, cls in benchmarks.items():
        logging.info(f"Running benchmark: {name}")
        result = cls.run_benchmark(iters=10, warmup=2)
        updated = updated or validate_and_update_benchmark_result(
            result,
            os.path.join(RESULTS_PATH, name),
            name=name,
            rtol=0.02,
            children_rtol=0.05,
        )


def get_benchmark_label(name):
    device_name = get_device_name()
    return f"{name} on {device_name} at commit {get_git_commit()}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
