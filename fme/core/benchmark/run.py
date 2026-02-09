import argparse

import torch

from fme.core.benchmark.benchmark import get_benchmarks, run_benchmark


def main(names: list[str] | None, iters: int):
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

    for name, fn in benchmarks_to_run.items():
        print(f"Running benchmark: {name}")
        result = run_benchmark(fn, iters=iters)
        print(f"  Result: {result}")


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
        "--iters",
        type=int,
        default=10,
        help="Number of iterations to run each benchmark for.",
    )
    args = parser.parse_args()

    main(names=[args.benchmark] if args.benchmark else None, iters=args.iters)
