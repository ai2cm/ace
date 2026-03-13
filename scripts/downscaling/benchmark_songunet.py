#!/usr/bin/env python
"""Benchmark SongUNet v2 vs v3 variants and generate comparison plots.

Requires CUDA. Uses the existing fme benchmark infrastructure.

Usage:
    python scripts/downscaling/benchmark_songunet.py \
        [--output-dir DIR] [--iters N] [--warmup N]
"""

import argparse
import json
import pathlib
import subprocess

import matplotlib.pyplot as plt
import torch

from fme.core.benchmark.benchmark import BenchmarkResult, get_benchmarks


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
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
        return commit
    except Exception:
        return "unknown"


def get_device_name() -> str:
    return torch.cuda.get_device_properties(0).name


def is_apex_available() -> bool:
    try:
        from fme.downscaling.modules.physicsnemo_unets_v2.group_norm import (
            apex_available,
        )

        return bool(apex_available)
    except Exception:
        return False


def _json_default(obj):
    """json.dumps default hook that converts tensors to Python objects."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def plot_comparison(
    results: dict[str, BenchmarkResult],
    title: str,
    path: str | pathlib.Path,
    device_name: str,
    git_commit: str,
    compile_times: dict[str, float | None] | None = None,
):
    """Plot side-by-side stacked bars for multiple benchmark results.

    Uses blue family for v2 variants and orange family for v3 variants.
    Each bar is stacked by timer children (e.g. mapping, encoder, decoder).
    "Self" time (parent minus children) is shown in a lighter shade.
    """
    names = list(results.keys())
    if not names:
        return

    # Color families: blue for v2, orange for v3
    v2_base = (0.122, 0.467, 0.706)  # #1f77b4
    v3_base = (1.0, 0.498, 0.055)  # #ff7f0e

    def blend_with_white(rgb, amount):
        return tuple(c + (1.0 - c) * amount for c in rgb)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.6
    x_positions = list(range(len(names)))

    for i, name in enumerate(names):
        result = results[name]
        timer = result.timer
        root_count = timer.count

        base_rgb = v2_base if name.startswith("songunetv2") else v3_base
        children = list(timer.children.items())

        bottom = 0.0
        n_children = len(children)
        for j, (child_name, child_timer) in enumerate(children):
            child_avg = float(child_timer.count * child_timer.avg_time) / root_count
            # Vary shade per child: first child darkest, later lighter
            lighten = 0.1 + (0.4 * (j / max(n_children - 1, 1)))
            color = blend_with_white(base_rgb, lighten)
            ax.bar(
                i,
                child_avg,
                bottom=bottom,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=child_name if i == 0 else None,
            )
            if child_avg / timer.avg_time >= 0.05:
                ax.text(
                    i,
                    bottom + child_avg / 2,
                    f"{child_name}\n{child_avg:.1f}ms",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            bottom += child_avg

        # Self time in lighter shade
        self_time = max(timer.avg_time - bottom, 0.0)
        if self_time > 0:
            light_color = blend_with_white(base_rgb, 0.7)
            ax.bar(
                i,
                self_time,
                bottom=bottom,
                width=bar_width,
                color=light_color,
                edgecolor="white",
                linewidth=0.5,
            )

        # Total time label on top
        ax.text(
            i,
            timer.avg_time + timer.avg_time * 0.02,
            f"{timer.avg_time:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        # Compile time annotation
        if compile_times and compile_times.get(name) is not None:
            ct = compile_times[name]
            ax.annotate(
                f"compile: {ct:.1f}s",
                xy=(i, timer.avg_time),
                xytext=(i + 0.3, timer.avg_time * 1.15),
                fontsize=8,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Time per forward pass (ms)")
    ax.set_title(f"{title}\n{device_name} | commit {git_commit}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SongUNet v2 vs v3 variants" " and generate comparison plots."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/songunet_benchmarks/",
        help=(
            "Directory for output PNGs and JSON"
            " (default: results/songunet_benchmarks/)"
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        default=False,
        help="Show all labels on benchmark PNGs (passed through to to_png)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark script.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iters = args.iters
    warmup = args.warmup
    all_labels = args.all_labels

    device_name = get_device_name()
    git_commit = get_git_commit()
    has_apex = is_apex_available()

    print(f"Device: {device_name}")
    print(f"Git commit: {git_commit}")
    print(f"Apex available: {has_apex}")
    print(f"Iterations: {iters}, Warmup: {warmup}")
    print(f"Output directory: {output_dir}")
    print()

    # Get all registered benchmarks and filter to songunet ones
    all_benchmarks = get_benchmarks()
    songunet_names = sorted(
        n
        for n in all_benchmarks
        if n.startswith("songunetv2") or n.startswith("songunetv3")
    )

    # Skip apex variants if apex is not available
    if not has_apex:
        skipped = [n for n in songunet_names if "apex" in n]
        if skipped:
            print(f"Skipping apex variants (apex not available): {skipped}")
        songunet_names = [n for n in songunet_names if "apex" not in n]

    print(f"Benchmarks to run: {songunet_names}")
    print()

    # Run benchmarks
    results: dict[str, BenchmarkResult] = {}
    compile_times: dict[str, float | None] = {}

    for name in songunet_names:
        cls = all_benchmarks[name]
        print(f"--- Running benchmark: {name} ---")

        result = cls.run_benchmark(iters=iters, warmup=warmup)
        results[name] = result

        # Extract compile time from diagnostics if present
        ct = result.diagnostics.get("compile_time_s", None)
        if ct is not None and isinstance(ct, torch.Tensor):
            ct = ct.item()
        compile_times[name] = ct

        # Print per-benchmark summary
        print(f"  avg_time: {result.timer.avg_time:.2f} ms")
        print(
            f"  memory: {result.memory.max_alloc / (1024 * 1024):.1f} MB alloc, "
            f"{result.memory.max_reserved / (1024 * 1024):.1f} MB reserved"
        )
        print(f"  cpu_time: {result.cpu_time:.2f} ms")
        if ct is not None:
            print(f"  compile_time: {ct:.2f} s")

        # Save individual PNG
        safe_name = name.replace("/", "_").replace(".", "_").lower()
        safe_device = device_name.replace(" ", "_").replace("/", "_").lower()
        png_path = output_dir / f"{safe_name}_{safe_device}_{git_commit}.png"
        label = f"{name} on {device_name} at {git_commit}"
        result.to_png(png_path, label=label, all_labels=all_labels)
        print(f"  Saved PNG: {png_path}")

        # Save JSON
        json_path = output_dir / f"{safe_name}_{safe_device}_{git_commit}.json"
        with open(json_path, "w") as f:
            json.dump(result.asdict(), f, indent=2, default=_json_default)
        print(f"  Saved JSON: {json_path}")
        print()

    # --- Summary table ---
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (
        f"{'Benchmark':<30} {'Avg Time (ms)':>15}"
        f" {'Memory (MB)':>15} {'CPU Time (ms)':>15}"
    )
    print(header)
    print("-" * 80)
    for name in songunet_names:
        if name not in results:
            continue
        r = results[name]
        mem_mb = r.memory.max_alloc / (1024 * 1024)
        line = (
            f"{name:<30} {r.timer.avg_time:>15.2f} {mem_mb:>15.1f} {r.cpu_time:>15.2f}"
        )
        ct = compile_times.get(name)
        if ct is not None:
            line += f"  (compile: {ct:.1f}s)"
        print(line)
    print("=" * 80)
    print()

    # --- Comparison Plots ---
    print("Generating comparison plots...")
    print()

    # Plot 1: v2 vs v3 baseline
    plot1_names = [n for n in ["songunetv2", "songunetv3"] if n in results]
    if len(plot1_names) == 2:
        plot1_path = output_dir / "comparison_v2_vs_v3_baseline.png"
        plot_comparison(
            {n: results[n] for n in plot1_names},
            title="SongUNet v2 vs v3 (baseline)",
            path=plot1_path,
            device_name=device_name,
            git_commit=git_commit,
        )
        print(f"Saved: {plot1_path}")
    else:
        print(f"Skipping plot 1 (need songunetv2 and songunetv3, have: {plot1_names})")

    # Plot 2: v2 vs v3 compiled
    plot2_names = [n for n in ["songunetv2", "songunetv3_compiled"] if n in results]
    if len(plot2_names) == 2:
        plot2_path = output_dir / "comparison_v2_vs_v3_compiled.png"
        plot_comparison(
            {n: results[n] for n in plot2_names},
            title="SongUNet v2 vs v3 (compiled)",
            path=plot2_path,
            device_name=device_name,
            git_commit=git_commit,
            compile_times=compile_times,
        )
        print(f"Saved: {plot2_path}")
    else:
        print(
            "Skipping plot 2 (need songunetv2 and songunetv3_compiled, "
            f"have: {plot2_names})"
        )

    # Plot 3: All non-apex variants
    plot3_names = [n for n in songunet_names if "apex" not in n and n in results]
    if len(plot3_names) >= 2:
        plot3_path = output_dir / "comparison_all_variants.png"
        plot_comparison(
            {n: results[n] for n in plot3_names},
            title="All SongUNet variants",
            path=plot3_path,
            device_name=device_name,
            git_commit=git_commit,
            compile_times=compile_times,
        )
        print(f"Saved: {plot3_path}")
    else:
        print(f"Skipping plot 3 (need >= 2 results, have: {plot3_names})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
