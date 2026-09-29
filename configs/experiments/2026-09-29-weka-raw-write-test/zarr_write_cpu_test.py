"""Write one benchmark window of ACE-layout zarr shards with the default and the
zarrs codec pipelines, reporting wall time and CPU seconds, alongside the CPU
quota the container was given. Separates codec-side cost from storage cost on
the same node the data writing benchmark runs on.
"""

import argparse
import os
import shutil
import time
from typing import Any

import numpy as np
import zarr

N_VARS, N_IC, N_TIMES, N_LAT, N_LON = 53, 2, 50, 180, 360
ZARRS: dict[str, Any] = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
PIPELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "zarrs": ZARRS,
    "zarrs-threads8": {**ZARRS, "threading.max_workers": 8},
    "zarrs-threads16": {**ZARRS, "threading.max_workers": 16},
    "zarrs-chunks8": {**ZARRS, "codec_pipeline.chunk_concurrent_maximum": 8},
    "zarrs-chunks16": {**ZARRS, "codec_pipeline.chunk_concurrent_maximum": 16},
}


def _cpu_quota() -> str:
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        if os.path.exists(path):
            with open(path) as f:
                return f"{path}={f.read().strip()}"
    return "no cgroup cpu limit file found"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _load_average() -> str:
    try:
        return " ".join(f"{x:.1f}" for x in os.getloadavg())
    except OSError:
        return "unknown"


def _affinity_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    return len(get_affinity(0)) if get_affinity is not None else None


def _synthetic_window(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.linspace(-1, 1, N_LAT)[None, None, :, None]
    lon = np.linspace(0, 2 * np.pi, N_LON)[None, None, None, :]
    t = np.arange(N_TIMES)[None, :, None, None]
    noise = 0.05 * rng.standard_normal((N_IC, N_TIMES, N_LAT, N_LON))
    base = (np.sin(lon + 0.1 * t) * np.cos(np.pi * lat / 2) + noise).astype("f4")
    return {f"v{i}": (base * (1 + 0.01 * i)).astype("f4") for i in range(N_VARS)}


def _write_window(
    root: str, label: str, config: dict[str, Any], data: dict[str, np.ndarray]
):
    path = os.path.join(root, f"{label}.zarr")
    shutil.rmtree(path, ignore_errors=True)
    with zarr.config.set(config):
        group = zarr.open_group(path, mode="w")
        for name in data:
            group.create_array(
                name=name,
                shape=(N_IC, 2 * N_TIMES, N_LAT, N_LON),
                chunks=(1, 1, N_LAT, N_LON),
                shards=(N_IC, N_TIMES, N_LAT, N_LON),
                dtype="f4",
            )
        group = zarr.open_group(path, mode="r+")
        wall0, cpu0 = time.perf_counter(), time.process_time()
        for name, array in data.items():
            group[name][:, 0:N_TIMES] = array
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    shutil.rmtree(path, ignore_errors=True)
    nbytes = sum(a.nbytes for a in data.values())
    print(
        f"label={label} root={root} window_mb={nbytes / 1e6:.0f} wall_s={wall:.2f} "
        f"mb_per_s={nbytes / wall / 1e6:.0f} cpu_s={cpu:.2f} "
        f"effective_cores={cpu / wall:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Directories to write into.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINE_CONFIGS),
        choices=PIPELINE_CONFIGS,
    )
    args = parser.parse_args()
    print(
        f"cpu_count={os.cpu_count()} affinity={_affinity_count()} "
        f"quota={_cpu_quota()} zarr={zarr.__version__}"
    )
    print(f"cpu_model={_cpu_model()!r} loadavg_1_5_15={_load_average()}")
    data = _synthetic_window()
    for root in args.roots:
        os.makedirs(root, exist_ok=True)
        for _ in range(args.repeats):
            for label in args.pipelines:
                _write_window(root, label, PIPELINE_CONFIGS[label], data)
    print(f"loadavg_1_5_15_at_end={_load_average()}")


if __name__ == "__main__":
    main()
