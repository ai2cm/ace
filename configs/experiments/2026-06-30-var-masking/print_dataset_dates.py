"""Print start/end dates of datasets and inference windows for a train config.

Reads every dataset referenced in the config (train_loader, validation,
inference) from the /climate-default mount, and computes the start/end date of
each inference option from its start_indices + n_forward_steps.

Usage:
    python print_dataset_dates.py [CONFIG.yaml]
"""

import os
import sys

import pandas as pd
import xarray as xr
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(
    HERE,
    "run_configs",
    "ace-train-config-4deg-nc-sfno-c96-mask5-co2default.yaml",
)
# Config data_paths point at the /climate-default mount; the datasets actually
# live in this GCS bucket, keyed by the same basename.
GCS_BUCKET = "gs://vcm-ml-intermediate"
FMT = "%Y-%m-%d"

_time_cache: dict[str, xr.DataArray] = {}


def zarr_path(data_path: str, file_pattern: str) -> str:
    return f"{GCS_BUCKET}/{os.path.basename(data_path)}/{file_pattern}"


def load_time(path: str) -> xr.DataArray:
    """Return the (cached) time coordinate of a zarr store."""
    if path not in _time_cache:
        ds = xr.open_zarr(path, decode_timedelta=True)
        _time_cache[path] = ds["time"].load()
    return _time_cache[path]


def fmt(t) -> str:
    return pd.Timestamp(t).strftime(FMT)


def step_days(time: xr.DataArray) -> float:
    return (pd.Timestamp(time.values[1]) - pd.Timestamp(time.values[0])).days


def collect_datasets(cfg: dict) -> list[dict]:
    """Every (data_path, file_pattern, subset) referenced in the config."""
    rows = []

    def add(entry: dict, role: str):
        rows.append(
            {
                "role": role,
                "data_path": entry["data_path"],
                "file_pattern": entry["file_pattern"],
                "subset_start": entry.get("subset", {}).get("start_time"),
                "subset_stop": entry.get("subset", {}).get("stop_time"),
            }
        )

    for entry in cfg.get("inference", []):
        add(entry["loader"]["dataset"], f"inference:{entry['name']}")

    for role in ("train_loader", "validation"):
        node = cfg.get(role, {})
        loader = node.get("loader", node)
        dataset = loader.get("dataset", {})
        if "concat" in dataset:
            for i, sub in enumerate(dataset["concat"]):
                add(sub, f"{role}:concat[{i}]")
        elif dataset:
            add(dataset, role)
    return rows


def dataset_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        path = zarr_path(r["data_path"], r["file_pattern"])
        try:
            time = load_time(path)
            ds_start, ds_end = fmt(time.values[0]), fmt(time.values[-1])
            n = time.size
        except Exception as e:  # noqa: BLE001 - report unreadable stores inline
            ds_start = ds_end = f"<error: {type(e).__name__}>"
            n = None
        out.append(
            {
                "role": r["role"],
                "dataset": os.path.basename(r["data_path"]),
                "file": r["file_pattern"],
                "n_steps": n,
                "data_start": ds_start,
                "data_end": ds_end,
                "subset_start": r["subset_start"] or "",
                "subset_stop": r["subset_stop"] or "",
            }
        )
    return pd.DataFrame(out)


def inference_table(cfg: dict) -> pd.DataFrame:
    out = []
    for entry in cfg.get("inference", []):
        loader = entry["loader"]
        dataset = loader["dataset"]
        times = sorted(loader["start_indices"]["times"])
        first_ic = pd.Timestamp(times[0])
        last_ic = pd.Timestamp(times[-1])
        nfs = entry["n_forward_steps"]
        path = zarr_path(dataset["data_path"], dataset["file_pattern"])
        try:
            time = load_time(path)
            dt = step_days(time)
            data_end = pd.Timestamp(time.values[-1])
            window_end = last_ic + pd.Timedelta(days=nfs * dt)
            fits = "yes" if window_end <= data_end else "NO (overshoot)"
            window_end_s = window_end.strftime(FMT)
            data_end_s = data_end.strftime(FMT)
        except Exception as e:  # noqa: BLE001
            fits = f"<error: {type(e).__name__}>"
            window_end_s = data_end_s = ""
        out.append(
            {
                "name": entry["name"],
                "dataset": os.path.basename(dataset["data_path"]),
                "n_forward_steps": nfs,
                "first_ic": first_ic.strftime(FMT),
                "last_ic": last_ic.strftime(FMT),
                "window_end": window_end_s,
                "data_end": data_end_s,
                "fits": fits,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"Config: {config_path}\n")
    print("=== Datasets ===")
    print(dataset_table(collect_datasets(cfg)).to_string(index=False))
    print("\n=== Inference windows ===")
    print(inference_table(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
