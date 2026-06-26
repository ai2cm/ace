"""Print a time-axis summary for each dataset in a mapping.yaml.

By default one row per dataset, aggregated over its `<run>.zarr` members.
With --verbose, one row per zarr.

Columns: name, run, start, end, n (points), dtype, dt (modal step), uniform.
In aggregate mode `run` is the run count; `dt`/`dtype`/`uniform` collapse to a
single value when all runs agree, else list the distinct values / "n".

Usage:
    python describe_mapping.py [mapping.yaml] [--verbose]
Requires GCS auth (gcsfs) for gs:// paths.
"""

import sys

import fsspec
import numpy as np
import xarray as xr
import yaml


def list_runs(dataset: str) -> list[tuple[str, str]]:
    """Return [(run_name, zarr_path), ...] for a dataset path.

    A path ending in `.zarr` is a single run; otherwise list `*.zarr` children.
    """
    dataset = dataset.rstrip("/")
    if dataset.endswith(".zarr"):
        name = dataset.rsplit("/", 1)[-1][: -len(".zarr")]
        return [(name, dataset)]

    fs, _ = fsspec.core.url_to_fs(dataset)
    proto = dataset.split("://", 1)[0]
    runs = []
    for entry in fs.ls(dataset, detail=False):
        entry = entry.rstrip("/")
        if entry.endswith(".zarr"):
            run = entry.rsplit("/", 1)[-1][: -len(".zarr")]
            # fs.ls drops the protocol; re-attach it for xr.open_zarr.
            runs.append((run, f"{proto}://{entry}"))
    return sorted(runs)


def summarize_time(path: str) -> dict:
    """Open only the time coord of a zarr and summarize its spacing."""
    ds = xr.open_zarr(path, chunks=None, decode_timedelta=False)
    t = ds["time"].values
    n = len(t)
    out = {
        "t0": None,
        "t1": None,
        "n": n,
        "dtype": str(t.dtype),
        "dt": None,
        "uniform": None,
    }
    if n == 0:
        return out
    out["t0"], out["t1"] = t[0], t[-1]
    if n > 1:
        vals, counts = np.unique(np.diff(t), return_counts=True)
        out["dt"] = vals[counts.argmax()]
        out["uniform"] = len(vals) == 1
    return out


def fmt_time(v) -> str:
    """Format a time value as `YYYY-MM-DD HH:MM:SS` (drop sub-second)."""
    if v is None:
        return "-"
    try:  # cftime objects and pandas Timestamps have strftime
        return v.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:  # numpy datetime64 does not
        import pandas as pd

        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S")


def fmt_dt(d) -> str:
    """Express a time step in hours, e.g. `24h`."""
    if d is None:
        return "-"
    try:  # numpy timedelta64
        hours = d / np.timedelta64(1, "h")
    except TypeError:  # python timedelta (from cftime diff)
        hours = d.total_seconds() / 3600
    return f"{hours:g}h"


def _collapse(values: list) -> str:
    """One value if all runs agree, else comma-joined distinct values."""
    distinct = list(dict.fromkeys(str(v) for v in values))
    return distinct[0] if len(distinct) == 1 else ",".join(distinct)


def aggregate(name: str, runs: list[dict]) -> dict:
    return {
        "name": name,
        "run": len(runs),
        "start": fmt_time(min(r["t0"] for r in runs)),
        "end": fmt_time(max(r["t1"] for r in runs)),
        "n": sum(r["n"] for r in runs),
        "dtype": _collapse([r["dtype"] for r in runs]),
        "dt": _collapse([fmt_dt(r["dt"]) for r in runs]),
        "uniform": "y"
        if all(r["uniform"] for r in runs) and len({str(r["dt"]) for r in runs}) == 1
        else "n",
    }


def print_table(rows: list[dict], cols: list[str]):
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(mapping_path: str, verbose: bool):
    with open(mapping_path) as f:
        mapping = yaml.safe_load(f)

    cols = ["name", "run", "start", "end", "n", "dtype", "dt", "uniform"]
    rows = []
    for pair in mapping["dataset_pairs"]:
        dataset = pair["dataset"]
        name = dataset.rstrip("/").rsplit("/", 1)[-1].replace(".zarr", "")
        run_infos = []
        for run, path in list_runs(dataset):
            try:
                run_infos.append({"run": run, **summarize_time(path)})
            except Exception as e:  # noqa: BLE001 - report, keep going
                run_infos.append(
                    {
                        "run": run,
                        "t0": None,
                        "t1": None,
                        "n": 0,
                        "dtype": f"ERROR: {e}",
                        "dt": None,
                        "uniform": False,
                    }
                )

        if verbose:
            for r in run_infos:
                rows.append(
                    {
                        "name": name,
                        "run": r["run"],
                        "start": str(r["t0"]),
                        "end": str(r["t1"]),
                        "n": r["n"],
                        "dtype": r["dtype"],
                        "dt": str(r["dt"]),
                        "uniform": "y" if r["uniform"] else "n",
                    }
                )
        else:
            rows.append(aggregate(name, run_infos))

    print_table(rows, cols)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--verbose"]
    verbose = "--verbose" in sys.argv[1:]
    main(args[0] if args else "mapping.yaml", verbose)
