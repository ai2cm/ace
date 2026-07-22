"""Fetch training/climate/spectral metrics for every run of every wandb project.

For each project listed under ``wandb_projects`` in ``projects.yaml``, pull the
metric histories of all finished runs and cache them to
``<out-dir>/<project>.pkl`` (default ``pickles/<project>.pkl``). Each pickle
holds ``list[dict]`` with keys ``run_id``, ``run_name``, ``metrics``.

Metric fetch mirrors
``explore2/alexeyy/masking/2026-07-06-varmasking-training-metrics-v8-v2.ipynb``:
  * CLIMATE/SPECTRAL metrics are plotly figure refs living in ``run.summary``
    (only the latest ref survives); their figure JSON is downloaded and cached.
  * TRAIN metrics are plain scalars in the history stream, each fetched on its
    own cadence (folding them into shared keys would intersect staggered
    cadences away via run.history's inner join).

Backfill logic (same as the notebook): the pickle only ever holds finished
runs. Each invocation is additive -- it fetches runs that finished since the
last save and backfills any metric newly added to ``METRICS`` for runs already
cached; nothing already cached is re-fetched. Non-finished legacy entries are
pruned from disk.

Usage:
    python fetch_wandb_metrics.py [--projects PATH] [--entity ENTITY] \
        [--out-dir DIR] [--cache-dir DIR] [--workers N] [--force]
"""

import argparse
import json
import math
import os
import pathlib
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import wandb
import yaml
from tqdm.auto import tqdm

# Transient network faults (ReadTimeout, connection resets) are retried with
# exponential backoff before a run is given up on.
FETCH_RETRIES = 4
FETCH_BACKOFF_S = 2.0


def _with_retries(fn, label):
    """Run fn(), retrying transient errors with exponential backoff."""
    for attempt in range(FETCH_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- retry any transient fault
            if attempt == FETCH_RETRIES - 1:
                raise
            wait = FETCH_BACKOFF_S * (2**attempt)
            print(f"  retry {label} in {wait:.0f}s (attempt {attempt + 1}): {exc}")
            time.sleep(wait)


HERE = pathlib.Path(__file__).parent
DEFAULT_PROJECTS = HERE / "projects.yaml"
DEFAULT_OUT = HERE / "pickles"
DEFAULT_CACHE = HERE / "wandb_plotly_cache"
DEFAULT_ENTITY = "ai2cm"

TRAIN_METRICS = [
    "weather_2024/day_5_norm/weighted_rmse/channel_mean",
    "long_46year/time_mean_norm/rmse/channel_mean",
    "train/mean/loss",
    "val/mean/loss",
]

CLIMATE_METRICS = [
    "long_46year/annual/surface_temperature",
    "long_46year/annual/air_temperature_0",
    "long_46year/annual/air_temperature_7",
    "long_46year/annual/specific_total_water_3",
    "long_46year/annual/PRATEsfc",
    "long_46year/annual/total_water_path",
]

# Same variables as CLIMATE_METRICS, wavenumber spectrum instead of annual mean.
SPECTRAL_METRICS = [
    "long_46year/power_spectrum/surface_temperature",
    "long_46year/power_spectrum/air_temperature_0",
    "long_46year/power_spectrum/air_temperature_7",
    "long_46year/power_spectrum/specific_total_water_3",
    "long_46year/power_spectrum/PRATEsfc",
    "long_46year/power_spectrum/total_water_path",
]

METRICS = TRAIN_METRICS + CLIMATE_METRICS + SPECTRAL_METRICS

# Annual-mean and power-spectrum diagnostics are logged as plotly figure refs
# (summary-only); everything else is a plain scalar in the history stream.
PLOTLY_METRICS = set(CLIMATE_METRICS) | set(SPECTRAL_METRICS)
# TRAIN_METRICS log on their own cadence, so they're fetched individually;
# folding them into the shared keys would intersect staggered cadences away
# (the classic run.history inner-join collapse).
SEPARATE_METRICS = set(TRAIN_METRICS)


def _fetch_plotly_fig(run, ref, cache_dir):
    path = ref.get("path")
    sha256 = ref.get("sha256", "")
    name = (sha256[:20] if sha256 else os.path.basename(path)) + ".json"
    cached = os.path.join(cache_dir, name)
    if not os.path.exists(cached):
        os.makedirs(cache_dir, exist_ok=True)
        dl_dir = os.path.join(cache_dir, "_dl", run.id)
        os.makedirs(dl_dir, exist_ok=True)
        run.file(path).download(root=dl_dir, replace=True)
        shutil.copy(os.path.join(dl_dir, path), cached)
    with open(cached) as f:
        return json.load(f)


def get_run_metrics(run, metric_keys, cache_dir):
    buckets: dict = {m: ([], [], []) for m in metric_keys}

    # Plotly figs live in run.summary (the latest logged ref), never fetched
    # from the history stream.
    for m in PLOTLY_METRICS & set(metric_keys):
        ref = run.summary.get(m)
        if hasattr(ref, "get") and ref.get("path"):
            buckets[m] = ([-1], [None], [_fetch_plotly_fig(run, ref, cache_dir)])

    # History metrics. PLOTLY_METRICS are excluded UNCONDITIONALLY (summary-only
    # by design) -- leaking any of them into the history fetch needs a key that
    # never exists there, which empties run.history's inner join to 0 rows and
    # silently drops real metrics like val/mean/loss.
    history_keys = [m for m in metric_keys if m not in PLOTLY_METRICS]
    shared = [m for m in history_keys if m not in SEPARATE_METRICS]
    groups = ([shared] if shared else []) + [
        [m] for m in history_keys if m in SEPARATE_METRICS
    ]

    for group in groups:
        hist = run.history(keys=["epoch"] + group, samples=100000)
        for _, row in hist.iterrows():
            for m in group:
                val = row.get(m)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                buckets[m][0].append(row.get("_step"))
                buckets[m][1].append(row.get("epoch"))
                buckets[m][2].append(val)

    # Runs without an "epoch" column (checkpoint-eval runs) fall back to a
    # 1..N index.
    for m in metric_keys:
        if buckets[m][1] and buckets[m][1][0] is None:
            buckets[m] = (
                buckets[m][0],
                list(range(1, len(buckets[m][2]) + 1)),
                buckets[m][2],
            )

    return buckets


def _load_projects(path):
    entries = yaml.safe_load(path.read_text())["wandb_projects"]
    names = []
    for entry in entries:
        names.append(entry if isinstance(entry, str) else entry["name"])
    return names


def _fetch_project(project, entity, out_dir, cache_dir, workers, force, timeout):
    pickle_path = out_dir / f"{project}.pkl"
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(f"{entity}/{project}"))
    id_to_run = {r.id: r for r in runs}
    print(f"{project}: {len(runs)} run(s) in wandb")

    # Pickle only ever holds finished runs. Load and (unless --force) treat
    # cached entries as done.
    existing = {}
    if pickle_path.exists() and not force:
        with open(pickle_path, "rb") as fh:
            existing = {e["run_id"]: e for e in pickle.load(fh)}
        print(f"  loaded {len(existing)} cached run(s) from {pickle_path}")

    # Drop any cached entry whose current live state isn't "finished" so a
    # crashed/running run doesn't get backfilled on a finished namesake's behalf.
    _stale = [
        rid for rid, r in id_to_run.items() if rid in existing and r.state != "finished"
    ]
    for rid in _stale:
        del existing[rid]
    if _stale:
        print(f"  pruned {len(_stale)} non-finished cached entry(ies)")

    new_run_ids = [r.id for r in runs if r.state == "finished" and r.id not in existing]
    stale_metric_ids = [
        rid
        for rid, e in existing.items()
        if rid in id_to_run and any(m not in e["metrics"] for m in METRICS)
    ]
    print(f"  {len(new_run_ids)} newly finished run(s) to fetch")
    print(f"  {len(stale_metric_ids)} cached run(s) need metric(s) backfilled")

    def _fetch_new_run(rid):
        run = id_to_run[rid]
        try:
            metrics = _with_retries(
                lambda: get_run_metrics(run, METRICS, cache_dir), run.name
            )
        except Exception as exc:  # one bad run must not abort the whole pickle
            print(f"  warn: {run.name}: fetch failed: {exc}")
            return None
        return {"run_id": run.id, "run_name": run.name, "metrics": metrics}

    def _fetch_missing_metrics(rid):
        run = id_to_run[rid]
        missing = [m for m in METRICS if m not in existing[rid]["metrics"]]
        try:
            return rid, _with_retries(
                lambda: get_run_metrics(run, missing, cache_dir), run.name
            )
        except Exception as exc:
            print(f"  warn: {run.name}: backfill failed: {exc}")
            return rid, None

    if new_run_ids or stale_metric_ids:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for entry in tqdm(
                ex.map(_fetch_new_run, new_run_ids),
                total=len(new_run_ids),
                desc=f"{project}: new runs",
            ):
                if entry is not None:
                    existing[entry["run_id"]] = entry
            for rid, new_metrics in tqdm(
                ex.map(_fetch_missing_metrics, stale_metric_ids),
                total=len(stale_metric_ids),
                desc=f"{project}: backfill",
            ):
                if new_metrics is not None:
                    existing[rid]["metrics"].update(new_metrics)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(list(existing.values()), f)
        print(f"  saved {len(existing)} run(s) to {pickle_path}")
    else:
        print("  nothing to do")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects",
        type=pathlib.Path,
        default=DEFAULT_PROJECTS,
        help=f"YAML file listing wandb_projects (default: {DEFAULT_PROJECTS}).",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"wandb entity (default: {DEFAULT_ENTITY}).",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=f"Directory for <project>.pkl files (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=DEFAULT_CACHE,
        help=f"Directory for cached plotly figure JSON (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread pool size for concurrent run fetches (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request wandb API HTTP timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached pickles and re-fetch every finished run.",
    )
    args = parser.parse_args()

    projects = _load_projects(args.projects)
    for project in projects:
        _fetch_project(
            project,
            args.entity,
            args.out_dir,
            str(args.cache_dir),
            args.workers,
            args.force,
            args.timeout,
        )


if __name__ == "__main__":
    main()
