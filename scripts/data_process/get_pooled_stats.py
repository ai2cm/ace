# Computes pooled normalization statistics over N datasets defined in a single
# config file (see `stats_config.yaml`). Each entry is a zarr store plus a time
# slice; the same store may appear multiple times with different slices. Stats
# are pooled across all entries weighted by number of timesteps, using the same
# variance-combination math as `combine_stats.py`. The pooled stats are written
# at the output root as:
#   centering.nc            (pooled per-variable mean)
#   scaling-full-field.nc   (pooled per-variable std of the full field)
#   scaling-residual.nc     (pooled per-variable std of the time-difference)
# and each individual dataset/slice's own stats (the same three files) are written
# to a per-pair subdirectory, mirroring the get_stats.py + combine_stats.py layout.
#
# The dependencies of this script are installed in the "fv3net" conda
# environment (same as get_stats.py / combine_stats.py).
#
# Usage:
#   # launch a CPU beaker job that reads the data and writes the stats:
#   python get_pooled_stats.py submit stats_config.yaml gs://bucket/out-dir
#   # or run the calculation directly (needs data access + deps):
#   python get_pooled_stats.py compute stats_config.yaml gs://bucket/out-dir

import argparse
import dataclasses
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import List, Optional

import dacite
import fsspec
import xarray as xr
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_REL = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

# Beaker / gantry defaults, mirroring configs/experiments/.../run-ace-train.sh.
DEFAULT_WORKSPACE = "ai2/ace"
# CPU-only stats job; runs on the same cluster/priority as copy_zarr_to_weka.sh.
DEFAULT_CLUSTERS = ["ai2/phobos"]
DEFAULT_PRIORITY = "high"
DEFAULT_BUDGET = "ai2/ace"
DEFAULT_WEKA = "climate-default:/climate-default"
GCP_CREDS = "/tmp/google_application_credentials.json"

# Auxiliary variables that exist for masking / vertical integrals but are not ML
# inputs/outputs, so they need no normalization constants. Mirrors get_stats.py.
DROP_VARIABLES = (
    [
        "land_sea_mask",
        "mask_HI",
        "mask_sea_ice_volume",
        "mask_sea_ice_fraction",
        "mask_ocean_sea_ice_fraction",
    ]
    + [f"pressure_thickness_of_atmospheric_layer_{i}" for i in range(8)]
    + [f"ak_{i}" for i in range(9)]
    + [f"bk_{i}" for i in range(9)]
    + [f"idepth_{i}" for i in range(19)]
    + [f"mask_{i}" for i in range(19)]
)

# Candidate horizontal dimension names; stats reduce over time + whichever of
# these are present, leaving any vertical/level dim intact.
HORIZONTAL_DIMS = [
    "latitude",
    "longitude",
    "lat",
    "lon",
    "grid_xt",
    "grid_yt",
    "x",
    "y",
]

# Files marking a zarr store root (v3 / v2 respectively); used to discover stores
# nested under a dataset directory.
ZARR_ROOT_MARKERS = ("zarr.json", ".zmetadata")


def _resolve_zarr_store(dataset: str) -> str:
    """Resolve a config `dataset` entry to a single zarr store path.

    The entry may point directly at a store, or at a directory that contains one
    store somewhere beneath it (e.g. `.../era5-.../` holding `....zarr`). In the
    latter case we recursively search for the store root, raising if zero or more
    than one is found so the choice is never silent.
    """
    dataset = dataset.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(dataset)

    def _is_store(path: str) -> bool:
        return any(fs.exists(f"{path}/{m}") for m in ZARR_ROOT_MARKERS)

    if _is_store(dataset):
        return dataset

    # Recursively find store roots: every path holding a root marker whose parent
    # is not itself part of a store (deduping the per-array markers a store has).
    roots = set()
    for marker in ZARR_ROOT_MARKERS:
        for hit in fs.glob(f"{dataset}/**/{marker}"):
            roots.add(hit[: -(len(marker) + 1)])
    store_roots = sorted(r for r in roots if not _is_store(r.rsplit("/", 1)[0]))

    if len(store_roots) == 0:
        raise ValueError(
            f"No zarr store found under {dataset!r}; expected a directory with "
            f"one of {ZARR_ROOT_MARKERS} somewhere beneath it."
        )
    if len(store_roots) > 1:
        resolved = [fs.unstrip_protocol(r) for r in store_roots]
        raise ValueError(
            f"Found multiple zarr stores under {dataset!r}: {resolved}. Point the "
            f"config `dataset` at a single store."
        )
    return fs.unstrip_protocol(store_roots[0])


@dataclasses.dataclass
class DatasetPair:
    dataset: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    # tolerate the alternate `stop_time` key seen in some configs
    stop_time: Optional[str] = None

    @property
    def end(self) -> Optional[str]:
        if self.end_time is not None and self.stop_time is not None:
            raise ValueError(
                f"{self.dataset}: set only one of end_time/stop_time, not both."
            )
        return self.end_time if self.end_time is not None else self.stop_time


@dataclasses.dataclass
class Config:
    dataset_pairs: List[DatasetPair]


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files."""
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


def _reduction_dims(ds: xr.Dataset) -> List[str]:
    dims = [d for d in HORIZONTAL_DIMS if d in ds.dims]
    if not dims:
        raise ValueError(
            f"No horizontal dims found in {list(ds.dims)}; expected some of "
            f"{HORIZONTAL_DIMS}."
        )
    return ["time"] + dims


def compute_pair_stats(pair: DatasetPair) -> dict:
    """Open one dataset, slice in time, and return its centering / full-field /
    residual stats plus the timestep count used as the pooling weight."""
    store = _resolve_zarr_store(pair.dataset)
    if store != pair.dataset.rstrip("/"):
        logging.info(f"Resolved {pair.dataset} -> {store}")
    try:
        import dask

        with dask.config.set({"array.chunk-size": "128MiB"}):
            ds = xr.open_zarr(store, chunks={"time": "auto"})
    except ImportError as e:
        logging.warning(f"Could not import dask ({e}), chunking is disabled.")
        ds = xr.open_zarr(store)

    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(pair.start_time, pair.end))

    n_samples = len(ds.time)
    if n_samples < 2:
        raise ValueError(
            f"{pair.dataset} sliced to [{pair.start_time}, {pair.end}] has "
            f"{n_samples} timesteps; check the time bounds in the config."
        )

    dims = _reduction_dims(ds)
    logging.info(
        f"{pair.dataset} [{pair.start_time}, {pair.end}]: {n_samples} steps, "
        f"reducing over {dims}"
    )
    return {
        "store": store,
        "centering": ds.mean(dim=dims).compute(),
        "scaling_full_field": ds.std(dim=dims).compute(),
        "scaling_residual": ds.diff("time").std(dim=dims).compute(),
        "n_samples": n_samples,
    }


def _stats_files(stats: dict) -> dict:
    """Map a per-pair / pooled stats dict to its {filename: DataArray} layout."""
    return {
        "centering.nc": stats["centering"],
        "scaling-full-field.nc": stats["scaling_full_field"],
        "scaling-residual.nc": stats["scaling_residual"],
    }


def _pair_subdir(index: int, pair: DatasetPair, store: str) -> str:
    """Per-pair output subdirectory name. The leading index keeps it unique even
    when the same store appears with overlapping/identical slices."""
    name = store.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".zarr"):
        name = name[: -len(".zarr")]
    start = pair.start_time or "min"
    end = pair.end or "max"
    return f"{index:02d}_{name}_{start}_{end}"


def pool_stats(per_pair: List[dict]) -> dict:
    """Pool per-pair stats weighted by timestep count.

    - centering: weighted mean of the means.
    - scaling-residual: sqrt of weighted-mean variance.
    - scaling-full-field: weighted-mean of total variance, where total variance
      adds the spread of per-pair means about the pooled mean (between-dataset
      variance) to each pair's within-dataset variance. This is what widens the
      std of near-constant forcings (e.g. global_mean_co2) to span the union.
    """
    samples = xr.DataArray([float(p["n_samples"]) for p in per_pair], dims=["run"])

    centering = xr.concat([p["centering"] for p in per_pair], dim="run")
    pooled_centering = centering.weighted(samples).mean(dim="run")

    residual = xr.concat([p["scaling_residual"] for p in per_pair], dim="run")
    pooled_residual = (residual**2).weighted(samples).mean(dim="run") ** 0.5

    full_field = xr.concat([p["scaling_full_field"] for p in per_pair], dim="run")
    between = (centering - pooled_centering) ** 2
    total_variance = between + full_field**2
    pooled_full_field = total_variance.weighted(samples).mean(dim="run") ** 0.5

    return {
        "centering.nc": pooled_centering,
        "scaling-full-field.nc": pooled_full_field,
        "scaling-residual.nc": pooled_residual,
    }


def write_stats(stats: dict, out_dir: str, config_yaml: str):
    history = f"Created by scripts/data_process/get_pooled_stats.py from {config_yaml}."
    if out_dir.endswith("/"):
        out_dir = out_dir[:-1]

    if out_dir.startswith("gs:"):
        tmp = tempfile.TemporaryDirectory()
        local_dir, remote_dir = tmp.name, out_dir
    else:
        os.makedirs(out_dir, exist_ok=True)
        local_dir, remote_dir = out_dir, None

    for filename, da in stats.items():
        da.attrs["history"] = history
        local_path = os.path.join(local_dir, filename)
        da.to_netcdf(local_path)
        if remote_dir is not None:
            copy(local_path, remote_dir + "/" + filename)
        logging.info(f"Wrote {(remote_dir or local_dir)}/{filename}")


def compute(config_yaml: str, output_directory: str):
    """Compute the pooled stats, log them, and write them. Runs in the beaker
    container, or locally if you have the data access and dependencies."""
    logging.basicConfig(level=logging.INFO)

    with open(config_yaml, "r") as f:
        config_data = yaml.safe_load(f)
    config = dacite.from_dict(data_class=Config, data=config_data)

    if output_directory.endswith("/"):
        output_directory = output_directory[:-1]

    per_pair = []
    for index, pair in enumerate(config.dataset_pairs):
        stats = compute_pair_stats(pair)
        per_pair.append(stats)
        subdir = _pair_subdir(index, pair, stats["store"])
        write_stats(_stats_files(stats), output_directory + "/" + subdir, config_yaml)

    pooled = pool_stats(per_pair)

    for name, da in pooled.items():
        logging.info(f"{name}:\n{da}")

    write_stats(pooled, output_directory, config_yaml)


def submit(args):
    """Launch this script as a CPU beaker job via gantry. The container reads
    the gs:// / weka datasets and writes the pooled stats to output_directory."""
    config_rel = pathlib.Path(args.config_yaml).resolve().relative_to(REPO_ROOT)
    image = (REPO_ROOT / "latest_deps_only_image.txt").read_text().strip()

    cmd = [
        "gantry",
        "run",
        "--name",
        args.name,
        "--task-name",
        args.name,
        "--description",
        "Compute pooled normalization stats over N datasets",
        "--beaker-image",
        image,
        "--workspace",
        args.beaker_workspace,
        "--priority",
        args.beaker_priority,
        "--preemptible",
        "--budget",
        args.budget,
        "--gpus",
        "0",
        "--shared-memory",
        "64GiB",
        "--weka",
        args.weka,
        "--env",
        f"GOOGLE_APPLICATION_CREDENTIALS={GCP_CREDS}",
        "--dataset-secret",
        f"google-credentials:{GCP_CREDS}",
        "--system-python",
        "--install",
        "pip install 'dask[array]' && pip install --no-deps .",
        "--allow-dirty",
    ]
    for cluster in args.beaker_cluster:
        cmd += ["--cluster", cluster]
    cmd += [
        "--",
        "python",
        str(SCRIPT_REL),
        "compute",
        str(config_rel),
        args.output_directory,
    ]

    print("Submitting:", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled normalization stats over all datasets in a "
            "dataset_pairs config. Use 'submit' to launch a beaker job, or "
            "'compute' to run the calculation directly."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_compute = sub.add_parser("compute", help="Run the calculation in-process.")
    p_compute.add_argument("config_yaml", help="Path to the dataset_pairs config.")
    p_compute.add_argument(
        "output_directory", help="Where to write the stats (local path or gs://)."
    )

    p_submit = sub.add_parser("submit", help="Launch a CPU beaker job via gantry.")
    p_submit.add_argument("config_yaml", help="Path to the dataset_pairs config.")
    p_submit.add_argument(
        "output_directory", help="Where to write the stats (should be gs://)."
    )
    p_submit.add_argument("--name", default="pooled-stats", help="Beaker job name.")
    p_submit.add_argument("--beaker-workspace", default=DEFAULT_WORKSPACE)
    p_submit.add_argument(
        "--beaker-cluster", nargs="+", default=DEFAULT_CLUSTERS, metavar="CLUSTER"
    )
    p_submit.add_argument("--beaker-priority", default=DEFAULT_PRIORITY)
    p_submit.add_argument("--budget", default=DEFAULT_BUDGET)
    p_submit.add_argument("--weka", default=DEFAULT_WEKA)
    p_submit.add_argument(
        "--dry-run", action="store_true", help="Print the gantry command only."
    )

    args = parser.parse_args()
    if args.mode == "compute":
        compute(args.config_yaml, args.output_directory)
    else:
        submit(args)


if __name__ == "__main__":
    main()
