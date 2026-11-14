# Computes pooled normalization statistics over N datasets defined in a single
# config file (see `configs/stats_config.yaml`). Each entry is a zarr store (or a
# directory of stores, e.g. an ensemble of `ic_XXXX.zarr` members) plus a time
# slice; the slice applies to every store the entry resolves to, and the same
# store may appear multiple times with different slices. Each store gets its own
# stats; stats are pooled across all of them weighted by number of timesteps,
# using the same
# variance-combination math as `combine_stats.py`. The pooled stats are written
# at the output root as:
#   centering.nc            (pooled per-variable mean)
#   scaling-full-field.nc   (pooled per-variable std of the full field)
#   scaling-residual.nc     (pooled per-variable std of the time-difference)
# and each individual dataset/slice's own stats (the same three files) are written
# to a per-pair subdirectory, mirroring the get_stats.py + combine_stats.py layout.
#
# Entries may also be tagged with a `group`, in which case each group named in
# the config's `groups` list is pooled separately from the same per-store
# moments and written to `groups/{name}/`. This gives every grouping strategy of
# an experiment from a single pass over the data.
#
# Entries may also rename, derive and rescale variables before their stats are
# computed, mirroring the same-named `XarrayDataConfig` options used at training
# time (`renamed_variables`, `derived_variables`, `overwrite.multiply_scalar`),
# so that every store contributes to the stats of the variables training sees.
#
# Each store is read once, in blocks of time steps reduced by a pool of worker
# processes (`--workers`), without dask.
#
# The dependencies of this script are those of fme (it runs in the fme image).
#
# The per-store stats files double as a checkpoint: a store whose three files
# are already in the output directory is loaded instead of recomputed, so a
# preempted job resumes where it stopped. `--shard I/N` computes only every
# N-th store (and skips the pooling), so N jobs can fill one output directory in
# parallel before a final run without `--shard` pools them.
#
# Usage:
#   # launch a CPU beaker job that reads the data and writes the stats:
#   python get_pooled_stats.py submit configs/stats_config.yaml gs://bucket/out-dir
#   # or run the calculation directly (needs data access + deps):
#   python get_pooled_stats.py compute configs/stats_config.yaml gs://bucket/out-dir

import argparse
import concurrent.futures
import dataclasses
import itertools
import logging
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterable

import dacite
import fsspec
import numpy as np
import torch
import xarray as xr
import yaml
from fs_utils import is_local, makedirs, path_exists

from fme.core.dataset.xarray import DERIVED_VARIABLES, _rename_variables

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_REL = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

# Beaker / gantry defaults, mirroring configs/experiments/.../run-ace-train.sh.
DEFAULT_WORKSPACE = "ai2/ace"
# CPU-only stats job; runs on the same cluster/priority as copy_zarr_to_weka.sh.
DEFAULT_CLUSTERS = ["ai2/phobos"]
DEFAULT_PRIORITY = "high"
DEFAULT_BUDGET = "ai2/atec-climate"
DEFAULT_WEKA = "climate-default:/climate-default"
GCP_CREDS = "/tmp/google_application_credentials.json"
# Worker processes reading the stores; each uses about one core and, for 1deg
# data, a few GiB of memory.
DEFAULT_WORKERS = 64

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


def _resolve_zarr_stores(dataset: str) -> list[str]:
    """Resolve a config `dataset` entry to the zarr store path(s) it contains.

    The entry may point directly at a store, or at a directory that contains one
    or more stores somewhere beneath it (e.g. an ensemble directory holding
    `ic_0001.zarr`, `ic_0002.zarr`, ...). In the latter case we recursively
    search for every store root and return them all sorted; each store is treated
    as its own dataset (own stats) and pooled with the rest. Raises if zero stores
    are found so an empty/typo'd path is never silent.
    """
    dataset = dataset.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(dataset)

    def _is_store(path: str) -> bool:
        return any(fs.exists(f"{path}/{m}") for m in ZARR_ROOT_MARKERS)

    if _is_store(dataset):
        return [dataset]

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
    return [fs.unstrip_protocol(r) for r in store_roots]


@dataclasses.dataclass
class DatasetPair:
    """One zarr store (or directory of stores) plus the time slice to use.

    Parameters:
        dataset: Path to a zarr store, or to a directory containing stores.
        start_time: Inclusive start of the time slice; None for the first step.
        end_time: Inclusive end of the time slice; None for the last step.
        group: Name of the group this entry's stores belong to, for `groups`
            pooling. A store may belong to several groups at once (e.g. a
            fine-grained "amip" group and a coarse "c96" group), in which case
            list them under `groups` instead.
        groups: Names of the groups this entry's stores belong to.
        renamed_variables: Mapping from variable names in the stores to the
            names to use instead, as `XarrayDataConfig.renamed_variables`.
        multiply_scalar: Scalars to multiply variables by, keyed by the names
            after renaming, as `XarrayDataConfig.overwrite.multiply_scalar`.
        derived_variables: Names of variables to compute from other variables,
            each a key of fme's `DERIVED_VARIABLES`, as
            `XarrayDataConfig.derived_variables`.

    The transforms are applied as at training time: variables are renamed,
    then derived, then scaled.
    """

    dataset: str
    start_time: str | None = None
    end_time: str | None = None
    group: str | None = None
    groups: list[str] | None = None
    renamed_variables: dict[str, str] = dataclasses.field(default_factory=dict)
    multiply_scalar: dict[str, float] = dataclasses.field(default_factory=dict)
    derived_variables: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.group is not None and self.groups is not None:
            raise ValueError(f"{self.dataset}: set only one of group/groups, not both.")
        new_names = list(self.renamed_variables.values())
        if len(new_names) != len(set(new_names)):
            raise ValueError(
                f"{self.dataset}: renamed_variables must not rename multiple "
                f"variables to the same name, got {self.renamed_variables}."
            )
        unknown = set(self.derived_variables) - set(DERIVED_VARIABLES)
        if unknown:
            raise ValueError(
                f"{self.dataset}: Unknown derived_variables {sorted(unknown)}, "
                f"must be one of {sorted(DERIVED_VARIABLES)}."
            )
        if len(self.derived_variables) != len(set(self.derived_variables)):
            raise ValueError(
                f"{self.dataset}: derived_variables has duplicates, got "
                f"{self.derived_variables}."
            )

    @property
    def time_slice(self) -> slice:
        return slice(self.start_time, self.end_time)

    @property
    def group_names(self) -> list[str]:
        if self.groups is not None:
            return list(self.groups)
        if self.group is not None:
            return [self.group]
        return []

    def output_subdir(self, index: int, store: str) -> str:
        """Per-pair output subdirectory name. The leading index keeps it unique
        even when the same store appears with overlapping/identical slices."""
        name = store.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            name = name[: -len(".zarr")]
        start = self.start_time or "min"
        end = self.end_time or "max"
        return f"{index:02d}_{name}_{start}_{end}"


@dataclasses.dataclass
class Config:
    """Stats configuration.

    Parameters:
        dataset_pairs: The stores to compute stats over, each with an optional
            time slice and group membership.
        groups: Names of the groups to additionally pool separately, written to
            `groups/{name}/` alongside the root pooled stats. Every name must be
            claimed by at least one dataset pair. Leave empty to write only the
            root pooled stats.
    """

    dataset_pairs: list[DatasetPair]
    groups: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if len(self.dataset_pairs) == 0:
            raise ValueError("dataset_pairs must list at least one dataset.")
        claimed = {name for pair in self.dataset_pairs for name in pair.group_names}
        missing = set(self.groups) - claimed
        if missing:
            raise ValueError(
                f"Groups {sorted(missing)} are not claimed by any dataset pair."
            )
        unused = claimed - set(self.groups)
        if unused:
            raise ValueError(
                f"Dataset pairs claim groups {sorted(unused)} which are not "
                "listed in `groups`; add them or remove the tag."
            )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # strict so a misspelled key (e.g. `stop_time`) raises instead of
        # silently leaving the time slice open-ended.
        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(strict=True)
        )


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files."""
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


def _reduction_dims(ds: xr.Dataset) -> list[str]:
    dims = [d for d in HORIZONTAL_DIMS if d in ds.dims]
    if not dims:
        raise ValueError(
            f"No horizontal dims found in {list(ds.dims)}; expected some of "
            f"{HORIZONTAL_DIMS}."
        )
    return ["time"] + dims


def _derive_variable(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Compute a derived variable lazily with fme's own `DERIVED_VARIABLES`
    function, applied blockwise to the (time-broadcast) input arrays."""
    derived = DERIVED_VARIABLES[name]
    # Static inputs (e.g. land_fraction) are broadcast over time, as at training.
    inputs = xr.broadcast(*[ds[input_name] for input_name in derived.inputs])
    dtype = np.result_type(*[da.dtype for da in inputs])

    def _compute(*arrays: np.ndarray) -> np.ndarray:
        tensors = {
            input_name: torch.as_tensor(np.ascontiguousarray(array))
            for input_name, array in zip(derived.inputs, arrays)
        }
        return derived.compute(tensors).numpy()

    return xr.apply_ufunc(_compute, *inputs, dask="parallelized", output_dtypes=[dtype])


def apply_transforms(ds: xr.Dataset, pair: DatasetPair) -> xr.Dataset:
    """Rename, derive and scale variables as `XarrayDataset` does at training
    time: renaming happens when files are opened, derived variables are computed
    from the loaded (renamed) variables, and `overwrite` is applied last."""
    ds = _rename_variables(ds, pair.renamed_variables)
    for name in pair.derived_variables:
        if name in ds.variables:
            raise ValueError(
                f"Cannot derive variable '{name}' which already exists in the "
                "dataset."
            )
        missing = [i for i in DERIVED_VARIABLES[name].inputs if i not in ds]
        if missing:
            raise ValueError(
                f"Inputs {missing} needed to derive variable '{name}' are not in "
                "the dataset."
            )
        ds = ds.assign({name: _derive_variable(ds, name)})
    scaled = {
        var: ds[var] * multiplier for var, multiplier in pair.multiply_scalar.items()
    }
    return ds.assign(scaled)


# Target number of timesteps read by one worker task, when the store does not
# say (through its shard size) how many are cheapest to read together.
TARGET_TIME_BLOCK = 256
# Smallest number of timesteps a task is split down to when a store would
# otherwise give the pool too few tasks to keep its workers busy.
MIN_TIME_BLOCK = 16
# Timesteps reduced at once within a task: small enough that the float64
# temporaries of a 1deg field stay a few MiB, which is much faster than
# reducing a whole block of them at once.
REDUCTION_STEPS = 8

# The store a worker process has open, kept between its tasks so that the
# store's metadata is parsed once per process rather than once per task.
_OPEN_STORE: dict[str, xr.Dataset] = {}

_Moments = tuple[np.ndarray, np.ndarray, np.ndarray]


def _init_worker():
    """Keep each worker process to about one core: the parallelism is across
    processes, so thread pools inside them only oversubscribe the node."""
    import zarr

    zarr.config.set({"threading.max_workers": 2})
    torch.set_num_threads(1)


def _open_store(store: str) -> xr.Dataset:
    if store not in _OPEN_STORE:
        _OPEN_STORE.clear()
        # chunks=None gives lazily indexed numpy arrays: a variable is read
        # only when its values are asked for, and no dask graph is involved.
        # cache=False so the values are dropped again once they are reduced.
        _OPEN_STORE[store] = xr.open_dataset(
            store, engine="zarr", chunks=None, cache=False
        )
    return _OPEN_STORE[store]


def _moments(x: np.ndarray, axes: tuple[int, ...]) -> _Moments:
    """Count, mean and sum of squared deviations from that mean of the non-NaN
    values of `x` over `axes`, in float64. The mean is 0 where the count is 0."""
    total = x.sum(axis=axes, keepdims=True)
    if np.isnan(total).any():
        valid = ~np.isnan(x)
        n = valid.sum(axis=axes, keepdims=True).astype(np.float64)
        total = np.where(valid, x, 0.0).sum(axis=axes, keepdims=True)
        mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
        deviation = np.where(valid, x - mean, 0.0)
    else:
        count = float(np.prod([x.shape[axis] for axis in axes]))
        n = np.full(total.shape, count)
        mean = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
        deviation = x - mean
    m2 = np.square(deviation, out=deviation).sum(axis=axes)
    return n.squeeze(axis=axes), mean.squeeze(axis=axes), m2


def _combine_moments(a: _Moments, b: _Moments) -> _Moments:
    """Moments of the union of two disjoint samples (Chan et al. pairwise
    update), which unlike sums of squares does not lose precision when the
    spread is small relative to the mean."""
    n_a, mean_a, m2_a = a
    n_b, mean_b, m2_b = b
    n = n_a + n_b
    delta = mean_b - mean_a
    weight_b = np.divide(n_b, n, out=np.zeros_like(n), where=n > 0)
    mean = mean_a + delta * weight_b
    m2 = m2_a + m2_b + delta**2 * n_a * weight_b
    return n, mean, m2


def _time_moments(
    values: np.ndarray, axes: tuple[int, ...], lead: int
) -> tuple[_Moments, _Moments]:
    """Moments over `axes` of `values[lead:]` and of the differences between
    consecutive steps of `values`, whose first axis is time (and is in `axes`).

    The steps are reduced a few at a time and the results combined, so that the
    float64 copies made on the way stay small.
    """
    full: _Moments | None = None
    difference: _Moments | None = None
    for start in range(lead, values.shape[0], REDUCTION_STEPS):
        # One step before the slab is included, for the difference ending in
        # the slab's first step.
        before = min(start, 1)
        slab = np.asarray(
            values[start - before : start + REDUCTION_STEPS], dtype=np.float64
        )
        slab_full = _moments(slab[before:], axes)
        slab_difference = _moments(np.diff(slab, axis=0), axes)
        full = slab_full if full is None else _combine_moments(full, slab_full)
        difference = (
            slab_difference
            if difference is None
            else _combine_moments(difference, slab_difference)
        )
    if full is None or difference is None:
        raise ValueError("No time steps to reduce.")
    return full, difference


def _block_moments(
    store: str, start: int, stop: int, first: int, transforms: DatasetPair | None
) -> dict[str, dict]:
    """Moments of every variable over the time steps [start, stop) of a store,
    and of its time differences ending in those steps. `first` is the first
    step of the whole time slice, which has no time difference ending in it.

    Variables are read one at a time, so memory use is that of one variable
    over the block rather than of the whole block.
    """
    ds = _open_store(store)
    lead = 0 if start == first else 1
    block = ds.isel(time=slice(start - lead, stop))
    if transforms is not None:
        block = apply_transforms(block, transforms)
    block = block.drop_vars(DROP_VARIABLES, errors="ignore")
    reduction_dims = _reduction_dims(block)

    result = {}
    for name, da in block.data_vars.items():
        has_time = "time" in da.dims
        if not has_time and start != first:
            continue  # a static variable is the same in every block
        if not np.issubdtype(da.dtype, np.number):
            raise ValueError(f"{store}: variable {name!r} has dtype {da.dtype}.")
        if has_time:
            da = da.transpose("time", ...)
        axes = tuple(i for i, dim in enumerate(da.dims) if dim in reduction_dims)
        values = da.values
        if has_time:
            full, difference = _time_moments(values, axes, lead)
        else:
            # As xarray's Dataset.diff, which leaves variables without the
            # dimension unchanged.
            full = difference = _moments(np.asarray(values, dtype=np.float64), axes)
        dtype = da.dtype if np.issubdtype(da.dtype, np.floating) else np.float64
        result[name] = {
            "dims": tuple(d for d in da.dims if d not in reduction_dims),
            "dtype": np.dtype(dtype).str,
            "full": full,
            "difference": difference,
        }
    return result


def _time_block_size(ds: xr.Dataset, n_steps: int, min_blocks: int) -> int:
    """Number of time steps per task: the time extent of a shard if the store
    is sharded, so each task reads whole files, else a multiple of the chunk.
    It is halved (down to MIN_TIME_BLOCK) until `n_steps` gives `min_blocks`
    tasks."""
    size = TARGET_TIME_BLOCK
    for da in ds.data_vars.values():
        if "time" in da.dims and da.ndim > 1:
            time_axis = da.dims.index("time")
            shards = da.encoding.get("shards")
            chunks = da.encoding.get("chunks")
            if shards is not None:
                size = int(shards[time_axis])
            elif chunks is not None:
                chunk = int(chunks[time_axis])
                size = chunk * max(1, round(TARGET_TIME_BLOCK / chunk))
            break
    while n_steps < min_blocks * size and size // 2 >= MIN_TIME_BLOCK:
        size //= 2
    return size


def _time_blocks(start: int, stop: int, size: int) -> list[tuple[int, int]]:
    """Split [start, stop) at the multiples of `size`."""
    edges = [start, *range((start // size + 1) * size, stop, size), stop]
    return list(zip(edges[:-1], edges[1:]))


def compute_store_stats(
    store: str,
    time_slice: slice,
    transforms: DatasetPair | None = None,
    pool: concurrent.futures.Executor | None = None,
    min_blocks: int = 1,
) -> dict:
    """Open one zarr store, slice it in time, and return its centering /
    full-field / residual stats plus the timestep count used as the pooling
    weight. If `transforms` is given, its renamed / derived / scaled variables
    are applied first, so all three stats (including the time-difference
    residual) are of the transformed variables.

    The stats are those xarray gives for `ds.mean(dims)`, `ds.std(dims)` and
    `ds.diff("time").std(dims)` over time and the horizontal dims, skipping
    NaNs. They are accumulated in float64 over blocks of time steps, each
    reading every variable once; the blocks are computed by `pool` if given
    (see `_init_worker`), else in this process. `min_blocks` is the number of
    blocks to aim for at least, so that a pool has enough tasks for its workers.
    """
    ds = xr.open_zarr(store, chunks=None)
    positions = xr.DataArray(
        np.arange(ds.sizes["time"]), coords={"time": ds["time"]}, dims=["time"]
    )
    selected = positions.sel(time=time_slice).values

    bounds = f"[{time_slice.start}, {time_slice.stop}]"
    n_samples = len(selected)
    if n_samples < 2:
        raise ValueError(
            f"{store} sliced to {bounds} has {n_samples} timesteps; check the "
            "time bounds in the config."
        )
    start, stop = int(selected[0]), int(selected[-1]) + 1
    if stop - start != n_samples:
        raise ValueError(f"{store} sliced to {bounds} is not contiguous in time.")

    blocks = _time_blocks(start, stop, _time_block_size(ds, n_samples, min_blocks))
    logging.info(
        f"{store} {bounds}: {n_samples} steps in {len(blocks)} blocks, reducing "
        f"over {_reduction_dims(ds)}"
    )
    arguments = (
        itertools.repeat(store),
        [block_start for block_start, _ in blocks],
        [block_stop for _, block_stop in blocks],
        itertools.repeat(start),
        itertools.repeat(transforms),
    )
    if pool is None:
        results: Iterable[dict[str, dict]] = map(_block_moments, *arguments)
    else:
        results = pool.map(_block_moments, *arguments)

    total: dict[str, dict] = {}
    for i, result in enumerate(results):
        for name, moments in result.items():
            if name not in total:
                total[name] = moments
            else:
                for key in ("full", "difference"):
                    total[name][key] = _combine_moments(total[name][key], moments[key])
        if (i + 1) % max(50, len(blocks) // 10) == 0:
            logging.info(f"{store} {bounds}: {i + 1}/{len(blocks)} blocks done")

    def _mean(moments: dict, key: str) -> np.ndarray:
        n, mean, _ = moments[key]
        return np.where(n > 0, mean, np.nan).astype(moments["dtype"])

    def _std(moments: dict, key: str) -> np.ndarray:
        n, _, m2 = moments[key]
        variance = np.divide(m2, n, out=np.full_like(m2, np.nan), where=n > 0)
        return np.sqrt(variance).astype(moments["dtype"])

    def _dataset(function, key: str) -> xr.Dataset:
        return xr.Dataset(
            {name: (m["dims"], function(m, key)) for name, m in total.items()}
        )

    return {
        "store": store,
        "centering": _dataset(_mean, "full"),
        "scaling_full_field": _dataset(_std, "full"),
        "scaling_residual": _dataset(_std, "difference"),
        "n_samples": n_samples,
    }


def _stats_files(stats: dict) -> dict:
    """Map a per-store stats dict to its {filename: Dataset} output layout."""
    return {
        "centering.nc": stats["centering"],
        "scaling-full-field.nc": stats["scaling_full_field"],
        "scaling-residual.nc": stats["scaling_residual"],
    }


def pool_stats(per_pair: list[dict]) -> dict:
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


def write_stats(stats: dict, out_dir: str, config_yaml: str, n_samples: int):
    """Write the three stats netCDFs to `out_dir` (local or remote).

    `n_samples` is recorded in the file attrs so a later re-pooling of a subset
    of these stats has its weights without re-reading any zarr.
    """
    history = f"Created by scripts/data_process/get_pooled_stats.py from {config_yaml}."
    out_dir = out_dir.rstrip("/")
    if is_local(out_dir):
        makedirs(out_dir)

    # netCDFs are written locally first since xarray cannot write directly to
    # remote storage; `copy` handles both local and remote destinations.
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, ds in stats.items():
            ds.attrs["history"] = history
            ds.attrs["n_samples"] = n_samples
            local_path = os.path.join(tmpdir, filename)
            ds.to_netcdf(local_path)
            copy(local_path, out_dir + "/" + filename)
            logging.info(f"Wrote {out_dir}/{filename}")


def load_store_stats(store: str, out_dir: str) -> dict | None:
    """Load the per-store stats `write_stats` wrote to `out_dir`, in the layout
    `compute_store_stats` returns, or None if any of the three files is missing.

    The directory name identifies the store and time slice (see
    `DatasetPair.output_subdir`) but not the transforms, so an output directory
    must not be reused after changing an entry's renames / derived / scalings.
    """
    out_dir = out_dir.rstrip("/")
    keys = {
        "centering.nc": "centering",
        "scaling-full-field.nc": "scaling_full_field",
        "scaling-residual.nc": "scaling_residual",
    }
    if not all(path_exists(out_dir + "/" + filename) for filename in keys):
        return None
    stats: dict = {"store": store}
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, key in keys.items():
            local_path = os.path.join(tmpdir, filename)
            copy(out_dir + "/" + filename, local_path)
            with xr.open_dataset(local_path) as ds:
                stats[key] = ds.load()
    n_samples = {int(stats[key].attrs["n_samples"]) for key in keys.values()}
    if len(n_samples) != 1:
        raise ValueError(f"Inconsistent n_samples {n_samples} in {out_dir}.")
    stats["n_samples"] = n_samples.pop()
    return stats


def parse_shard(shard: str) -> tuple[int, int]:
    """Parse an `I/N` shard spec into (I, N), with 0 <= I < N."""
    try:
        index, count = (int(part) for part in shard.split("/"))
    except ValueError:
        raise ValueError(f"shard must be I/N, got {shard!r}.") from None
    if not 0 <= index < count:
        raise ValueError(f"shard must have 0 <= I < N, got {shard!r}.")
    return index, count


def compute(
    config_yaml: str,
    output_directory: str,
    shard: str | None = None,
    n_workers: int = DEFAULT_WORKERS,
):
    """Compute the pooled stats, log them, and write them. Runs in the beaker
    container, or locally if you have the data access and dependencies.

    Stores whose stats are already in `output_directory` are loaded rather than
    recomputed. With `shard` set to `I/N`, only the stores whose index is I
    modulo N are computed and nothing is pooled. The stores are read by
    `n_workers` processes.
    """
    logging.basicConfig(level=logging.INFO)
    shard_index, shard_count = parse_shard(shard) if shard is not None else (0, 1)

    config = Config.from_file(config_yaml)
    output_directory = output_directory.rstrip("/")

    # Worker processes are spawned, not forked: the parent has already started
    # threads (torch, zarr's event loop) that a fork would leave in a bad state.
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )
    logging.info(f"Started a pool of {n_workers} worker processes")

    try:
        per_pair = []
        index = 0
        for pair in config.dataset_pairs:
            stores = _resolve_zarr_stores(pair.dataset)
            if stores != [pair.dataset.rstrip("/")]:
                logging.info(f"Resolved {pair.dataset} -> {stores}")
            for store in stores:
                store_index = index
                index += 1
                if store_index % shard_count != shard_index:
                    continue
                store_directory = (
                    output_directory + "/" + pair.output_subdir(store_index, store)
                )
                stats = load_store_stats(store, store_directory)
                if stats is not None:
                    logging.info(f"Reusing existing stats in {store_directory}")
                else:
                    stats = compute_store_stats(
                        store,
                        pair.time_slice,
                        transforms=pair,
                        pool=pool,
                        min_blocks=4 * n_workers,
                    )
                    write_stats(
                        _stats_files(stats),
                        store_directory,
                        config_yaml,
                        n_samples=stats["n_samples"],
                    )
                stats["groups"] = pair.group_names
                per_pair.append(stats)

        if shard is not None:
            logging.info(f"Shard {shard}: wrote {len(per_pair)} stores, not pooling.")
            return

        total_samples = sum(p["n_samples"] for p in per_pair)
        pooled = pool_stats(per_pair)

        # Show every variable, not just the first and last few, so the pooled
        # values can be read from the job logs.
        with xr.set_options(display_max_rows=10_000):
            for name, da in pooled.items():
                logging.info(f"{name}:\n{da}")

        write_stats(pooled, output_directory, config_yaml, n_samples=total_samples)

        # Pool each group separately from the same per-store moments, so every
        # grouping strategy is derived from one pass over the data.
        for group_name in config.groups:
            members = [p for p in per_pair if group_name in p["groups"]]
            group_samples = sum(p["n_samples"] for p in members)
            logging.info(
                f"Pooling group {group_name!r} over {len(members)} stores "
                f"({group_samples} steps)"
            )
            write_stats(
                pool_stats(members),
                output_directory + "/groups/" + group_name,
                config_yaml,
                n_samples=group_samples,
            )

        # Save the input config alongside the stats so the output is reproducible.
        copy(config_yaml, output_directory + "/" + os.path.basename(config_yaml))
        logging.info(f"Wrote {output_directory}/{os.path.basename(config_yaml)}")
    finally:
        pool.shutdown(cancel_futures=True)


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
        "pip install --no-deps .",
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
    if args.shard is not None:
        cmd += ["--shard", args.shard]
    cmd += ["--workers", str(args.workers)]

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

    shard_help = (
        "I/N: compute only the stores whose index is I modulo N and skip the "
        "pooling; run again without --shard to pool."
    )
    p_compute.add_argument("--shard", default=None, help=shard_help)
    workers_help = "Number of worker processes reading the stores."
    p_compute.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help=workers_help
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
    p_submit.add_argument("--shard", default=None, help=shard_help)
    p_submit.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help=workers_help
    )
    p_submit.add_argument(
        "--dry-run", action="store_true", help="Print the gantry command only."
    )

    args = parser.parse_args()
    if args.mode == "compute":
        compute(
            args.config_yaml,
            args.output_directory,
            shard=args.shard,
            n_workers=args.workers,
        )
    else:
        if args.shard is not None:
            parse_shard(args.shard)
        submit(args)


if __name__ == "__main__":
    main()
