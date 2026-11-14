"""Check that the data a 1deg 6-hourly FM base config reads is in place.

For each config given:
- every (data_path, file_pattern) store of train_loader, validation and
  inference opens, is 6-hourly and is on a 180 x 360 grid; the variables of
  the stepper's in_names / out_names that a train store lacks (after its
  renames and derived variables) are listed, but are not an error, as the
  configs allow missing variables;
- every normalization file of stepper.step.config.normalization exists and
  has a finite value for every in / out name (scalings that are not positive
  are listed).

With --constant-co2-store, also check that the store has the constant
global_mean_co2 that add_constant_co2.py appends.

Run it where the configs' paths are mounted (a Beaker job with weka). Exits
non-zero if any check fails.

Usage:
    python verify_fm_1deg_data.py CONFIG [CONFIG ...] [--constant-co2-store STORE]
"""

import argparse
import sys

import numpy as np
import xarray as xr
import yaml

from fme.core.dataset.xarray import DERIVED_VARIABLES

EXPECTED_STEP = np.timedelta64(6, "h")
EXPECTED_GRID = (180, 360)
HORIZONTAL_DIMS = [("lat", "lon"), ("latitude", "longitude"), ("grid_yt", "grid_xt")]
CO2_NAME = "global_mean_co2"
CO2_VALUE = np.float32(284.317e-6)


def _dataset_entries(node) -> list[dict]:
    """Every mapping with a data_path anywhere beneath `node`."""
    entries = []
    if isinstance(node, dict):
        if "data_path" in node:
            entries.append(node)
        for value in node.values():
            entries.extend(_dataset_entries(value))
    elif isinstance(node, list):
        for value in node:
            entries.extend(_dataset_entries(value))
    return entries


def _normalization_paths(node) -> list[str]:
    paths = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("global_means_path", "global_stds_path"):
                paths.append(value)
            else:
                paths.extend(_normalization_paths(value))
    return paths


def _store_path(entry: dict) -> str:
    return f"{entry['data_path'].rstrip('/')}/{entry['file_pattern']}"


def _check_store(path: str) -> tuple[xr.Dataset | None, list[str]]:
    try:
        ds = xr.open_zarr(path, chunks=None)
    except Exception as error:
        return None, [f"{path}: cannot open ({type(error).__name__}: {error})"]
    errors = []
    times = ds["time"].values[:2]
    step = times[1] - times[0]
    if np.timedelta64(step) != EXPECTED_STEP:
        errors.append(f"{path}: time step is {step}, not 6 hours")
    grids = [
        (ds.sizes[lat], ds.sizes[lon])
        for lat, lon in HORIZONTAL_DIMS
        if lat in ds.sizes and lon in ds.sizes
    ]
    if grids != [EXPECTED_GRID]:
        errors.append(f"{path}: grid is {grids}, not {EXPECTED_GRID}")
    print(
        f"  store ok={not errors} steps={ds.sizes['time']} "
        f"{str(ds['time'].values[0])[:13]}..{str(ds['time'].values[-1])[:13]} "
        f"variables={len(ds.data_vars)} {path}"
    )
    return ds, errors


def _missing_names(ds: xr.Dataset, entry: dict, names: list[str]) -> list[str]:
    renamed = entry.get("renamed_variables") or {}
    available = {renamed.get(name, name) for name in ds.variables}
    for name in entry.get("derived_variables") or []:
        if all(i in available for i in DERIVED_VARIABLES[name].inputs):
            available.add(name)
    return sorted(set(names) - available)


def _check_normalization(path: str, names: list[str]) -> list[str]:
    try:
        ds = xr.open_dataset(path).load()
    except Exception as error:
        return [f"{path}: cannot open ({type(error).__name__}: {error})"]
    errors = []
    missing = sorted(set(names) - set(ds.data_vars))
    if missing:
        errors.append(f"{path}: lacks {missing}")
    is_scaling = "scaling" in path.rsplit("/", 1)[-1]
    for name in names:
        if name not in ds:
            continue
        value = ds[name].values
        if not np.all(np.isfinite(value)):
            errors.append(f"{path}: {name} is {value}")
        elif is_scaling and not np.all(value > 0):
            # Not an error: a variable that is constant within a group (e.g.
            # the piControl CO2) has zero scaling there, and is pinned to the
            # pooled stats by the configs.
            print(f"    note: scaling of {name} is {value} in {path}")
    print(
        f"  stats ok={not errors} variables={len(ds.data_vars)} "
        f"n_samples={ds.attrs.get('n_samples')} {path}"
    )
    return errors


def check_config(config_path: str) -> list[str]:
    print(f"== {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    step = config["stepper"]["step"]["config"]
    names = sorted(set(step["in_names"]) | set(step["out_names"]))
    train_stores = {_store_path(e): e for e in _dataset_entries(config["train_loader"])}
    stores = {_store_path(e) for e in _dataset_entries(config)}

    errors = []
    for path in sorted(stores):
        ds, store_errors = _check_store(path)
        errors.extend(store_errors)
        if ds is not None and path in train_stores:
            missing = _missing_names(ds, train_stores[path], names)
            if missing:
                print(f"    train store lacks {missing}")
    for path in sorted(set(_normalization_paths(step["normalization"]))):
        errors.extend(_check_normalization(path, names))
    return errors


def check_constant_co2(path: str) -> list[str]:
    print(f"== constant {CO2_NAME} in {path}")
    ds = xr.open_zarr(path, chunks=None)
    if CO2_NAME not in ds:
        return [f"{path}: has no {CO2_NAME}"]
    co2 = ds[CO2_NAME]
    values = co2.values
    print(
        f"  dims={co2.dims} dtype={co2.dtype} shape={values.shape} "
        f"min={values.min()!r} max={values.max()!r} attrs={co2.attrs}"
    )
    errors = []
    if co2.dims != ("time",) or values.shape != (ds.sizes["time"],):
        errors.append(f"{path}: {CO2_NAME} has dims {co2.dims} shape {values.shape}")
    if co2.dtype != np.float32 or not np.all(values == CO2_VALUE):
        errors.append(f"{path}: {CO2_NAME} is not float32 {CO2_VALUE!r} throughout")
    if co2.attrs.get("units") != "volume mixing ratio":
        errors.append(f"{path}: {CO2_NAME} units are {co2.attrs.get('units')!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--constant-co2-store", default=None)
    args = parser.parse_args()

    errors = []
    for config_path in args.configs:
        errors.extend(check_config(config_path))
    if args.constant_co2_store is not None:
        errors.extend(check_constant_co2(args.constant_co2_store))

    if errors:
        print(f"FAILED: {len(errors)} problems")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
