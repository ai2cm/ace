"""Check the scalar normalization stats of the FM C96 AMIP-only runs (or of
their ERA5-only `era5-1979` baseline), and the training store's time
coverage, inside a beaker job (both live on weka).

Runs after `get_pooled_stats.py compute
scripts/data_process/configs/fm-c96amip-stats.yaml OUT` (or
fm-era5-1979-stats.yaml, with --group era5). It checks that:

- OUT/{centering,scaling-full-field,scaling-residual}.nc (and groups/GROUP/,
  GROUP = c96 by default, which pools the same entries) hold every variable
  of in_names and out_names of each TRAIN_CONFIG, as scalars, finite, with
  positive full-field stds and positive residual stds for the prognostic
  variables (in in_names and out_names; fme applies the residual scaling to
  those only, and input-only forcings such as land_fraction can be constant
  in time, with residual std 0, as in the v2 stats); groups/GROUP equals
  the root;
- the files build with fme's NormalizationConfig;
- the store is daily and gap-free from 1979-01-01 through 2014-12-31, so it
  covers the training splits (1979-1993, 1995-2008), the validation years
  (1994, 2014) and the inline inference on it (8 ICs from 2009-01-01 to
  2009-11-15, each run 1825 days); it prints the training timesteps per split.

With --reference DIR (a scalar stats root, e.g. the v2 fm-balanced stats'
groups/c96), it also prints the ratio of each new value to the reference.
Exits non-zero if any check fails.

Usage:
    python verify_c96amip_stats.py OUT STORE TRAIN_CONFIG [TRAIN_CONFIG ...]
        [--reference DIR] [--group GROUP]
"""

import argparse
import datetime
import pathlib
import sys

import numpy as np
import xarray as xr
import yaml

from fme.core.normalizer import NormalizationConfig

FILES = ("centering.nc", "scaling-full-field.nc", "scaling-residual.nc")
STD_FILES = FILES[1:]
COVERAGE = ("1979-01-01", "2014-12-31")
TRAIN_SPLITS = (("1979-01-01", "1993-12-31"), ("1995-01-01", "2008-12-31"))
INFERENCE_LAST_IC = "2009-11-15"
INFERENCE_STEPS = 1825


def _names(config_path: str) -> tuple[set[str], set[str]]:
    """(in_names | out_names, in_names & out_names) of a training config."""
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    step = cfg["stepper"]["step"]["config"]
    in_names, out_names = set(step["in_names"]), set(step["out_names"])
    return in_names | out_names, in_names & out_names


def check_stats(
    out: pathlib.Path,
    names: set[str],
    prognostic: set[str],
    group_name: str,
    failures: list[str],
) -> None:
    for subdir in (out, out / "groups" / group_name):
        for name in FILES:
            ds = xr.open_dataset(subdir / name)
            missing = sorted(names - set(ds.data_vars))
            if missing:
                failures.append(f"{subdir}/{name} lacks {missing}")
            for var in sorted(names & set(ds.data_vars)):
                value = ds[var].values
                if value.ndim != 0:
                    failures.append(f"{subdir}/{name} {var} not scalar: {value.shape}")
                elif not np.isfinite(value):
                    failures.append(f"{subdir}/{name} {var} not finite: {value}")
                elif name in STD_FILES and not value > 0:
                    if name == "scaling-residual.nc" and var not in prognostic:
                        print(f"note: {subdir}/{name} {var} = {value} (not prognostic)")
                    else:
                        failures.append(f"{subdir}/{name} {var} not positive: {value}")
    for name in FILES:
        root = xr.open_dataset(out / name)
        group = xr.open_dataset(out / "groups" / group_name / name)
        for var in sorted(names):
            if var in root and var in group:
                if not np.allclose(root[var].values, group[var].values, rtol=1e-6):
                    failures.append(
                        f"groups/{group_name}/{name} {var} differs from the root"
                    )
    for scaling in STD_FILES:
        normalizer = NormalizationConfig(
            global_means_path=str(out / "centering.nc"),
            global_stds_path=str(out / scaling),
        ).build(sorted(names))
        print(f"built NormalizationConfig with {scaling}: {len(normalizer.means)} vars")


def print_table(out: pathlib.Path, names: set[str], reference: pathlib.Path | None):
    datasets = {name: xr.open_dataset(out / name) for name in FILES}
    refs = (
        {name: xr.open_dataset(reference / name) for name in FILES}
        if reference is not None
        else {}
    )
    header = f"{'variable':48s}" + "".join(
        f"{n.removesuffix('.nc'):>22s}" for n in FILES
    )
    if refs:
        header += "   ratio to reference (mean, full std, residual std)"
    print(header)
    for var in sorted(names):
        row = f"{var:48s}" + "".join(
            f"{float(datasets[n][var]):22.6g}" for n in FILES if var in datasets[n]
        )
        if refs:
            ratios = []
            for n in FILES:
                if var in refs[n] and float(refs[n][var]) != 0:
                    ratios.append(
                        f"{float(datasets[n][var]) / float(refs[n][var]):.3f}"
                    )
                else:
                    ratios.append("-")
            row += "   " + ", ".join(ratios)
        print(row)


def check_store(store: str, failures: list[str]) -> None:
    ds = xr.open_zarr(store)
    times = ds.indexes["time"]
    print(f"store {store}: {len(times)} steps, {times[0]} .. {times[-1]}")
    start, stop = (datetime.date.fromisoformat(t) for t in COVERAGE)
    window = ds.sel(time=slice(*COVERAGE)).indexes["time"]
    expected = (stop - start).days + 1
    days = np.array(
        [(datetime.date(t.year, t.month, t.day) - start).days for t in window]
    )
    if len(window) != expected or not np.array_equal(days, np.arange(expected)):
        failures.append(
            f"store not daily and gap-free over {COVERAGE}: {len(window)} steps, "
            f"expected {expected}"
        )
    last_ic = datetime.date.fromisoformat(INFERENCE_LAST_IC)
    end = last_ic + datetime.timedelta(days=INFERENCE_STEPS)
    print(f"inline inference: last IC {last_ic}, ends {end} (within coverage)")
    if end > stop:
        failures.append(f"inline inference ends {end}, after {stop}")
    for a, b in TRAIN_SPLITS:
        n = ds.sel(time=slice(a, b)).sizes["time"]
        print(f"training split {a} .. {b}: {n} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("store")
    parser.add_argument("train_configs", nargs="+")
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--group", default="c96")
    args = parser.parse_args()
    names: set[str] = set()
    prognostic: set[str] = set()
    for path in args.train_configs:
        config_names, config_prognostic = _names(path)
        if names and (config_names, config_prognostic) != (names, prognostic):
            raise SystemExit(f"{path}: in/out names differ from the first config")
        names, prognostic = config_names, config_prognostic
    print(
        f"{len(names)} variables in in_names | out_names, {len(prognostic)} prognostic"
    )
    failures: list[str] = []
    check_stats(args.out, names, prognostic, args.group, failures)
    print_table(args.out, names, args.reference)
    check_store(args.store, failures)
    print("\n=== SUMMARY ===")
    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
