"""Check per-grid-cell normalization stats written by get_pooled_stats.py.

Runs on beaker (the stats live on weka), after
`get_pooled_stats.py compute scripts/data_process/configs/
fm-v3-e3sm-balanced-percell-stats.yaml OUT` (or another per-cell stats
config, e.g. fm-v3-{c96amip,era5-1979}-percell-stats.yaml for the FM v3
grid's single-source runs). For the pooled stats at OUT and
each group under OUT/groups/, it checks that:

- every variable of the training config's in_names and out_names is present;
- per-cell means are finite and stds positive in every cell, and counts the
  cells whose std would be floored at 1% and 10% of the variable's mean std
  over cells (fme's per_grid_cell_min_std_fraction);
- the per-cell horizontal coordinates equal, dim by dim in order, those of
  every store of the training config's train_loader (stores name their dims
  lat/lon or grid_yt/grid_xt; fme matches them by position);
- the scalar-kept variables are scalars equal to the reference values;
- each group_placeholders variable is a per-cell 0 (mean) / 1 (std);
- the files load and build with `NormalizationConfig(per_grid_cell=True)`.

It prints a summary, writes maps of the pooled per-cell mean and std of a few
variables to OUT/plots/, and exits non-zero if any check fails.

Usage:
    python verify_percell_stats.py OUT TRAIN_CONFIG STATS_CONFIG
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

from fme.core.normalizer import NormalizationConfig  # noqa: E402

FILES = ("centering.nc", "scaling-full-field.nc", "scaling-residual.nc")
STD_FILES = FILES[1:]
FLOOR_FRACTIONS = (0.01, 0.1)
PLOT_VARIABLES = ("TMP2m", "PRATEsfc", "air_temperature_7")


def _find_key(tree, key: str):
    """First value of `key` in a nested dict/list structure."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        children = list(tree.values())
    elif isinstance(tree, list):
        children = tree
    else:
        return None
    for child in children:
        found = _find_key(child, key)
        if found is not None:
            return found
    return None


class Checker:
    def __init__(self):
        self.failures: list[str] = []

    def check(self, condition: bool, message: str):
        if not condition:
            self.failures.append(message)
            print(f"FAIL: {message}")


def _load(directory: pathlib.Path) -> dict[str, xr.Dataset]:
    return {name: xr.open_dataset(directory / name).load() for name in FILES}


def _check_directory(
    checker: Checker,
    label: str,
    directory: pathlib.Path,
    names: list[str],
    scalar_names: list[str],
    reference: dict[str, xr.Dataset],
    grids: dict[str, list[tuple[str, np.ndarray]]],
):
    print(f"\n=== {label} ===")
    files = _load(directory)
    per_cell = [
        name
        for name in names
        if name not in scalar_names and name in files["centering.nc"]
    ]
    horizontal = files["centering.nc"][per_cell[0]].dims
    print(f"per-cell dims: {horizontal}")
    floored: dict[float, dict[tuple[str, str], int]] = {
        fraction: {} for fraction in FLOOR_FRACTIONS
    }
    min_ratio: dict[tuple[str, str], float] = {}
    for filename, ds in files.items():
        missing = sorted(set(names) - set(ds.data_vars))
        checker.check(not missing, f"{label}/{filename}: missing {missing}")
        for name in names:
            if name not in ds:
                continue
            da = ds[name]
            if name in scalar_names:
                checker.check(
                    da.dims == (), f"{label}/{filename}: {name} has dims {da.dims}"
                )
                expected = float(reference[filename][name])
                checker.check(
                    float(da) == expected,
                    f"{label}/{filename}: {name}={float(da)!r} != reference "
                    f"{expected!r}",
                )
                continue
            checker.check(
                da.dims == horizontal, f"{label}/{filename}: {name} has dims {da.dims}"
            )
            values = da.values.astype(np.float64)
            n_bad = int((~np.isfinite(values)).sum())
            checker.check(
                n_bad == 0, f"{label}/{filename}: {name} non-finite in {n_bad} cells"
            )
            if filename in STD_FILES:
                n_nonpositive = int((values <= 0).sum())
                checker.check(
                    n_nonpositive == 0,
                    f"{label}/{filename}: {name} std <= 0 in {n_nonpositive} cells",
                )
                mean_std = np.nanmean(values)
                min_ratio[(filename, name)] = float(np.nanmin(values) / mean_std)
                for fraction in FLOOR_FRACTIONS:
                    count = int((values < fraction * mean_std).sum())
                    if count:
                        floored[fraction][(filename, name)] = count
        for store, grid in grids.items():
            checker.check(
                len(grid) == len(horizontal),
                f"{label}/{filename}: dims {horizontal} vs {store} {grid}",
            )
            for dim, (store_dim, store_values) in zip(horizontal, grid):
                values = ds[dim].values
                same = np.array_equal(values, store_values)
                close = values.shape == store_values.shape and np.allclose(
                    values, store_values, rtol=1e-5, atol=1e-4
                )
                checker.check(
                    close,
                    f"{label}/{filename}: {dim} does not match {store_dim} of {store}",
                )
                if close and not same:
                    print(
                        f"note: {label}/{filename}: {dim} only close to "
                        f"{store_dim} of {store} (max abs diff "
                        f"{np.abs(values - store_values).max():.3g})"
                    )

    n_cells = int(np.prod([files["centering.nc"].sizes[d] for d in horizontal]))
    print(f"grid cells: {n_cells}")
    for fraction in FLOOR_FRACTIONS:
        counts = floored[fraction]
        total = sum(counts.values())
        print(
            f"cells floored at fraction {fraction}: {total} in total, over "
            f"{len(counts)} (file, variable) pairs"
        )
        for (filename, name), count in sorted(counts.items()):
            print(f"  {filename:24s} {name:48s} {count}")
    worst = sorted(min_ratio.items(), key=lambda item: item[1])[:8]
    print("smallest min-std / mean-std ratios:")
    for (filename, name), ratio in worst:
        print(f"  {filename:24s} {name:48s} {ratio:.4g}")

    for std_file in STD_FILES:
        try:
            normalizer = NormalizationConfig(
                global_means_path=str(directory / "centering.nc"),
                global_stds_path=str(directory / std_file),
                per_grid_cell=True,
            ).build(names)
            n_per_cell = sum(1 for v in normalizer.means.values() if v.numel() > 1)
            print(
                f"built NormalizationConfig(per_grid_cell=True) with {std_file}: "
                f"{len(normalizer.means)} variables, {n_per_cell} per-cell"
            )
        except Exception as error:  # report every failure, not just the first
            checker.check(False, f"{label}: building with {std_file} failed: {error}")


def _plot(files: dict[str, xr.Dataset], out_dir: pathlib.Path):
    out_dir.mkdir(exist_ok=True)
    for name in PLOT_VARIABLES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
        for ax, (filename, title) in zip(
            axes, (("centering.nc", "mean"), ("scaling-full-field.nc", "std"))
        ):
            da = files[filename][name]
            y, x = da.dims
            mesh = ax.pcolormesh(da[x], da[y], da.values, shading="auto")
            fig.colorbar(mesh, ax=ax)
            ax.set_title(f"pooled {name} per-cell {title}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        path = out_dir / f"pooled-{name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_dir", type=pathlib.Path)
    parser.add_argument("train_config", type=pathlib.Path)
    parser.add_argument("stats_config", type=pathlib.Path)
    args = parser.parse_args()

    train = yaml.safe_load(args.train_config.read_text())
    stats_config = yaml.safe_load(args.stats_config.read_text())
    names = sorted(
        set(_find_key(train, "in_names")) | set(_find_key(train, "out_names"))
    )
    scalar_names = stats_config["scalar_variables"]
    reference_root = pathlib.Path(stats_config["scalar_reference"]["directory"])
    reference_groups = stats_config["scalar_reference"].get("groups", {})
    print(f"{len(names)} training variables; scalar-kept: {scalar_names}")

    grids = {}
    for entry in train["train_loader"]["dataset"]["concat"]:
        store = f"{entry['data_path']}/{entry['file_pattern']}"
        if store not in grids:
            with xr.open_zarr(store) as ds:
                # The horizontal dims of the first time-varying field.
                field = next(
                    da
                    for da in ds.data_vars.values()
                    if da.ndim == 3 and da.dims[0] == "time"
                )
                grids[store] = [(dim, ds[dim].values) for dim in field.dims[1:]]
    print(f"compared coordinates against {len(grids)} training stores")

    checker = Checker()
    directories = {"pooled": (args.stats_dir, reference_root)}
    for group in stats_config["groups"]:
        directories[f"groups/{group}"] = (
            args.stats_dir / "groups" / group,
            reference_root / "groups" / reference_groups.get(group, group),
        )
    for label, (directory, reference_directory) in directories.items():
        _check_directory(
            checker,
            label,
            directory,
            names,
            scalar_names,
            _load(reference_directory),
            grids,
        )

    for group, placeholders in stats_config.get("group_placeholders", {}).items():
        group_files = _load(args.stats_dir / "groups" / group)
        for name in placeholders:
            for filename, ds in group_files.items():
                expected = 0.0 if filename == "centering.nc" else 1.0
                checker.check(
                    ds[name].ndim == 2 and bool((ds[name] == expected).all()),
                    f"groups/{group}/{filename}: {name} is not a per-cell "
                    f"{expected} placeholder",
                )

    _plot(_load(args.stats_dir), args.stats_dir / "plots")

    print("\n=== SUMMARY ===")
    if checker.failures:
        print(f"{len(checker.failures)} checks FAILED:")
        for failure in checker.failures:
            print(f"  {failure}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
