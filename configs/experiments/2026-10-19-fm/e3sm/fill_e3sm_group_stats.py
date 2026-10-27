"""Write e3sm group stats with TMP850 and h500 filled from the pooled stats.

Grouped normalization requires every group to provide constants for every
non-pinned variable, but the E3SM store has no TMP850 or h500, so
get_pooled_stats.py writes groups/e3sm/ without them. This copies
groups/e3sm/*.nc into a new directory and adds TMP850 and h500 from the
pooled *.nc files of the same stats root. Those variables are NaN-filled and
masked out of the loss for every E3SM sample, so the filled values only set
the normalization of a variable E3SM never provides.

Refuses to run if the output directory already exists.

Usage:
    python fill_e3sm_group_stats.py STATS_ROOT OUTPUT_DIR

where STATS_ROOT is a local copy of the v3 stats directory (holding
centering.nc and groups/e3sm/centering.nc, etc.).
"""

import argparse
import pathlib

import xarray as xr

FILES = ("centering.nc", "scaling-full-field.nc", "scaling-residual.nc")
FILLED_VARIABLES = ("TMP850", "h500")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_root", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    group_dir = args.stats_root / "groups" / "e3sm"
    datasets = {}
    for name in FILES:
        group = xr.open_dataset(group_dir / name).load()
        pooled = xr.open_dataset(args.stats_root / name).load()
        for variable in FILLED_VARIABLES:
            if variable in group:
                raise ValueError(f"{group_dir / name} already has {variable}")
            group[variable] = pooled[variable].assign_attrs(
                description="Pooled value: E3SM has no data for this variable."
            )
        datasets[name] = group

    args.output_dir.mkdir(parents=True)
    for name, ds in datasets.items():
        ds.to_netcdf(args.output_dir / name)
        print(f"Wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
