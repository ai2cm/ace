"""Inventory CMIP6 daily-table datasets available in the Pangeo GCS catalog.

Queries the Pangeo CMIP6 intake-esm catalog for the configured variables x
experiments x table_id, parses ``member_id`` into r/i/p/f components, and
writes a tidy parquet file. By default, catalog-only (fast). Pass
``--enrich-stores`` to additionally open each zarr and record calendar and
time range (slow — one GCS open per row).

Usage:
    python inventory.py --config configs/inventory.yaml
    python inventory.py --config <cfg> --enrich-stores

Run from inside scripts/cmip6_data/, or give absolute paths.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import InventoryConfig

PANGEO_CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv.gz"

_VARIANT_RE = re.compile(r"r(\d+)i(\d+)p(\d+)f(\d+)")


def parse_variant(variant_label: str) -> dict[str, int | None]:
    m = _VARIANT_RE.fullmatch(variant_label or "")
    if not m:
        return {"r": None, "i": None, "p": None, "f": None}
    return {
        "r": int(m.group(1)),
        "i": int(m.group(2)),
        "p": int(m.group(3)),
        "f": int(m.group(4)),
    }


def query_catalog(config: InventoryConfig) -> pd.DataFrame:
    logging.info("Reading Pangeo CMIP6 catalog: %s", PANGEO_CATALOG_URL)
    df = pd.read_csv(PANGEO_CATALOG_URL)
    logging.info("Catalog has %d total rows", len(df))

    pieces: list[pd.DataFrame] = []
    for query in config.queries:
        mask = df["table_id"].eq(query.table_id) & df["variable_id"].isin(
            query.variables
        )
        if query.filter_by_experiment:
            mask &= df["experiment_id"].isin(config.experiments)
        piece = df.loc[mask]
        logging.info(
            "  %-8s %d rows (%d variables, filter_by_experiment=%s)",
            query.table_id,
            len(piece),
            len(query.variables),
            query.filter_by_experiment,
        )
        pieces.append(piece)

    subset = pd.concat(pieces, ignore_index=True)
    logging.info("Filtered total: %d rows across %d queries", len(subset), len(pieces))
    return subset


def parse_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["member_id"].apply(parse_variant).apply(pd.Series)
    return df.assign(
        variant_r=parsed["r"],
        variant_i=parsed["i"],
        variant_p=parsed["p"],
        variant_f=parsed["f"],
    )


def enrich_from_stores(df: pd.DataFrame) -> pd.DataFrame:
    """Open each zarr store and add calendar + time range columns.

    Slow — one GCS open per row. Failures are logged and recorded as
    empty strings rather than aborting the run.
    """
    import fsspec
    import xarray as xr

    calendars, time_starts, time_ends, open_errors = [], [], [], []
    n = len(df)
    for i, zstore in enumerate(df["zstore"], start=1):
        if i % 50 == 0 or i == n:
            logging.info("Enriching %d / %d", i, n)
        try:
            ds = xr.open_zarr(
                fsspec.get_mapper(zstore), consolidated=True, decode_times=False
            )
            time = ds["time"]
            cal = time.encoding.get("calendar") or time.attrs.get("calendar") or ""
            units = time.attrs.get("units", "")
            if len(time) and units:
                first = xr.decode_cf(ds[["time"]].isel(time=[0]))["time"].values[0]
                last = xr.decode_cf(ds[["time"]].isel(time=[-1]))["time"].values[0]
                t0, t1 = str(first), str(last)
            else:
                t0, t1 = "", ""
            calendars.append(cal)
            time_starts.append(t0)
            time_ends.append(t1)
            open_errors.append("")
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to open %s: %s", zstore, e)
            calendars.append("")
            time_starts.append("")
            time_ends.append("")
            open_errors.append(f"{type(e).__name__}: {e}")
    return df.assign(
        calendar=calendars,
        time_start=time_starts,
        time_end=time_ends,
        open_error=open_errors,
    )


def build_inventory(
    config: InventoryConfig, enrich_stores: bool = False
) -> pd.DataFrame:
    df = query_catalog(config)
    df = parse_variant_columns(df)
    if enrich_stores:
        df = enrich_from_stores(df)
    return df


def write_inventory(df: pd.DataFrame, path: str) -> None:
    logging.info("Writing %d rows to %s", len(df), path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv") or path.endswith(".csv.gz"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(
            f"Unrecognized output extension: {path} "
            "(expected .parquet, .csv, or .csv.gz)"
        )


def summarize(
    df: pd.DataFrame,
    core_variables: list[str],
    forcing_variables: list[str],
    static_variables: list[str],
) -> str:
    lines = [f"Total rows: {len(df)}"]
    lines.append(f"Unique source_ids: {df['source_id'].nunique()}")
    lines.append("")

    # Per-table breakdown
    lines.append("Per-table rows / models / (distinct) members:")
    for tab, sub in df.groupby("table_id"):
        lines.append(
            f"  {tab:<8} {len(sub):>5} rows, "
            f"{sub['source_id'].nunique():>3} models, "
            f"{sub['member_id'].nunique():>4} distinct members"
        )
    lines.append("")

    day = df[df["table_id"] == "day"]

    # Members per (source_id, experiment) in day table
    members = (
        day.groupby(["source_id", "experiment_id"])["member_id"]
        .nunique()
        .reset_index(name="n_members")
    )
    lines.append("Members per (source_id, experiment_id) in 'day' — top 10:")
    for _, row in members.sort_values("n_members", ascending=False).head(10).iterrows():
        lines.append(
            f"  {row['source_id']:<24} {row['experiment_id']:<12} {row['n_members']}"
        )
    if len(members):
        lines.append(f"  (median n_members = {int(members['n_members'].median())})")
    lines.append("")

    # Core-variable coverage in day table
    have_core = day.groupby(["source_id", "experiment_id", "member_id"])[
        "variable_id"
    ].agg(set)
    full_core = have_core.apply(lambda s: set(core_variables).issubset(s))
    n_full = int(full_core.sum())
    n_total = len(full_core)
    lines.append(
        f"day-table datasets with ALL core variables: {n_full} / {n_total} "
        f"({(n_full / n_total * 100 if n_total else 0):.0f}%) — "
        f"{full_core[full_core].reset_index().source_id.nunique()} models"
    )

    # Forcing coverage
    models_day = set(day["source_id"].unique())
    models_by_table_var: dict[tuple[str, str], set[str]] = {}
    for (tab, var), sub in df.groupby(["table_id", "variable_id"]):
        models_by_table_var[(tab, var)] = set(sub["source_id"].unique())

    lines.append("")
    lines.append("Forcing-variable coverage (intersected with day-table models):")
    # ts is in Amon, siconc in SImon, sftlf/orog in fx
    for var in forcing_variables:
        hits = set()
        for tab in ("Amon", "SImon"):
            hits |= models_by_table_var.get((tab, var), set())
        hits &= models_day
        lines.append(f"  {var:<10} {len(hits)} / {len(models_day)} day-table models")
    for var in static_variables:
        hits = models_by_table_var.get(("fx", var), set()) & models_day
        lines.append(f"  {var:<10} {len(hits)} / {len(models_day)} day-table models")

    # Per-variable global coverage (any experiment, any member)
    lines.append("")
    lines.append("Models publishing each (table_id, variable_id):")
    per = (
        df.groupby(["table_id", "variable_id"])["source_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    for (tab, var), n in per.items():
        lines.append(f"  {tab:<8} {var:<10} {n}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to inventory YAML")
    parser.add_argument(
        "--enrich-stores",
        action="store_true",
        help="Open each zarr to record calendar + time range (slow)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = InventoryConfig.from_file(args.config)
    df = build_inventory(config, enrich_stores=args.enrich_stores)
    write_inventory(df, config.output_path)

    from config import CORE_VARIABLES, FORCING_VARIABLES, STATIC_VARIABLES

    print()
    print(summarize(df, CORE_VARIABLES, FORCING_VARIABLES, STATIC_VARIABLES))


if __name__ == "__main__":
    main()
