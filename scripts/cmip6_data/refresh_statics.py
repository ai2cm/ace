"""One-shot fix: re-regrid + rewrite HGTsfc / land_fraction for the
~10 datasets where the prior fx-grid-selection bug landed mismatched
``orog`` / ``sftlf`` sources on a sparse-union grid.

What this script does, per affected dataset:

1. Open the existing zarr and **delete** the stale ``HGTsfc`` and
   ``land_fraction`` arrays in place.
2. Rebuild ``fx_zs`` from the inventory using the **grid-aware**
   selection (prefer fx rows whose ``grid_label`` matches the
   day-table's, fall through to any-grid only if the matching grid
   isn't published).
3. Open each fx variable on its own source grid, regrid separately,
   merge the regridded outputs (now all on the target Gauss-Legendre
   grid so the merge is a no-op spatially).
4. Apply ``clamp_static_fractions`` (rescales ``sftlf`` → ``land_fraction``
   in [0, 1]) and the CMIP→output renames (``orog`` → ``HGTsfc``).
5. Write the refreshed static variables back into the zarr in append
   mode, re-consolidate metadata.
6. Regenerate ``stats.nc`` so per-cell maps + cohort scalars reflect
   the fixed statics.
7. Append a ``static_refresh`` entry to the sidecar's ``migrations``
   audit list so the fix is visible in the provenance trail.

Why not a schema migration: the schema doesn't change — only specific
datasets get corrected. Doing it here keeps the migration framework
purely about schema and uses targeted-dataset list logic for this
one-off fix.

Why local rather than Argo: only 10 datasets, ~10s of regrid each,
public-GCS source data. Single-process pass costs minutes.

Usage::

    python refresh_statics.py --config configs/<process.yaml>
                              [--source-ids GFDL-CM4 CMCC-CM2-SR5]
                              [--dry-run] [--no-stats]

``--no-stats`` skips the per-dataset ``compute_and_write_stats`` pass.
Useful when running this script from outside the GCS bucket's region
(e.g. dev machine), where reading 30+ GB of zarr data over the network
is the slow step. Pair with a follow-up ``compute_stats.py`` run on an
in-region Argo pod with ``--source-ids`` set to the same models to
finish the job in a fraction of the wall time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from compute_stats import compute_and_write_stats  # noqa: E402
from config import (  # noqa: E402
    CMIP_TO_OUTPUT_RENAMES,
    DEFAULT_STATS_PERIODS,
    ProcessConfig,
)
from grid import make_target_grid  # noqa: E402
from process import _load_inventory, _open_zstore  # noqa: E402
from processing import (  # noqa: E402
    apply_output_renames,
    clamp_static_fractions,
    fill_orog_ocean_with_zero,
    regrid_variables,
)

# Variables we expect the fx-side rewrite to produce after rename.
# Anything in ``cfg.static_variables`` that we can resolve to a source
# fx zstore gets refreshed; these names are what survives the rename
# pass.
_OUTPUT_STATIC_NAMES = ("HGTsfc", "land_fraction")


def _read_sidecar(zarr_path: str) -> dict:
    url = zarr_path.rstrip("/") + "/metadata.json"
    with fsspec.open(url, "r") as f:
        return json.load(f)


def _write_sidecar(zarr_path: str, sidecar: dict) -> None:
    url = zarr_path.rstrip("/") + "/metadata.json"
    with fsspec.open(url, "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True, default=str)


def _scan_stale_statics(out_dir: str, source_ids: list[str] | None) -> pd.DataFrame:
    """Walk index.csv, open each stats.nc, and flag datasets whose
    ``HGTsfc`` or ``land_fraction`` ``finite_fraction`` is < 0.95.

    Threshold is generous — fixed regrids hit 1.0; the bug pattern
    produces 0.2 / 0.73. Anything in between is suspicious and we'd
    want to investigate before rewriting (so the threshold draws a
    clear line).
    """
    idx_url = f"{out_dir.rstrip('/')}/index.csv"
    idx = pd.read_csv(idx_url)
    idx = idx[idx.status == "ok"].reset_index(drop=True)
    if source_ids:
        idx = idx[idx.source_id.isin(source_ids)].reset_index(drop=True)

    rows: list[dict] = []
    n = len(idx)
    logging.info("Scanning %d ok datasets for stale statics...", n)
    for i, r in idx.iterrows():
        stats_path = r.output_zarr.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
        try:
            fs, rel = fsspec.core.url_to_fs(stats_path)
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                fs.get(rel, tmp.name)
                ds = xr.open_dataset(tmp.name).load()
        except Exception as e:  # noqa: BLE001
            logging.warning("  skip %s (cannot open stats.nc: %s)", stats_path, e)
            continue
        h_ff = lf_ff = 1.0
        h = ds.get("HGTsfc__finite_fraction")
        if h is not None:
            h_ff = (
                float(h.isel(period=0).values)
                if "period" in h.dims
                else float(h.values)
            )
        lf = ds.get("land_fraction__finite_fraction")
        if lf is not None:
            lf_ff = (
                float(lf.isel(period=0).values)
                if "period" in lf.dims
                else float(lf.values)
            )
        ds.close()
        if h_ff < 0.95 or lf_ff < 0.95:
            rows.append(
                {
                    "source_id": r.source_id,
                    "experiment": r.experiment,
                    "variant_label": r.variant_label,
                    "output_zarr": r.output_zarr,
                    "native_grid_label": r.get("native_grid_label", "")
                    or r.get("grid_label", ""),
                    "HGTsfc_finite_fraction": h_ff,
                    "land_fraction_finite_fraction": lf_ff,
                }
            )
        if (i + 1) % 50 == 0:
            logging.info("  %d/%d scanned, %d stale", i + 1, n, len(rows))
    logging.info("Found %d datasets needing static refresh.", len(rows))
    return pd.DataFrame(rows)


def _build_fx_zstores(
    inventory: pd.DataFrame,
    source_id: str,
    day_grid_label: str,
    static_variables: list[str],
) -> dict[str, str]:
    """Mirror the (fixed) fx-zstore selection in ``process.select_datasets``.

    Prefer fx rows whose ``grid_label`` matches the day-table grid;
    fall through to any-grid only when the matching grid isn't
    published for a given variable.
    """
    stat = inventory[
        (inventory["table_id"] == "fx") & (inventory["source_id"] == source_id)
    ]
    if not len(stat):
        return {}
    same_grid = stat[stat["grid_label"] == day_grid_label]
    preferred = same_grid if len(same_grid) else stat
    preferred = preferred.sort_values("variant_p")
    out: dict[str, str] = {}
    for _, r in preferred.drop_duplicates("variable_id").iterrows():
        if r["variable_id"] in static_variables:
            out[r["variable_id"]] = r["zstore"]
    return out


def _refresh_one(
    zarr_path: str,
    source_id: str,
    experiment: str,
    variant_label: str,
    day_grid_label: str,
    inventory: pd.DataFrame,
    config: ProcessConfig,
    target: xr.Dataset,
    dry_run: bool,
) -> dict:
    cfg = config.resolve(source_id, experiment, variant_label)
    fx_zs = _build_fx_zstores(
        inventory, source_id, day_grid_label, cfg.static_variables
    )
    if not fx_zs:
        logging.warning(
            "  no fx zstores available for %s/%s/%s; skipping",
            source_id,
            experiment,
            variant_label,
        )
        return {"refreshed": [], "fx_zs": {}}

    logging.info(
        "  fx_zs (grid-aware, day_grid=%s): %s",
        day_grid_label,
        {v: u.rsplit("/", 4)[-2] for v, u in fx_zs.items()},
    )

    # Open + regrid each fx variable on its own source grid. Critically
    # the same per-variable regrid the new process.py uses — so the
    # refreshed values match a fresh-run zarr byte-for-byte. Apply the
    # ``orog`` ocean-fill fix (CMCC family) before regrid.
    static_pieces: list[xr.Dataset] = []
    orog_warnings: list[str] = []
    for v, url in fx_zs.items():
        fx_src = _open_zstore(url)[[v]].squeeze(drop=True)
        if v == "orog":
            fx_src, orog_warnings = fill_orog_ocean_with_zero(fx_src)
            for msg in orog_warnings:
                logging.info("  %s", msg)
        piece, _ = regrid_variables(fx_src, target, cfg)
        static_pieces.append(piece)
    static_regridded = xr.merge(static_pieces, compat="override")
    static_regridded, _ = clamp_static_fractions(static_regridded)
    static_renamed = apply_output_renames(static_regridded, CMIP_TO_OUTPUT_RENAMES)

    # Sanity: every output we expected should be present after rename.
    for v in _OUTPUT_STATIC_NAMES:
        if v in static_renamed.data_vars:
            arr = static_renamed[v]
            n_finite = int(np.isfinite(arr.values).sum())
            n_total = int(arr.size)
            logging.info(
                "    %s post-rename: %d/%d finite (%.3f)",
                v,
                n_finite,
                n_total,
                n_finite / n_total,
            )

    refreshed = list(static_renamed.data_vars)
    if dry_run:
        logging.info("  [dry-run] would replace %s in %s", refreshed, zarr_path)
        return {
            "refreshed": refreshed,
            "fx_zs": fx_zs,
            "orog_warnings": orog_warnings,
        }

    # In-place rewrite: delete stale arrays, append fresh ones, then
    # re-consolidate.
    group = zarr.open_group(zarr_path, mode="r+")
    for v in refreshed:
        if v in group:
            del group[v]
    # Load to dense numpy so ``to_zarr`` doesn't try to re-fetch the
    # source from dask (these are tiny: 45 × 90 float32 ≈ 16 KB each).
    static_renamed = static_renamed.load()
    static_renamed.to_zarr(zarr_path, mode="a", consolidated=False, zarr_format=3)
    zarr.consolidate_metadata(zarr_path)
    logging.info("  rewrote %s in %s", refreshed, zarr_path)
    return {
        "refreshed": refreshed,
        "fx_zs": fx_zs,
        "orog_warnings": orog_warnings,
    }


def _regenerate_stats(zarr_path: str, sidecar: dict, target_grid_name: str) -> None:
    stats_path = zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"
    identity = {
        "source_id": sidecar.get("source_id", ""),
        "experiment": sidecar.get("experiment", ""),
        "variant_label": sidecar.get("variant_label", ""),
        "label": sidecar.get("label", ""),
    }
    ds = xr.open_zarr(zarr_path, consolidated=True)
    try:
        compute_and_write_stats(
            ds,
            stats_path,
            identity,
            target_grid_name,
            periods=tuple(DEFAULT_STATS_PERIODS),
        )
    finally:
        ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pangeo process YAML config")
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=None,
        help="Restrict the scan to these source_ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect + log affected datasets but don't modify anything.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help=(
            "Skip the per-dataset compute_and_write_stats pass; only "
            "rewrite the static fields in the zarr. Stats will need "
            "to be regenerated separately (e.g. via Argo compute-stats "
            "with --source-ids + --force)."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ProcessConfig.from_file(args.config)
    out_dir = config.output_directory.rstrip("/")
    target_grid_name = config.defaults.target_grid.name

    stale = _scan_stale_statics(out_dir, args.source_ids)
    if not len(stale):
        logging.info("Nothing to refresh.")
        return

    inventory = _load_inventory(config.inventory_path)
    target = make_target_grid(target_grid_name)

    for _, r in stale.iterrows():
        logging.info(
            "Refreshing %s / %s / %s (HGTsfc=%.3f, land_fraction=%.3f)",
            r.source_id,
            r.experiment,
            r.variant_label,
            r.HGTsfc_finite_fraction,
            r.land_fraction_finite_fraction,
        )
        result = _refresh_one(
            r.output_zarr,
            r.source_id,
            r.experiment,
            r.variant_label,
            r.native_grid_label,
            inventory,
            config,
            target,
            args.dry_run,
        )
        if args.dry_run or not result.get("refreshed"):
            continue
        sidecar = _read_sidecar(r.output_zarr)
        if not args.no_stats:
            _regenerate_stats(r.output_zarr, sidecar, target_grid_name)
        sidecar.setdefault("migrations", []).append(
            {
                "kind": "static_refresh",
                "reason": (
                    "Re-regridded statics using grid-aware fx selection "
                    "(previous run merged mismatched-grid orog/sftlf) "
                    "and the orog ocean-fill fix (CMCC family)."
                ),
                "refreshed": result["refreshed"],
                "fx_zs": {v: u for v, u in result["fx_zs"].items()},
                "orog_warnings": result.get("orog_warnings", []),
            }
        )
        _write_sidecar(r.output_zarr, sidecar)
        logging.info("  sidecar updated.")


if __name__ == "__main__":
    main()
