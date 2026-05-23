"""Apply registered schema migrations across an existing CMIP6 archive.

Each dataset's ``metadata.json`` sidecar carries a ``schema_version``
field. ``migrate.py`` walks the index, finds datasets at an older
version than the current ``SCHEMA_VERSION``, and applies the
registered migration chain in order. Datasets predating this
framework — sidecars without the field — are treated as ``"0.0.0"``.

Usage::

    python migrate.py --config configs/<process_config>.yaml
    python migrate.py --config ... --dry-run             # plan only
    python migrate.py --config ... --workers 4           # parallel
    python migrate.py --config ... --source-ids X Y Z    # subset

The script writes the updated sidecar in place after each migration
step. If a step fails partway, the sidecar still reflects the most
recently completed step — re-running the script picks up where it
left off.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import fsspec
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig  # noqa: E402
from migrations import chain_for  # noqa: E402
from schema_version import SCHEMA_VERSION, version_lt  # noqa: E402


def _load_sidecar(zarr_path: str) -> Optional[dict]:
    """Read the ``metadata.json`` sidecar at ``<zarr_path>/metadata.json``.
    Returns ``None`` if missing (dataset has no ``status=ok`` sidecar to
    migrate).
    """
    sidecar_url = f"{zarr_path.rstrip('/')}/metadata.json"
    fs, rel = fsspec.core.url_to_fs(sidecar_url)
    if not fs.exists(rel):
        return None
    with fs.open(rel, "r") as f:
        return json.load(f)


def _write_sidecar(zarr_path: str, sidecar: dict) -> None:
    """Overwrite the ``metadata.json`` sidecar."""
    sidecar_url = f"{zarr_path.rstrip('/')}/metadata.json"
    fs, rel = fsspec.core.url_to_fs(sidecar_url)
    with fs.open(rel, "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True, default=str)


def migrate_one(
    zarr_path: str,
    target_version: str = SCHEMA_VERSION,
    *,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Migrate a single dataset to ``target_version``.

    Returns ``(status, detail)`` where ``status`` is one of:

    * ``"current"`` — already at ``target_version``, nothing to do.
    * ``"missing"`` — no sidecar at this path (skipped).
    * ``"migrated"`` — chain ran successfully; ``detail`` lists the
      versions touched.
    * ``"error"`` — chain raised; ``detail`` is the traceback.
    * ``"would-migrate"`` (dry-run only) — what *would* be applied.
    """
    sidecar = _load_sidecar(zarr_path)
    if sidecar is None:
        return "missing", zarr_path

    current = sidecar.get("schema_version", "0.0.0")
    if not version_lt(current, target_version):
        return "current", f"{zarr_path} already at {current}"

    try:
        chain = chain_for(current, target_version)
    except RuntimeError as e:
        return "error", f"{zarr_path}: {e}"

    if dry_run:
        steps = ", ".join(f"{m.from_version}→{m.to_version}" for m in chain)
        return (
            "would-migrate",
            f"{zarr_path}: {current} → {target_version} via [{steps}]",
        )

    applied: list[str] = []
    for migration in chain:
        try:
            sidecar = migration.apply(zarr_path, sidecar)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            # Persist whatever progress we made before the failure so
            # the next run resumes from the right spot.
            _write_sidecar(zarr_path, sidecar)
            step = f"{migration.from_version}→{migration.to_version}"
            return (
                "error",
                f"{zarr_path}: {step} failed: {e}\n{tb}",
            )
        applied.append(f"{migration.from_version}→{migration.to_version}")
        # Persist after each step so a partial failure leaves a
        # consistent on-disk state.
        _write_sidecar(zarr_path, sidecar)

    return "migrated", f"{zarr_path}: applied [{', '.join(applied)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Process YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan migrations without writing anything.",
    )
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=None,
        help="Restrict to a subset of source_ids.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes. Defaults to the pod's "
            "CPU count. Use 1 for easier debugging."
        ),
    )
    parser.add_argument(
        "--target-version",
        default=SCHEMA_VERSION,
        help=(
            f"Migrate to this version instead of the current "
            f"SCHEMA_VERSION ({SCHEMA_VERSION}). Mostly useful for "
            "step-by-step debugging."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    out_dir = cfg.output_directory.rstrip("/")
    index_path = f"{out_dir}/index.csv"
    fs, rel = fsspec.core.url_to_fs(index_path)
    if not fs.exists(rel):
        raise FileNotFoundError(f"{index_path} not found; run process.py first")
    idx = pd.read_csv(index_path)
    ok = idx[idx.status == "ok"].reset_index(drop=True)
    if args.source_ids:
        ok = ok[ok.source_id.isin(args.source_ids)].reset_index(drop=True)

    zarr_paths = ok["output_zarr"].tolist()
    logging.info(
        "Found %d ok datasets in index; migrating to %s",
        len(zarr_paths),
        args.target_version,
    )

    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(zarr_paths))) if zarr_paths else 1

    summary: dict[str, int] = {
        "current": 0,
        "missing": 0,
        "migrated": 0,
        "would-migrate": 0,
        "error": 0,
    }
    if workers == 1:
        for i, p in enumerate(zarr_paths):
            status, detail = migrate_one(
                p, target_version=args.target_version, dry_run=args.dry_run
            )
            summary[status] = summary.get(status, 0) + 1
            logging.info("[%d/%d] %s: %s", i + 1, len(zarr_paths), status, detail)
    else:
        # ``spawn`` to dodge the gcsfs/gRPC fork issue (same reason
        # ``compute_stats.py`` uses it).
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    migrate_one,
                    p,
                    target_version=args.target_version,
                    dry_run=args.dry_run,
                ): p
                for p in zarr_paths
            }
            for i, fut in enumerate(as_completed(futures)):
                status, detail = fut.result()
                summary[status] = summary.get(status, 0) + 1
                logging.info("[%d/%d] %s: %s", i + 1, len(zarr_paths), status, detail)

    logging.info("Migration summary: %s", summary)
    if summary["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
