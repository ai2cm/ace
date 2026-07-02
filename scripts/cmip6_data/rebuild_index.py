"""Rebuild the central index from per-dataset metadata.json sidecars.

Scans ``<output_directory>`` for all ``metadata.json`` files written by
``process.py`` and writes a fresh ``index.csv`` / ``index.parquet``.
Useful after parallel Argo runs where each pod processes a subset of
datasets independently.

Usage:
    python rebuild_index.py --config configs/pilot.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import ProcessConfig  # noqa: E402
from index import write_index  # noqa: E402
from process import _scan_all_sidecars  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the process YAML (pilot.yaml)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    rows = _scan_all_sidecars(cfg.output_directory)
    rows.sort(key=lambda r: (r.source_id, r.experiment, r.variant_label))
    write_index(rows, cfg.output_directory)
    logging.info("Rebuilt index with %d datasets.", len(rows))


if __name__ == "__main__":
    main()
