"""Inventory CMIP6 datasets available on ESGF.

Queries an ESGF search node for the configured variables x experiments x
table_ids, collects dataset-level metadata (source_id, member_id, grid),
and writes a tidy CSV. The output format matches the Pangeo inventory
(``inventory.py``) so both can be consumed by ``process_esgf.py``.

Unlike the Pangeo inventory, this does NOT record per-file download URLs
— those are resolved at processing time via file-level ESGF queries to
handle replica selection dynamically.

Usage:
    python inventory_esgf.py --config configs/inventory_esgf.yaml
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import ESGFInventoryConfig  # noqa: E402

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0
_REQUEST_DELAY = 0.3
_PAGE_SIZE = 500

_VARIANT_RE = re.compile(r"r(\d+)i(\d+)p(\d+)f(\d+)")


def _parse_variant(member_id: str) -> dict[str, int | None]:
    m = _VARIANT_RE.fullmatch(member_id or "")
    if not m:
        return {"r": None, "i": None, "p": None, "f": None}
    return {
        "r": int(m.group(1)),
        "i": int(m.group(2)),
        "p": int(m.group(3)),
        "f": int(m.group(4)),
    }


def _esgf_search(node: str, params: dict, timeout: int = 60) -> dict:
    params.setdefault("format", "application/solr+json")
    url = node + "?" + urllib.parse.urlencode(params)
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"ESGF query failed after {_MAX_RETRIES} attempts: {e}"
                ) from e
            time.sleep(_RETRY_DELAY)
    raise AssertionError("unreachable")


def _paginate_datasets(
    node: str,
    base_params: dict,
) -> list[dict]:
    """Paginate through ESGF Dataset results, returning all docs."""
    all_docs: list[dict] = []
    offset = 0
    while True:
        params = {**base_params, "offset": offset, "limit": _PAGE_SIZE}
        data = _esgf_search(node, params)
        docs = data.get("response", {}).get("docs", [])
        num_found = data.get("response", {}).get("numFound", 0)
        all_docs.extend(docs)
        offset += len(docs)
        if offset >= num_found or not docs:
            break
        time.sleep(_REQUEST_DELAY)
    return all_docs


def query_datasets(config: ESGFInventoryConfig) -> pd.DataFrame:
    """Query ESGF for all matching datasets and return a tidy DataFrame."""
    node = config.search_node
    rows: list[dict] = []

    for query in config.queries:
        experiments: list[str | None] = (
            list(config.experiments) if query.filter_by_experiment else [None]
        )
        for variable in query.variables:
            for experiment in experiments:
                base: dict = {
                    "type": "Dataset",
                    "project": "CMIP6",
                    "table_id": query.table_id,
                    "variable_id": variable,
                    "fields": (
                        "source_id,experiment_id,member_id,"
                        "grid_label,activity_id,instance_id"
                    ),
                }
                if experiment is not None:
                    base["experiment_id"] = experiment

                try:
                    docs = _paginate_datasets(node, base)
                except RuntimeError as e:
                    logging.warning(
                        "  SKIP %s/%s/%s: %s",
                        query.table_id,
                        variable,
                        experiment,
                        e,
                    )
                    continue

                seen: set[tuple] = set()
                for doc in docs:
                    source_id = doc.get("source_id", [None])[0]
                    exp_id = doc.get("experiment_id", [None])[0]
                    member_id = doc.get("member_id", [None])[0]
                    grid_label = doc.get("grid_label", [None])[0]
                    activity_id = doc.get("activity_id", [None])[0]
                    if not all([source_id, exp_id, member_id, grid_label]):
                        continue
                    key = (source_id, exp_id, member_id, grid_label)
                    if key in seen:
                        continue
                    seen.add(key)

                    parsed = _parse_variant(member_id)
                    rows.append(
                        {
                            "activity_id": activity_id or "",
                            "source_id": source_id,
                            "experiment_id": exp_id,
                            "member_id": member_id,
                            "table_id": query.table_id,
                            "variable_id": variable,
                            "grid_label": grid_label,
                            "variant_r": parsed["r"],
                            "variant_i": parsed["i"],
                            "variant_p": parsed["p"],
                            "variant_f": parsed["f"],
                        }
                    )

                logging.info(
                    "  %-8s %-10s %-12s: %d unique datasets (%d docs)",
                    query.table_id,
                    variable,
                    experiment or "(all)",
                    len(seen),
                    len(docs),
                )
                time.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates()
    logging.info("Total: %d rows", len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to ESGF inventory YAML")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ESGFInventoryConfig.from_file(args.config)
    logging.info("ESGF search node: %s", config.search_node)

    df = query_datasets(config)
    df.to_csv(config.output_path, index=False)
    logging.info("Wrote %d rows to %s", len(df), config.output_path)


if __name__ == "__main__":
    main()
