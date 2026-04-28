"""Build the per-dataset presence table from ``index.csv``.

Reads the central index produced by ``process.py`` and the inventory
that fed the run, and emits three views into
``<output_directory>``:

- ``presence.csv`` (and ``presence.parquet`` when pyarrow is
  installed) — wide pivot, one row per attempted dataset, one
  column per variable. Cell encoding:

  * ``2`` = written to the dataset's zarr
  * ``1`` = available in the source inventory but not written
    (the dataset is skipped or failed; the cell tells you which
    variables the source had even though we couldn't ingest)
  * ``0`` = the source publication didn't have the variable

- ``presence.png`` — heatmap version of the same matrix, with rows
  sorted by ``(source_id, experiment, variant_label)`` and columns
  grouped by category (core → derived → forcing → static →
  optional). Datasets' status (ok / skipped / failed) is shown as a
  side stripe.

- ``presence.md`` — per-model rollup with a one-line summary for
  every dataset and a compact category-level coverage table.

Usage:
    python make_presence.py --config configs/pilot.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    CORE_VARIABLES,
    FORCING_VARIABLES,
    OPTIONAL_VARIABLES,
    STATIC_VARIABLES,
    ProcessConfig,
)

# Variables that aren't in any of the user-facing lists but appear in
# the produced zarr — derived T layers and the masks/static fields.
# These are conceptual names for the presence table columns; the zarr
# stores pressure-named variants (ua1000, ta_derived_layer_1000_850, etc.).
DERIVED_VARIABLES = ["ta_derived_layer"]
EXTRA_VARIABLES = ["below_surface_mask", "siconc_mask"]

_PLEV8_HPA = [1000, 850, 700, 500, 250, 100, 50, 10]
_3D_CORE = {"ua", "va", "hus", "zg"}


def _flattened_forms(var: str) -> list[str]:
    """Return the pressure-named zarr variable(s) for a conceptual variable."""
    if var in _3D_CORE or var == "below_surface_mask":
        return [f"{var}{p}" for p in _PLEV8_HPA]
    if var == "ta_derived_layer":
        return [
            f"ta_derived_layer_{_PLEV8_HPA[i]}_{_PLEV8_HPA[i+1]}"
            for i in range(len(_PLEV8_HPA) - 1)
        ]
    return [var]


def _categorised_variables() -> dict[str, list[str]]:
    """Variable display order grouped by category. Used both for the
    heatmap column ordering and the markdown rollup.
    """
    return {
        "core": list(CORE_VARIABLES),
        "derived": list(DERIVED_VARIABLES),
        "forcing": list(FORCING_VARIABLES),
        "static": list(STATIC_VARIABLES),
        "extra": list(EXTRA_VARIABLES),
        "optional": list(OPTIONAL_VARIABLES),
    }


def _all_variables() -> list[str]:
    out: list[str] = []
    for vs in _categorised_variables().values():
        out.extend(vs)
    return out


# ---------------------------------------------------------------------------
# Build the per-dataset presence matrix
# ---------------------------------------------------------------------------


def _available_from_inventory(
    inv: pd.DataFrame,
) -> dict[tuple[str, str, str], set[str]]:
    """For each ``(source_id, experiment, member_id)`` in the inventory,
    the set of CMIP6 variable_ids the publisher had — combined across
    the day, Amon, SImon, and fx tables. Static fields (fx) aren't
    keyed by member, so we fold them in per source_id.
    """
    out: dict[tuple[str, str, str], set[str]] = {}
    rel = inv[inv["table_id"].isin(("day", "Amon", "SImon"))]
    for (s, e, m), g in rel.groupby(["source_id", "experiment_id", "member_id"]):
        out[(s, e, m)] = set(g["variable_id"].unique())
    fx = inv[inv["table_id"] == "fx"]
    fx_by_src = fx.groupby("source_id")["variable_id"].agg(set).to_dict()
    keys_by_src: dict[str, list[tuple[str, str, str]]] = {}
    for k in out:
        keys_by_src.setdefault(k[0], []).append(k)
    for src, fx_vars in fx_by_src.items():
        for k in keys_by_src.get(src, []):
            out[k] = out[k] | fx_vars
    return out


def _present_from_index(row: pd.Series) -> set[str]:
    s = row.get("variables_present", "")
    if not isinstance(s, str) or not s:
        return set()
    try:
        return set(json.loads(s))
    except json.JSONDecodeError:
        return set()


# Variables we synthesise during processing — present iff the dataset
# was ingested ok and the source had the inputs that feed them.
_DERIVED_INPUTS = {"zg", "hus"}
_BELOW_MASK_INPUTS = {"ua", "va", "hus", "zg"}


def _expected_extras(available: set[str]) -> set[str]:
    """For a non-ok dataset, what derived/extra variables would have
    been produced if processing succeeded?
    """
    extras: set[str] = set()
    if _DERIVED_INPUTS.issubset(available):
        extras |= set(DERIVED_VARIABLES)
    if _BELOW_MASK_INPUTS & available:
        extras.add("below_surface_mask")
    if "siconc" in available:
        extras.add("siconc_mask")
    return extras


def build_presence_table(
    index: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    available = _available_from_inventory(inventory)
    var_order = _all_variables()
    rows = []
    for _, r in index.iterrows():
        key = (r["source_id"], r["experiment"], r["variant_label"])
        avail = set(available.get(key, set()))
        avail |= _expected_extras(avail)  # derived + masks if inputs present
        present = _present_from_index(r)
        record: dict[str, object] = {
            "source_id": r["source_id"],
            "experiment": r["experiment"],
            "variant_label": r["variant_label"],
            "label": r.get("label", ""),
            "status": r["status"],
            "skip_reason": r.get("skip_reason", ""),
            "n_timesteps": int(r.get("n_timesteps", 0) or 0),
            "time_start": r.get("time_start", ""),
            "time_end": r.get("time_end", ""),
            "native_grid_label": r.get("native_grid_label", ""),
            "native_calendar": r.get("native_calendar", ""),
            "mask_source": r.get("mask_source", ""),
            "n_warnings": len(json.loads(r.get("warnings", "[]") or "[]")),
            "output_zarr": r.get("output_zarr", ""),
        }
        for v in var_order:
            if any(f in present for f in _flattened_forms(v)):
                record[v] = 2
            elif v in avail:
                record[v] = 1
            else:
                record[v] = 0
        rows.append(record)
    out = pd.DataFrame(rows)
    return out.sort_values(["source_id", "experiment", "variant_label"]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Heatmap PNG
# ---------------------------------------------------------------------------


def write_heatmap(presence: pd.DataFrame, path: str) -> None:
    """Render the presence matrix as a PNG.

    Three-tone heatmap (gray / amber / green), rows annotated with
    dataset status as a side stripe.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cats = _categorised_variables()
    var_order = _all_variables()
    matrix = presence[var_order].astype(int).to_numpy()

    n_rows, n_cols = matrix.shape
    height = max(6.0, 0.18 * n_rows + 1.5)
    width = max(10.0, 0.30 * n_cols + 4.0)
    fig, (ax, ax_status) = plt.subplots(
        1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [n_cols, 1]}
    )

    cmap = ListedColormap(["#dddddd", "#f5b942", "#3aa845"])  # gray, amber, green
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    # X-axis: variable names, with category-group separators.
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(var_order, rotation=90, fontsize=7)
    cum = 0
    for cat, vs in cats.items():
        if not vs:
            continue
        ax.axvline(cum - 0.5, color="white", linewidth=2)
        ax.text(
            cum + len(vs) / 2 - 0.5,
            -2.5,
            cat,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        cum += len(vs)

    # Y-axis: dataset labels.
    labels = [
        f"{r.source_id}/{r.experiment}/{r.variant_label}"
        for r in presence.itertuples(index=False)
    ]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Status side strip.
    status_color = {"ok": "#3aa845", "skipped": "#f5b942", "failed": "#cc3333"}
    status_palette = ListedColormap(
        [status_color.get(s, "#888888") for s in presence.status]
    )
    ax_status.imshow(
        np.arange(n_rows).reshape(-1, 1),
        aspect="auto",
        cmap=status_palette,
        interpolation="nearest",
    )
    ax_status.set_xticks([0])
    ax_status.set_xticklabels(["status"], fontsize=8)
    ax_status.set_yticks([])
    ax_status.set_ylim(n_rows - 0.5, -0.5)

    # Legend.
    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor="#3aa845", label="written"),
        Patch(facecolor="#f5b942", label="available, not written"),
        Patch(facecolor="#dddddd", label="not in source"),
        Patch(facecolor="#cc3333", label="failed"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=7)

    fig.suptitle("CMIP6 daily pilot — variable presence by dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote heatmap %s", path)


# ---------------------------------------------------------------------------
# Markdown rollup
# ---------------------------------------------------------------------------


def write_markdown(presence: pd.DataFrame, path: str) -> None:
    cats = _categorised_variables()
    var_order = _all_variables()

    lines: list[str] = ["# CMIP6 daily pilot — dataset presence", ""]
    n = len(presence)
    n_ok = (presence.status == "ok").sum()
    n_sk = (presence.status == "skipped").sum()
    n_fl = (presence.status == "failed").sum()
    lines.append(f"Total datasets: **{n}**  ok={n_ok}  skipped={n_sk}  failed={n_fl}\n")

    # Per-model summary section.
    for source_id, group in presence.groupby("source_id"):
        n_ok = (group.status == "ok").sum()
        n_sk = (group.status == "skipped").sum()
        n_fl = (group.status == "failed").sum()
        lines.append(f"## {source_id}  ({n_ok} ok / {n_sk} skipped / {n_fl} failed)")
        for _, r in group.iterrows():
            stamp = f"{r.experiment} {r.variant_label}"
            label = r.label
            present_n = sum(int(r[v]) == 2 for v in var_order)
            avail_n = sum(int(r[v]) >= 1 for v in var_order)
            if r.status == "ok":
                lines.append(
                    f"- **{stamp}** (`{label}`) — ok, {present_n} variables written"
                )
            else:
                lines.append(
                    f"- **{stamp}** (`{label}`) — {r.status}: {r.skip_reason} "
                    f"(source had {avail_n} of the relevant variables)"
                )
        lines.append("")

    # Category coverage matrix: rows = source_id, cols = category, value = how many
    # ok datasets have all vars in that category.
    lines.append("## Category coverage by source_id (ok datasets only)")
    lines.append("")
    header = "| source_id | n ok | " + " | ".join(cats.keys()) + " |"
    sep = "|" + "|".join("---" for _ in range(len(cats) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for source_id, group in presence.groupby("source_id"):
        ok_group = group[group.status == "ok"]
        if len(ok_group) == 0:
            cells = ["0"] + ["—"] * len(cats)
        else:
            cells = [str(len(ok_group))]
            for cat, vs in cats.items():
                if not vs:
                    cells.append("—")
                    continue
                # For each category, % of cells written across its ok datasets.
                count = 0
                total = 0
                for v in vs:
                    for _, r in ok_group.iterrows():
                        total += 1
                        if int(r[v]) == 2:
                            count += 1
                cells.append(f"{count}/{total}")
        lines.append(f"| {source_id} | " + " | ".join(cells) + " |")
    lines.append("")
    Path(path).write_text("\n".join(lines))
    logging.info("Wrote markdown %s", path)


# ---------------------------------------------------------------------------
# CSV / parquet writer
# ---------------------------------------------------------------------------


def write_table(presence: pd.DataFrame, root: str) -> None:
    csv_path = f"{root.rstrip('/')}/presence.csv"
    parquet_path = f"{root.rstrip('/')}/presence.parquet"
    presence.to_csv(csv_path, index=False)
    logging.info("Wrote %s (%d rows)", csv_path, len(presence))
    try:
        presence.to_parquet(parquet_path, index=False)
        logging.info("Wrote %s", parquet_path)
    except ImportError:
        logging.warning("Parquet engine not installed; skipping %s", parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the process YAML (pilot.yaml)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ProcessConfig.from_file(args.config)
    inv_path = cfg.inventory_path
    out_dir = cfg.output_directory.rstrip("/")
    index_path = f"{out_dir}/index.csv"

    inv = pd.read_csv(inv_path)
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"Could not find {index_path}. Run process.py first to produce the index."
        )
    idx = pd.read_csv(index_path)

    presence = build_presence_table(idx, inv)
    write_table(presence, out_dir)
    write_heatmap(presence, f"{out_dir}/presence.png")
    write_markdown(presence, f"{out_dir}/presence.md")


if __name__ == "__main__":
    main()
