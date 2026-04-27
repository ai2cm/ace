"""Inter-model consistency of climatology variance fraction (T6).

For each (variable, plev_index) we look at how ``clim_var_frac`` (the
share of total variance explained by the seasonal/spatial climatology,
as defined in ``compute_stats.py``) varies across models. The question
this answers is: do all models agree on whether a variable is
climatology-dominated or anomaly-dominated?

If yes, mean-removal is a uniform decision per variable. If no,
mean-removal helps very differently per model — and standardizing by
``std`` vs ``anom_std`` becomes a model-dependent choice.

Output: ``outputs/clim_var_frac.csv`` + ``outputs/clim_var_frac.md``
with median, IQR, min, max, and range across models.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("normalization.clim_var")


def _format(var: str, plev) -> str:
    if pd.isna(plev) or plev is None:
        return var
    return f"{var} (plev_index={int(plev)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", default="data/cmip6-daily-pilot/v0/stats.csv")
    parser.add_argument("--out-dir", default="normalization/outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pd.read_csv(args.stats)
    df = df[df["clim_var_frac"].notna()].copy()
    LOG.info("Loaded %d rows with clim_var_frac", len(df))

    # Collapse to per-label first so each model contributes one number.
    per_label = (
        df.groupby(["label", "variable", "plev_index"], dropna=False)["clim_var_frac"]
        .mean()
        .reset_index()
    )

    rows = []
    for (var, plev), g in per_label.groupby(["variable", "plev_index"], dropna=False):
        vals = g["clim_var_frac"].to_numpy()
        n = len(vals)
        if n < 2:
            continue
        med = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        rows.append(
            {
                "variable": var,
                "plev_index": plev,
                "n_models": n,
                "median": med,
                "iqr": float(q3 - q1),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "range": float(vals.max() - vals.min()),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(["range"], ascending=False)
        .reset_index(drop=True)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "clim_var_frac.csv", index=False)

    lines = ["# Climatology variance fraction across models", ""]
    lines.append(
        "Per-variable share of total variance explained by the seasonal "
        "+ spatial climatology, with dispersion across models. "
        "**clim_var_frac near 1** = mostly explained by the time-mean field "
        "(removing climatology helps a lot). **near 0** = anomalies dominate "
        "(climatology removal is nearly a no-op)."
    )
    lines.append("")
    lines.append(
        "Sorted by range across models — large range = the "
        "climatology/anomaly split *itself* is model-dependent."
    )
    lines.append("")
    lines.append("| variable | n | median | IQR | min | max | range |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in table.iterrows():
        lines.append(
            f"| {_format(r['variable'], r['plev_index'])} "
            f"| {int(r['n_models'])} "
            f"| {r['median']:.2f} "
            f"| {r['iqr']:.2f} "
            f"| {r['min']:.2f} "
            f"| {r['max']:.2f} "
            f"| {r['range']:.2f} |"
        )
    lines.append("")
    (out_dir / "clim_var_frac.md").write_text("\n".join(lines))
    LOG.info("Wrote %s", out_dir / "clim_var_frac.md")


if __name__ == "__main__":
    main()
