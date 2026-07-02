"""Aggregate v2 stats and produce coverage + climatology plots.

v2 schema (SCHEMA_VERSION 0.4.0) adds:
- ``period`` coord on every scalar (we ``.isel(period=...)`` before
  flattening to one row per (dataset, var, optional plev) — default
  period is ``full``)
- 3D plev vars: ``{var}__{stat}`` has shape ``(period, plev)``; we
  emit one column per plev as ``{var}{hPa}__{stat}``
- per-cell maps (``{var}__time_mean_map`` etc.) — skipped, they
  aren't useful as cohort scalars

Run from ``scripts/cmip6_data/`` after ``mkdir -p report/v2/plots``.
"""

from __future__ import annotations

import ast
import logging
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("report-v2")

BUCKET = (
    "gs://vcm-ml-intermediate/2026-05-22-cmip6-multimodel-daily-4deg-8plev-1940-2100"
)
RUN = f"{BUCKET}/v2"
PLOTS = Path(__file__).parent / "v2" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PERIOD = "full"

# Scalar stats we want to aggregate as one column per variable
# (plus per-plev for 3D vars). Maps are excluded.
SCALAR_SUFFIXES = (
    "__mean",
    "__std",
    "__clim_std",
    "__anom_std",
    "__clim_var_frac",
    "__d1_mean",
    "__d1_std",
    "__p01",
    "__p50",
    "__p99",
    "__skewness",
    "__kurtosis",
    "__autocorr_lag1",
    "__finite_fraction",
)

# Suffixes of per-cell map variables — we skip these entirely when
# flattening to scalars.
MAP_SUFFIXES = (
    "__time_mean_map",
    "__time_var_map",
    "__n_valid_map",
    "__d1_var_map",
    "__static_map",
)


def load_index() -> pd.DataFrame:
    df = pd.read_csv(f"{RUN}/index.csv")
    df = df[df.status == "ok"].reset_index(drop=True)
    for col in ("variables_present", "warnings", "cell_methods_mismatch"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df


def _stats_path(zarr_path: str) -> str:
    return zarr_path.rstrip("/").rsplit("/", 1)[0] + "/stats.nc"


def _flatten_stats(ds: xr.Dataset, period: str) -> dict:
    """Pick ``period`` from each scalar variable and emit one column
    per (var, optional plev). Maps are dropped."""
    out: dict = {}
    # period index
    if "period" in ds.coords:
        labels = [str(p) for p in ds["period"].values]
        try:
            pidx = labels.index(period)
        except ValueError:
            return out
    else:
        pidx = None

    plev_vals = ds["plev"].values if "plev" in ds.coords else None

    for v in ds.data_vars:
        if v.endswith(MAP_SUFFIXES):
            continue
        if not any(v.endswith(s) for s in SCALAR_SUFFIXES):
            continue
        da = ds[v]
        # Skip non-numeric / period_* attribute strings.
        if da.dtype.kind == "O":
            continue
        # Extract per-period slice.
        if "period" in da.dims and pidx is not None:
            da = da.isel(period=pidx)
        # Plev fan-out for 3D scalars.
        if "plev" in da.dims and plev_vals is not None:
            for k, hpa in enumerate(plev_vals):
                # Variable name embedding: split base/suffix and stick
                # hPa label after base.
                base, sep, suffix = v.rpartition("__")
                hpa_lab = int(round(float(hpa) / 100.0))
                col = f"{base}{hpa_lab}__{suffix}"
                try:
                    out[col] = float(da.isel(plev=k).values)
                except Exception:
                    out[col] = float("nan")
        else:
            try:
                out[v] = float(da.values)
            except Exception:
                out[v] = float("nan")
    return out


def _load_one_stats(row, period: str) -> tuple[dict, str | None]:
    try:
        import tempfile

        import fsspec

        path = _stats_path(row.output_zarr)
        fs, rel = fsspec.core.url_to_fs(path)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            fs.get(rel, tmp.name)
            ds = xr.open_dataset(tmp.name).load()
            ds.close()
        out = {
            "source_id": row.source_id,
            "experiment": row.experiment,
            "variant_label": row.variant_label,
            "label": row.label,
        }
        out.update(_flatten_stats(ds, period))
        return out, None
    except Exception as e:
        return {}, f"{row.source_id}/{row.experiment}/{row.variant_label}: {e}"


def aggregate_stats(df: pd.DataFrame, period: str) -> pd.DataFrame:
    log.info("Aggregating stats from %d datasets (period=%s)...", len(df), period)
    records: list[dict] = []
    errors: list[str] = []
    # ThreadPoolExecutor with 8+ workers segfaults — gcsfs's
    # concurrent download path interacts badly with something in this
    # process (likely h5netcdf + gcsfs both holding native state).
    # Stick to serial; per-file load is ~1.5s, so 246 files take ~6 min.
    workers = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_load_one_stats, row, period) for _, row in df.iterrows()]
        for i, fut in enumerate(as_completed(futs)):
            rec, err = fut.result()
            if err:
                errors.append(err)
            else:
                records.append(rec)
            if (i + 1) % 50 == 0:
                log.info("  %d/%d", i + 1, len(df))
    log.info("Errors: %d", len(errors))
    for e in errors[:5]:
        log.info("  %s", e)
    return pd.DataFrame.from_records(records)


def plot_coverage(df: pd.DataFrame):
    pivot = df.groupby(["source_id", "experiment"]).size().unstack(fill_value=0)
    # Order columns to match v2 scope.
    order = [c for c in ("historical", "ssp245", "ssp585") if c in pivot.columns]
    pivot = pivot.loc[:, order]
    pivot = pivot.sort_values("historical", ascending=False)
    fig, ax = plt.subplots(figsize=(6, max(8, 0.18 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                pivot.iloc[i, j],
                ha="center",
                va="center",
                color="white" if pivot.iloc[i, j] < pivot.values.max() / 2 else "black",
                fontsize=7,
            )
    plt.colorbar(im, ax=ax, label="# variants")
    ax.set_title("v2 coverage: variants per (model, experiment)")
    plt.tight_layout()
    plt.savefig(PLOTS / "coverage_matrix.png", dpi=110)
    plt.close()
    log.info("Wrote coverage_matrix.png")
    return pivot


def plot_variable_presence(df: pd.DataFrame):
    """Per-source, what variables are present?"""
    rows = []
    for _, r in df.iterrows():
        for v in r.variables_present:
            rows.append(
                {"source_id": r.source_id, "experiment": r.experiment, "variable": v}
            )
    long = pd.DataFrame(rows)
    # Per source × variable presence rate across its variants.
    pivot = long.groupby(["source_id", "variable"]).size().unstack(fill_value=0)
    # Cap at 1 (presence indicator).
    pivot = (pivot > 0).astype(int)
    # Drop columns where all sources have it (uninformative).
    keep = pivot.columns[(pivot.sum(axis=0) < len(pivot))]
    if len(keep) == 0:
        log.info("All variables present in all sources; skipping presence plot.")
        return
    pivot = pivot.loc[:, keep]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(8, 0.2 * len(keep)), max(8, 0.18 * len(pivot))))
    ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_title("Variable presence (per source); excludes universal vars")
    plt.tight_layout()
    plt.savefig(PLOTS / "variable_presence.png", dpi=110)
    plt.close()
    log.info("Wrote variable_presence.png")


def plot_finite_fraction(stats_df: pd.DataFrame):
    cols = [c for c in stats_df.columns if c.endswith("__finite_fraction")]
    cols = sorted(cols, key=lambda c: stats_df[c].mean())
    fig, ax = plt.subplots(figsize=(10, max(8, 0.15 * len(cols))))
    data = [stats_df[c].dropna().values for c in cols]
    labels = [c.replace("__finite_fraction", "") for c in cols]
    bp = ax.boxplot(
        data, vert=False, labels=labels, showfliers=True, patch_artist=True, widths=0.6
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.axvline(1.0, color="green", linestyle="--", alpha=0.5, label="all-finite")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50% finite")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("finite fraction across the dataset")
    ax.set_title("Finite-cell fraction by variable (v2)")
    ax.tick_params(axis="y", labelsize=5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "finite_fraction.png", dpi=110)
    plt.close()
    log.info("Wrote finite_fraction.png")


def plot_outlier_table(stats_df: pd.DataFrame):
    """Flag datasets >3σ from the cohort mean for key vars."""
    key_vars = [
        "tas",
        "amon_ts",
        "psl",
        "pr",
        "rlut",
        "huss",
        "ua250",
        "h500",
        "luh2_forest",
        "input4mips_co2",
        "log_input4mips_co2",
        "input4mips_so2",
        "DSWRFsfc",
        "DLWRFsfc",
        "TMP2m",
    ]
    rows = []
    for var in key_vars:
        col = f"{var}__mean"
        if col not in stats_df.columns:
            continue
        vals = stats_df[col].dropna()
        if len(vals) < 5:
            continue
        mu, sd = vals.mean(), vals.std()
        for idx, v in vals.items():
            z = (v - mu) / sd if sd > 0 else 0
            if abs(z) > 3:
                rows.append(
                    {
                        "variable": var,
                        "source_id": stats_df.loc[idx, "source_id"],
                        "experiment": stats_df.loc[idx, "experiment"],
                        "variant_label": stats_df.loc[idx, "variant_label"],
                        "value": v,
                        "cohort_mean": mu,
                        "z": z,
                    }
                )
    out = pd.DataFrame(rows).sort_values("z", key=lambda s: s.abs(), ascending=False)
    out.to_csv(PLOTS.parent / "outliers.csv", index=False)
    log.info("Wrote outliers.csv (%d rows)", len(out))
    return out


def plot_units_sanity(stats_df: pd.DataFrame):
    """Histogram per-dataset means of temperature-like vars — check
    for un-converted °C, scaling bugs, etc."""
    candidates = ["TMP2m", "amon_ts", "tas", "tos", "oday_tos", "simon_sitemptop"]
    avail = [v for v in candidates if f"{v}__mean" in stats_df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 4), sharey=True)
    if len(avail) == 1:
        axes = [axes]
    for ax, v in zip(axes, avail):
        vals = stats_df[f"{v}__mean"].dropna()
        ax.hist(vals, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(f"{v} mean (n={len(vals)})")
        ax.set_xlabel("K")
        ax.axvline(273.15, color="gray", linestyle="--", alpha=0.5, label="0 °C")
        bad = (vals < 100).sum()
        if bad:
            ax.set_title(f"{v}: {bad} datasets < 100K (!)")
    axes[0].set_ylabel("# datasets")
    plt.tight_layout()
    plt.savefig(PLOTS / "temperature_units.png", dpi=110)
    plt.close()
    log.info("Wrote temperature_units.png")


def plot_crossmodel_stats(stats_df: pd.DataFrame):
    """For each key var, a strip plot of per-dataset means across models."""
    key_vars = [
        "TMP2m",
        "amon_ts",
        "psl",
        "pr",
        "rlut",
        "ua250",
        "h500",
        "luh2_forest",
        "input4mips_co2",
        "log_input4mips_co2",
        "DSWRFsfc",
        "DLWRFsfc",
    ]
    present = [v for v in key_vars if f"{v}__mean" in stats_df.columns]
    if not present:
        return
    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, present):
        col = f"{v}__mean"
        order = stats_df.groupby("source_id")[col].mean().sort_values().index
        for i, src in enumerate(order):
            ys = stats_df[stats_df.source_id == src][col].dropna().values
            xs = np.full_like(ys, i, dtype=float) + np.random.uniform(
                -0.25, 0.25, size=len(ys)
            )
            ax.scatter(xs, ys, s=8, alpha=0.7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=5)
        ax.set_title(v, fontsize=9)
    for ax in axes[len(present) :]:
        ax.set_visible(False)
    plt.tight_layout()
    # 90 dpi keeps the file under the 250 KB pre-commit large-file
    # gate while still legible — this is the only plot whose 18
    # panels push it past the limit at the default 110.
    plt.savefig(PLOTS / "crossmodel_means.png", dpi=90)
    plt.close()
    log.info("Wrote crossmodel_means.png")


def plot_mask_coverage(df: pd.DataFrame):
    """Two-panel plot: ``mask_source`` distribution + the
    ``n_nan_input_cells`` outlier tail. The mask_source counts surface
    how the below-surface mask got derived for each dataset (NaN union
    from publisher, orog fallback, or no 3D state). The outlier tail
    surfaces publishers that emit whole-slab NaN volumes that the
    nearest-above fill has to mop up — useful to flag for manual
    inspection before training.
    """
    if "mask_source" not in df.columns or "n_nan_input_cells" not in df.columns:
        return
    counts = df.mask_source.value_counts()
    top_nan = df.nlargest(15, "n_nan_input_cells").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * len(top_nan))))

    ax = axes[0]
    ax.barh(range(len(counts)), counts.values, color="steelblue")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("# datasets")
    ax.set_title("mask_source distribution")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f"  {v}", va="center")

    ax = axes[1]
    labels = [
        f"{r.source_id}/{r.experiment}/{r.variant_label}" for _, r in top_nan.iterrows()
    ]
    ax.barh(range(len(top_nan)), top_nan.n_nan_input_cells.values, color="indianred")
    ax.set_yticks(range(len(top_nan)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("symlog")
    ax.set_xlabel("# NaN cells in input 3D state (pre-fill)")
    ax.set_title("Top n_nan_input_cells")
    for i, v in enumerate(top_nan.n_nan_input_cells.values):
        ax.text(v, i, f"  {int(v):,}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(PLOTS / "mask_coverage.png", dpi=110)
    plt.close()
    log.info("Wrote mask_coverage.png")


def plot_warming_response(stats_df: pd.DataFrame):
    """Per-source ssp585 - historical scalar deltas for warming-
    response and hydrological-response variables.

    For each ``source_id`` with both ``historical`` and ``ssp585``
    datasets, take the ensemble mean of each per-dataset scalar
    inside the experiment, then plot ``ssp585_mean − historical_mean``
    as a horizontal bar. Ordering by ΔTMP2m makes the climate-
    sensitivity pattern fall out visually: high-ECS models pile at
    the top, low-ECS at the bottom.

    Variables shown:

    - ``TMP2m`` (near-surface air temperature)
    - ``amon_ts`` (skin temperature)
    - ``pr`` (precipitation)
    - ``psl`` (sea-level pressure)
    - ``rlut`` (outgoing longwave radiation)
    - ``h500`` (geopotential height at 500 hPa)
    - ``ua250`` (zonal wind at 250 hPa — jet stream)
    """
    var_specs = [
        ("TMP2m", "K", "ΔTMP2m"),
        ("amon_ts", "K", "Δamon_ts"),
        ("pr", "kg m⁻² s⁻¹", "Δpr"),
        ("psl", "Pa", "Δpsl"),
        ("rlut", "W m⁻²", "Δrlut"),
        ("h500", "m", "Δh500"),
        ("ua250", "m s⁻¹", "Δua250"),
    ]
    present = [(v, u, t) for v, u, t in var_specs if f"{v}__mean" in stats_df.columns]
    if not present:
        return

    # Per-(source, experiment) ensemble mean over variants.
    def _delta(var: str) -> pd.Series:
        col = f"{var}__mean"
        per = (
            stats_df.dropna(subset=[col])
            .groupby(["source_id", "experiment"])[col]
            .mean()
            .unstack("experiment")
        )
        if "historical" not in per.columns or "ssp585" not in per.columns:
            return pd.Series(dtype=float)
        d = per["ssp585"] - per["historical"]
        return d.dropna()

    # Order: by ΔTMP2m if available, otherwise the first present var.
    primary = "TMP2m" if "TMP2m__mean" in stats_df.columns else present[0][0]
    primary_delta = _delta(primary).sort_values()
    if primary_delta.empty:
        log.info("No source has both historical and ssp585; skipping warming_response")
        return

    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, max(6, 0.28 * len(primary_delta) * nrows / 2)),
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (var, unit, title) in zip(axes, present):
        d = _delta(var).reindex(primary_delta.index)
        if d.dropna().empty:
            ax.set_visible(False)
            continue
        colors = ["indianred" if v >= 0 else "steelblue" for v in d.fillna(0).values]
        ax.barh(range(len(d)), d.values, color=colors)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.index, fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_title(f"{title} (ssp585 − historical)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[len(present) :]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS / "warming_response.png", dpi=110)
    plt.close()
    log.info("Wrote warming_response.png")


def plot_warning_breakdown(df: pd.DataFrame):
    if "warnings" not in df.columns:
        return
    counter: Counter = Counter()
    for ws in df.warnings:
        for w in ws or []:
            counter[w] += 1
    if not counter:
        log.info("No warnings recorded in index.")
        return
    items = counter.most_common(30)
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(labels))))
    ax.barh(range(len(labels)), counts, color="indianred")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("# datasets")
    ax.set_title("Top warning messages (v2)")
    plt.tight_layout()
    plt.savefig(PLOTS / "warning_breakdown.png", dpi=110)
    plt.close()
    log.info("Wrote warning_breakdown.png")


def write_summary(df: pd.DataFrame, stats_df: pd.DataFrame, outliers: pd.DataFrame):
    """A small text summary of what looks anomalous."""
    summary_lines = []
    summary_lines.append(f"# v2 stats summary (period={PERIOD})\n")
    summary_lines.append(
        f"- {len(df)} ok datasets, {df.source_id.nunique()} models, "
        f"{df.experiment.nunique()} experiments\n"
    )

    # Finite-fraction concerns: low or wildly variable.
    ff_cols = [c for c in stats_df.columns if c.endswith("__finite_fraction")]
    bad = []
    for c in ff_cols:
        vals = stats_df[c].dropna()
        if len(vals) == 0:
            continue
        if vals.min() < 0.5:
            bad.append((c, vals.min(), vals.median()))
    if bad:
        summary_lines.append("\n## Finite-fraction warnings (min < 0.5)\n")
        for c, mn, med in sorted(bad, key=lambda x: x[1]):
            summary_lines.append(
                f"- `{c.replace('__finite_fraction', '')}`: "
                f"min={mn:.3f}, median={med:.3f}\n"
            )

    # Variables present in fewer than 80% of datasets.
    var_counts: Counter = Counter()
    for vs in df.variables_present:
        for v in vs or []:
            var_counts[v] += 1
    sparse = [(v, c) for v, c in var_counts.items() if c < 0.8 * len(df)]
    if sparse:
        summary_lines.append("\n## Variables present in <80% of datasets\n")
        for v, c in sorted(sparse, key=lambda x: x[1]):
            summary_lines.append(
                f"- `{v}`: {c}/{len(df)} datasets ({100*c/len(df):.0f}%)\n"
            )

    summary_lines.append(
        f"\n## Outliers >3σ\n\n{len(outliers)} dataset×variable cells. "
        f"See `outliers.csv`.\n"
    )
    # Top 10 outliers inline.
    if len(outliers):
        summary_lines.append("\nTop 10:\n\n")
        for _, r in outliers.head(10).iterrows():
            summary_lines.append(
                f"- `{r.variable}` z={r.z:+.2f} "
                f"({r.source_id}/{r.experiment}/{r.variant_label}): "
                f"value={r.value:.3g} vs cohort_mean={r.cohort_mean:.3g}\n"
            )

    text = "".join(summary_lines)
    out_path = PLOTS.parent / "SUMMARY.md"
    out_path.write_text(text)
    log.info("Wrote %s", out_path)
    print("\n" + text)


def main():
    df = load_index()
    plot_coverage(df)
    plot_variable_presence(df)
    plot_warning_breakdown(df)
    plot_mask_coverage(df)
    stats_df = aggregate_stats(df, PERIOD)
    stats_df.to_csv(PLOTS.parent / "stats_aggregated.csv", index=False)
    plot_finite_fraction(stats_df)
    plot_units_sanity(stats_df)
    plot_crossmodel_stats(stats_df)
    plot_warming_response(stats_df)
    outliers = plot_outlier_table(stats_df)
    write_summary(df, stats_df, outliers)
    log.info(
        "Coverage: %d datasets across %d models × %d experiments",
        len(df),
        df.source_id.nunique(),
        df.experiment.nunique(),
    )


if __name__ == "__main__":
    main()
