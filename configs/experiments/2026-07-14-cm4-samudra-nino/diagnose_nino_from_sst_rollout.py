#!/usr/bin/env python3
"""Diagnose Nino3.4 skill from coupled AR SST rollouts.

Reads ocean ``autoregressive_predictions/target.nc`` (variable ``sst``),
computes the area-weighted Nino3.4 box mean, converts to monthly anomalies
matching the MLP label recipe (linear detrend + monthly climatology + optional
trailing running mean), and reports RMSE/MAE/ACC by lead month.

Lead month ``k`` is the calendar month ``k`` months after the IC month — the
same lead definition as ``nino34_lead_k``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_LABELS_DIR = (
    Path(__file__).resolve().parents[3] / "scripts" / "compute_nino_lead_labels"
)
sys.path.insert(0, str(_LABELS_DIR))
from compute_nino_lead_labels import (  # noqa: E402
    NINO34_LAT_BOUNDS,
    NINO34_LON_BOUNDS,
    nino_box_weighted_mean,
    subtract_linear_trend,
)

N_LEADS = 12


def _ym_key(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _lat_lon_dims(da: xr.DataArray) -> tuple[str, str]:
    spatial = [d for d in da.dims if d not in {"sample", "time"}]
    if "lat" in spatial and "lon" in spatial:
        return "lat", "lon"
    if len(spatial) < 2:
        raise ValueError(f"Could not find lat/lon dims in {da.dims}")
    return spatial[0], spatial[1]


def _box_mean_sst(dataset: xr.Dataset) -> xr.DataArray:
    """Area-weighted Nino3.4 SST with dims (sample, time)."""
    lat_dim, lon_dim = _lat_lon_dims(dataset["sst"])
    means = []
    for sample in range(dataset.sizes["sample"]):
        means.append(
            nino_box_weighted_mean(
                dataset["sst"].isel(sample=sample),
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                lat_bounds=NINO34_LAT_BOUNDS,
                lon_bounds=NINO34_LON_BOUNDS,
            )
        )
    return xr.concat(means, dim="sample")


def _monthly_climatology(box_mean: xr.DataArray) -> dict[int, float]:
    """Fit per-calendar-month climatology on a 1D or flattened series."""
    values = np.asarray(box_mean.values, dtype=np.float64).ravel()
    months = np.asarray(box_mean["time.month"].values, dtype=np.int64).ravel()
    climatology: dict[int, float] = {}
    for month in range(1, 13):
        sel = values[months == month]
        sel = sel[~np.isnan(sel)]
        if sel.size > 0:
            climatology[month] = float(sel.mean())
    return climatology


def _index_lookup(
    box_mean_1d: xr.DataArray,
    climatology: dict[int, float],
    n_running_months: int,
) -> dict[int, float]:
    """Ym -> monthly (optionally smoothed) anomaly index for one forecast."""
    values = np.asarray(box_mean_1d.values, dtype=np.float64)
    years = np.asarray(box_mean_1d["time.year"].values, dtype=np.int64)
    months = np.asarray(box_mean_1d["time.month"].values, dtype=np.int64)
    ym = years * 12 + (months - 1)
    climo = np.array(
        [climatology.get(int(m), np.nan) for m in months], dtype=np.float64
    )
    anom_native = values - climo

    unique_ym = np.array(sorted(np.unique(ym).tolist()), dtype=np.int64)
    monthly = np.full(unique_ym.shape, np.nan, dtype=np.float64)
    for i, key in enumerate(unique_ym):
        sel = anom_native[ym == key]
        sel = sel[~np.isnan(sel)]
        if sel.size > 0:
            monthly[i] = float(sel.mean())

    index: dict[int, float] = {}
    for i in range(len(unique_ym)):
        if i >= n_running_months - 1:
            window = monthly[i - n_running_months + 1 : i + 1]
            if np.any(~np.isnan(window)):
                index[int(unique_ym[i])] = float(np.nanmean(window))
    return index


def _init_year_month(init_time) -> tuple[int, int]:
    return int(init_time.year), int(init_time.month)


def diagnose(
    input_dir: Path,
    output_dir: Path,
    *,
    linear_detrend: bool = True,
    n_running_months: int = 1,
    checkpoint_dataset: str = "",
) -> xr.Dataset:
    prediction = xr.open_dataset(input_dir / "autoregressive_predictions.nc")
    target = xr.open_dataset(input_dir / "autoregressive_target.nc")
    try:
        if "sst" not in prediction or "sst" not in target:
            raise KeyError(
                "Expected 'sst' in prediction/target; got "
                f"pred={list(prediction.data_vars)} "
                f"target={list(target.data_vars)}"
            )

        pred_box = _box_mean_sst(prediction).compute()
        truth_box = _box_mean_sst(target).compute()

        # Fit seasonal climatology (and optional linear trend) on all target
        # members so every forecast shares the same baseline.
        if linear_detrend:
            pred_box = xr.concat(
                [
                    subtract_linear_trend(pred_box.isel(sample=s), "time")
                    for s in range(pred_box.sizes["sample"])
                ],
                dim="sample",
            )
            truth_box = xr.concat(
                [
                    subtract_linear_trend(truth_box.isel(sample=s), "time")
                    for s in range(truth_box.sizes["sample"])
                ],
                dim="sample",
            )

        # Build a series for climatology by concatenating target members.
        clim_source = xr.concat(
            [truth_box.isel(sample=s) for s in range(truth_box.sizes["sample"])],
            dim="time",
        )
        climatology = _monthly_climatology(clim_source)

        n_samples = pred_box.sizes["sample"]
        init_times = prediction["init_time"].values
        pred_leads = np.full((n_samples, N_LEADS), np.nan, dtype=np.float64)
        truth_leads = np.full((n_samples, N_LEADS), np.nan, dtype=np.float64)

        for s in range(n_samples):
            p_index = _index_lookup(
                pred_box.isel(sample=s), climatology, n_running_months
            )
            t_index = _index_lookup(
                truth_box.isel(sample=s), climatology, n_running_months
            )
            init_year, init_month = _init_year_month(init_times[s])
            base = _ym_key(init_year, init_month)
            for lead in range(1, N_LEADS + 1):
                key = base + lead
                pred_leads[s, lead - 1] = p_index.get(key, np.nan)
                truth_leads[s, lead - 1] = t_index.get(key, np.nan)

        lead_month = np.arange(1, N_LEADS + 1)
        pred_da = xr.DataArray(
            pred_leads,
            dims=("sample", "lead_month"),
            coords={"sample": np.arange(n_samples), "lead_month": lead_month},
            name="prediction",
        )
        truth_da = xr.DataArray(
            truth_leads,
            dims=("sample", "lead_month"),
            coords={"sample": np.arange(n_samples), "lead_month": lead_month},
            name="target",
        )
        error = pred_da - truth_da
        rmse = np.sqrt((error**2).mean("sample"))
        mae = np.abs(error).mean("sample")
        correlation = xr.corr(pred_da, truth_da, dim="sample")

        result = xr.Dataset(
            data_vars={
                "prediction": pred_da,
                "target": truth_da,
                "error": error,
                "rmse": rmse,
                "mae": mae,
                "correlation": correlation,
            },
            coords={
                "forecast": ("sample", np.arange(n_samples)),
                "init_time": ("sample", init_times),
            },
            attrs={
                "description": (
                    "Nino3.4 diagnosed from coupled AR SST rollouts "
                    f"(linear_detrend={linear_detrend}, "
                    f"running_mean_months={n_running_months})."
                ),
                "checkpoint_dataset": checkpoint_dataset,
                "sst_variable": "sst",
                "nino34_lat_bounds": str(NINO34_LAT_BOUNDS),
                "nino34_lon_bounds": str(NINO34_LON_BOUNDS),
            },
        ).swap_dims({"sample": "forecast"})
    finally:
        prediction.close()
        target.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(output_dir / "nino_ar_sst_forecasts.nc")

    rows = []
    for forecast in range(result.sizes["forecast"]):
        for lead in range(1, N_LEADS + 1):
            rows.append(
                {
                    "forecast": forecast,
                    "init_time": str(result.init_time.values[forecast]),
                    "lead_month": lead,
                    "prediction": float(
                        result.prediction.sel(forecast=forecast, lead_month=lead).item()
                    ),
                    "target": float(
                        result.target.sel(forecast=forecast, lead_month=lead).item()
                    ),
                    "error": float(
                        result.error.sel(forecast=forecast, lead_month=lead).item()
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "nino_ar_sst_forecasts.csv", index=False)

    summary = {
        "n_forecasts": result.sizes["forecast"],
        "linear_detrend": linear_detrend,
        "n_running_months": n_running_months,
        "rmse_by_lead": {
            str(int(k)): float(v) for k, v in result.rmse.to_series().items()
        },
        "mae_by_lead": {
            str(int(k)): float(v) for k, v in result.mae.to_series().items()
        },
        "correlation_by_lead": {
            str(int(k)): float(v) for k, v in result.correlation.to_series().items()
        },
    }
    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(result.lead_month, result.rmse, marker="o", label="RMSE")
    axes[0].plot(result.lead_month, result.mae, marker="o", label="MAE")
    axes[0].set(ylabel="Nino3.4 index error (K)", xticks=result.lead_month)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(result.lead_month, result.correlation, marker="o")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Lead (months)",
        ylabel="Correlation",
        xticks=result.lead_month,
        ylim=(-1, 1),
    )
    axes[1].grid(alpha=0.3)
    fig.savefig(output_dir / "skill_by_lead.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    forecast_idx = result.forecast.values
    ax.plot(forecast_idx, result.target.sel(lead_month=1), label="target")
    ax.plot(forecast_idx, result.prediction.sel(lead_month=1), label="prediction")
    ax.set(
        title="AR SST-diagnosed Nino3.4: lead month 1",
        xlabel="Forecast index",
        ylabel="Index (K)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "lead01_timeseries.png", dpi=150)
    plt.close(fig)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", default="")
    parser.add_argument(
        "--no-linear-detrend",
        action="store_true",
        help="Disable linear detrend (default: on, matching MLP labels).",
    )
    parser.add_argument(
        "--running-mean-months",
        type=int,
        default=1,
        help="Trailing monthly running mean (1=raw monthly anomaly, 5=ONI-like).",
    )
    args = parser.parse_args()
    result = diagnose(
        args.input_dir,
        args.output_dir,
        linear_detrend=not args.no_linear_detrend,
        n_running_months=args.running_mean_months,
        checkpoint_dataset=args.checkpoint_dataset,
    )
    print(result[["rmse", "mae", "correlation"]])


if __name__ == "__main__":
    main()
