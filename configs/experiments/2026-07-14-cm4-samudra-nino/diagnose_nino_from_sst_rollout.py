#!/usr/bin/env python3
"""Diagnose Nino3.4 skill from coupled AR SST rollouts.

Reads ocean monthly_mean_predictions/target.nc (preferred) or
autoregressive_predictions/target.nc (variable ``sst``), computes the
area-weighted Nino3.4 box mean, converts to monthly anomalies (FME / W&B
default: per-rollout monthly climatology on raw box SST, optional trailing
running mean), and reports RMSE/MAE/ACC by lead month.

Lead month ``k`` is the calendar month ``k`` months after the IC month — the
same lead definition as ``nino34_lead_k``.

By default each rollout builds its own monthly climatology for pred and target
(separately), matching ``RegionalIndexAggregator`` in inference. Optional
``linear_detrend`` + ``shared_truth_climatology`` reproduce the MLP label
recipe but distort lead-dependent RMSE when applied to short rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

N_LEADS = 12
NINO34_LAT_BOUNDS = (-5.0, 5.0)
NINO34_LON_BOUNDS = (190.0, 240.0)  # degrees east, 0-360 convention


def nino_box_weighted_mean(
    sst: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.DataArray:
    """Area-weighted mean SST over the Nino box, returned as a 1D time series."""
    lat = sst[lat_dim]
    lon = sst[lon_dim]
    lon360 = lon % 360

    lat_in = lat.where((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]), drop=True)
    if lon_bounds[0] <= lon_bounds[1]:
        lon_mask = (lon360 >= lon_bounds[0]) & (lon360 <= lon_bounds[1])
    else:
        lon_mask = (lon360 >= lon_bounds[0]) | (lon360 <= lon_bounds[1])
    lon_in = lon.where(lon_mask, drop=True)

    if lat_in.size == 0 or lon_in.size == 0:
        raise ValueError(
            "No grid cells fall within the Nino box "
            f"lat={lat_bounds}, lon={lon_bounds}."
        )

    sst_box = sst.sel({lat_dim: lat_in, lon_dim: lon_in})
    weights = np.cos(np.deg2rad(sst_box[lat_dim]))
    return sst_box.weighted(weights).mean(dim=[lat_dim, lon_dim], skipna=True)


def subtract_linear_trend(series: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Subtract a least-squares linear trend fit over the time index."""
    values = np.asarray(series.values, dtype=np.float64)
    x = np.arange(values.shape[0], dtype=np.float64)
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return series
    slope, intercept = np.polyfit(x[finite], values[finite], deg=1)
    trend = slope * x + intercept
    return series.copy(data=values - trend)


def _ym_key(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _lat_lon_dims(da: xr.DataArray) -> tuple[str, str]:
    spatial = [d for d in da.dims if d not in {"sample", "time"}]
    if "lat" in spatial and "lon" in spatial:
        return "lat", "lon"
    if len(spatial) < 2:
        raise ValueError(f"Could not find lat/lon dims in {da.dims}")
    return spatial[0], spatial[1]


def _open_sst_pair(input_dir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Open prediction/target SST, preferring monthly_mean files when present."""
    monthly_pred = input_dir / "monthly_mean_predictions.nc"
    monthly_tgt = input_dir / "monthly_mean_target.nc"
    raw_pred = input_dir / "autoregressive_predictions.nc"
    raw_tgt = input_dir / "autoregressive_target.nc"
    if monthly_pred.exists() and monthly_tgt.exists():
        return xr.open_dataset(monthly_pred), xr.open_dataset(monthly_tgt)
    if raw_pred.exists() and raw_tgt.exists():
        return xr.open_dataset(raw_pred), xr.open_dataset(raw_tgt)
    raise FileNotFoundError(
        f"No SST prediction/target pair under {input_dir}. "
        "Expected monthly_mean_*.nc or autoregressive_*.nc."
    )


def _calendar_year_month(
    dataset: xr.Dataset, sample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, months) along time for one sample."""
    if "valid_time" in dataset:
        vt = dataset["valid_time"].isel(sample=sample)
        return (
            np.asarray(vt.dt.year.values, dtype=np.int64),
            np.asarray(vt.dt.month.values, dtype=np.int64),
        )
    time = dataset["time"]
    if "sample" in time.dims:
        time = time.isel(sample=sample)
    return (
        np.asarray(time.dt.year.values, dtype=np.int64),
        np.asarray(time.dt.month.values, dtype=np.int64),
    )


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


def _per_series_climatology(
    values_1d: np.ndarray, months: np.ndarray
) -> dict[int, float]:
    """Per-calendar-month climatology from one rollout's box-mean series."""
    values = np.asarray(values_1d, dtype=np.float64)
    months = np.asarray(months, dtype=np.int64)
    climatology: dict[int, float] = {}
    for month in range(1, 13):
        sel = values[months == month]
        sel = sel[~np.isnan(sel)]
        if sel.size > 0:
            climatology[month] = float(sel.mean())
    return climatology


def _monthly_climatology_from_members(
    box_mean: xr.DataArray, dataset: xr.Dataset
) -> dict[int, float]:
    """Fit per-calendar-month climatology across all samples in one year job."""
    values_list: list[np.ndarray] = []
    months_list: list[np.ndarray] = []
    for sample in range(box_mean.sizes["sample"]):
        values_list.append(
            np.asarray(box_mean.isel(sample=sample).values, dtype=np.float64)
        )
        _, months = _calendar_year_month(dataset, sample)
        months_list.append(months)
    values = np.concatenate(values_list)
    months = np.concatenate(months_list)
    return _per_series_climatology(values, months)


def _index_lookup(
    values_1d: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    n_running_months: int,
    climatology: dict[int, float] | None = None,
) -> dict[int, float]:
    """Ym -> monthly (optionally smoothed) anomaly index for one forecast."""
    if climatology is None:
        climatology = _per_series_climatology(values_1d, months)
    ym = years * 12 + (months - 1)
    climo = np.array(
        [climatology.get(int(m), np.nan) for m in months], dtype=np.float64
    )
    anom_native = np.asarray(values_1d, dtype=np.float64) - climo

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
    linear_detrend: bool = False,
    shared_truth_climatology: bool = False,
    n_running_months: int = 1,
    checkpoint_dataset: str = "",
    write_plots: bool = True,
) -> xr.Dataset:
    prediction, target = _open_sst_pair(input_dir)
    try:
        if "sst" not in prediction or "sst" not in target:
            raise KeyError(
                "Expected 'sst' in prediction/target; got "
                f"pred={list(prediction.data_vars)} "
                f"target={list(target.data_vars)}"
            )

        pred_box = _box_mean_sst(prediction).compute()
        truth_box = _box_mean_sst(target).compute()

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

        shared_climatology: dict[int, float] | None = None
        if shared_truth_climatology:
            shared_climatology = _monthly_climatology_from_members(truth_box, target)

        n_samples = pred_box.sizes["sample"]
        init_times = prediction["init_time"].values
        pred_leads = np.full((n_samples, N_LEADS), np.nan, dtype=np.float64)
        truth_leads = np.full((n_samples, N_LEADS), np.nan, dtype=np.float64)

        for s in range(n_samples):
            p_years, p_months = _calendar_year_month(prediction, s)
            t_years, t_months = _calendar_year_month(target, s)
            p_climo = shared_climatology
            t_climo = shared_climatology
            p_index = _index_lookup(
                pred_box.isel(sample=s).values,
                p_years,
                p_months,
                n_running_months,
                climatology=p_climo,
            )
            t_index = _index_lookup(
                truth_box.isel(sample=s).values,
                t_years,
                t_months,
                n_running_months,
                climatology=t_climo,
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
        rmse = np.sqrt((error**2).mean("sample", skipna=True))
        mae = np.abs(error).mean("sample", skipna=True)
        correlation = xr.corr(
            pred_da.where(np.isfinite(pred_da) & np.isfinite(truth_da)),
            truth_da.where(np.isfinite(pred_da) & np.isfinite(truth_da)),
            dim="sample",
        )

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
                    f"shared_truth_climatology={shared_truth_climatology}, "
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
        "shared_truth_climatology": shared_truth_climatology,
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

    if write_plots:
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

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", default="")
    parser.add_argument(
        "--linear-detrend",
        action="store_true",
        help="Subtract a linear trend from each rollout's box mean before anomalies.",
    )
    parser.add_argument(
        "--shared-truth-climatology",
        action="store_true",
        help="Use one truth climatology pooled across all ICs (MLP label style).",
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
        linear_detrend=args.linear_detrend,
        shared_truth_climatology=args.shared_truth_climatology,
        n_running_months=args.running_mean_months,
        checkpoint_dataset=args.checkpoint_dataset,
    )
    print(result[["rmse", "mae", "correlation"]])


if __name__ == "__main__":
    main()
