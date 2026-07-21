#!/usr/bin/env python3
"""Compact broadcast Nino3.4 fields from evaluator output into scalar data."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

N_LEADS = 12
NINO_NAMES = [f"nino34_lead_{lead:02d}" for lead in range(1, N_LEADS + 1)]


def _spatial_mean(dataset: xr.Dataset, name: str) -> xr.DataArray:
    field = dataset[name]
    spatial_dims = [dim for dim in field.dims if dim not in {"sample", "time"}]
    return field.mean(spatial_dims, skipna=False)


def _format_time(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


def compact(input_dir: Path, output_dir: Path) -> xr.Dataset:
    prediction = xr.open_dataset(input_dir / "autoregressive_predictions.nc")
    target = xr.open_dataset(input_dir / "autoregressive_target.nc")

    try:
        pred = xr.concat(
            [_spatial_mean(prediction, name) for name in NINO_NAMES],
            dim=xr.IndexVariable("lead_month", np.arange(1, N_LEADS + 1)),
        ).transpose("sample", "time", "lead_month")
        truth = xr.concat(
            [_spatial_mean(target, name) for name in NINO_NAMES],
            dim=xr.IndexVariable("lead_month", np.arange(1, N_LEADS + 1)),
        ).transpose("sample", "time", "lead_month")

        # This evaluator has one forward step, so remove the singleton rollout
        # dimension. The lead_month dimension is the direct readout horizon.
        pred = pred.isel(time=0, drop=True)
        truth = truth.isel(time=0, drop=True)
        error = pred - truth

        rmse = np.sqrt((error**2).mean("sample"))
        mae = np.abs(error).mean("sample")
        correlation = xr.corr(pred, truth, dim="sample")

        result = (
            xr.Dataset(
                data_vars={
                    "prediction": pred,
                    "target": truth,
                    "error": error,
                    "rmse": rmse,
                    "mae": mae,
                    "correlation": correlation,
                },
                coords={
                    "forecast": ("sample", np.arange(pred.sizes["sample"])),
                    "init_time": ("sample", prediction["init_time"].values),
                    "valid_time": (
                        "sample",
                        prediction["valid_time"].isel(time=0).values,
                    ),
                },
                attrs={
                    "description": (
                        "Direct Nino3.4 scalar forecasts from one 5-day Samudra "
                        "step over held-out CM4 1pctCO2 years 0251-0255."
                    ),
                    "checkpoint_dataset": "01KXKZ85HTDSGGXWD2DPW2QRFW",
                    "readout_definition": (
                        "nino34_lead_01..12 predicted simultaneously by the MLP "
                        "readout head; values are linear-detrended monthly indices."
                    ),
                },
            )
            .swap_dims({"sample": "forecast"})
            .drop_vars("sample")
        )
    finally:
        prediction.close()
        target.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(output_dir / "nino_scalar_forecasts.nc")

    rows = []
    init_times = _format_time(result.init_time.values)
    valid_times = _format_time(result.valid_time.values)
    for forecast in range(result.sizes["forecast"]):
        for lead in range(1, N_LEADS + 1):
            rows.append(
                {
                    "forecast": forecast,
                    "init_time": init_times[forecast],
                    "valid_time": valid_times[forecast],
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
    pd.DataFrame(rows).to_csv(output_dir / "nino_scalar_forecasts.csv", index=False)

    summary = {
        "n_forecasts": result.sizes["forecast"],
        "rmse_by_lead": result.rmse.to_series().to_dict(),
        "mae_by_lead": result.mae.to_series().to_dict(),
        "correlation_by_lead": result.correlation.to_series().to_dict(),
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
        xlabel="Direct readout lead (months)",
        ylabel="Correlation",
        xticks=result.lead_month,
        ylim=(-1, 1),
    )
    axes[1].grid(alpha=0.3)
    fig.savefig(output_dir / "skill_by_lead.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    lead = 1
    ax.plot(result.init_time, result.target.sel(lead_month=lead), label="target")
    ax.plot(
        result.init_time,
        result.prediction.sel(lead_month=lead),
        label="prediction",
    )
    ax.set(title="Direct Nino3.4 prediction: lead month 1", ylabel="Index (K)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "lead01_timeseries.png", dpi=150)
    plt.close(fig)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compact(args.input_dir, args.output_dir)
    print(result[["rmse", "mae", "correlation"]])


if __name__ == "__main__":
    main()
