# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run SamudrACE from the published checkpoint and plot forecast contours.

Sanity check for the SamudrACE coupled atmosphere-ocean emulator: runs a
120 day deterministic forecast from a CM4 preindustrial-control initial
condition and plots the atmosphere and ocean state at four lead times, the SST change
from the initial condition, and the global evolution of both components.

Everything -- checkpoint, initial condition, and forcing -- is downloaded from
the ``allenai/SamudrACE-CM4-piControl`` HuggingFace repository on first run
(roughly 1 GB, cached afterwards).

This review-only script is intentionally not part of the pull request.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
import torch
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import earth2studio.run as run  # noqa: E402
from earth2studio.data.samudrace import SamudrACEData  # noqa: E402
from earth2studio.io import ZarrBackend  # noqa: E402
from earth2studio.models.px import SamudrACE  # noqa: E402

# The SamudrACE checkpoint was trained on a GFDL CM4 preindustrial-control run,
# so the initial condition is a CM4 model-year timestamp, not a real date.
# Model years overflow datetime64[ns], hence second precision throughout.
INITIAL_TIME = np.datetime64("0311-01-01T00:00:00", "s")
FORCING_SCENARIO = "0311"

# The atmosphere steps every 6 hours and the ocean every 5 days, so a coupled
# cycle is 20 atmosphere steps. 24 cycles is 120 days.
STEPS_PER_CYCLE = 20
N_CYCLES = 24
NSTEPS = N_CYCLES * STEPS_PER_CYCLE

# Fields to run and plot: one atmosphere field and one ocean field.
RUN_VARIABLES = ("t2m", "sst")

# Contour rows. The SST anomaly row is what shows the ocean actually
# evolving: the raw SST field barely changes by eye over 120 days.
CONTOUR_ROWS = (
    ("t2m", "2 m temperature (atmosphere)", "K", "RdYlBu_r", (0, 30, 60, 120), False),
    ("sst", "Sea surface temperature (ocean)", "K", "viridis", (0, 30, 60, 120), False),
    (
        "sst",
        "SST change from initial condition",
        "K",
        "RdBu_r",
        (30, 60, 90, 120),
        True,
    ),
)
N_COLUMNS = 4

OUTPUT_PATH = Path("samudrace_sanity_contours.png")
STORE_PATH = Path("samudrace_sanity.zarr")


def run_forecast(device: torch.device) -> xr.Dataset:
    """Run the SamudrACE forecast and return its output as a dataset.

    Parameters
    ----------
    device : torch.device
        Device to run inference on.

    Returns
    -------
    xr.Dataset
        Forecast with dims (time, lead_time, lat, lon), one variable per
        plotted field.
    """
    package = SamudrACE.load_default_package()
    model = SamudrACE.load_model(package, scenario=FORCING_SCENARIO)
    data = SamudrACEData()

    io = ZarrBackend(
        file_name=str(STORE_PATH),
        chunks={"time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": True},
    )
    run.deterministic(
        [INITIAL_TIME],
        NSTEPS,
        model,
        data,
        io,
        output_coords=OrderedDict(
            {"variable": np.array(RUN_VARIABLES, dtype=object)}
        ),
        device=device,
    )
    return xr.open_zarr(STORE_PATH, consolidated=False).isel(time=0).load()


def global_mean(field: xr.DataArray) -> xr.DataArray:
    """Area-weighted global mean, ignoring masked (NaN) points.

    Parameters
    ----------
    field : xr.DataArray
        Field with lat and lon dimensions.

    Returns
    -------
    xr.DataArray
        Mean over lat and lon.
    """
    weights = np.cos(np.deg2rad(field["lat"]))
    return field.weighted(weights.fillna(0.0)).mean(dim=("lat", "lon"))


def plot_forecast(forecast: xr.Dataset) -> None:
    """Plot forecast contours at several lead times and the global series.

    Parameters
    ----------
    forecast : xr.Dataset
        Forecast returned by :func:`run_forecast`.
    """
    lead_days = forecast["lead_time"] / np.timedelta64(1, "D")

    def at_lead(name: str, day: float) -> xr.DataArray:
        return forecast[name].isel(lead_time=int(np.abs(lead_days - day).argmin()))

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = GridSpec(4, 3 * N_COLUMNS, figure=fig, height_ratios=[1, 1, 1, 0.8])

    for row, (name, title, units, cmap, days, anomaly) in enumerate(CONTOUR_ROWS):
        fields = [at_lead(name, day) for day in days]
        if anomaly:
            fields = [field - at_lead(name, 0) for field in fields]
            bound = max(float(np.nanpercentile(np.abs(f), 99.0)) for f in fields)
            levels = np.linspace(-bound, bound, 21)
        else:
            lower = min(float(np.nanpercentile(f, 1.0)) for f in fields)
            upper = max(float(np.nanpercentile(f, 99.0)) for f in fields)
            levels = np.linspace(lower, upper, 21)

        for column, (day, field) in enumerate(zip(days, fields)):
            ax = fig.add_subplot(grid[row, 3 * column : 3 * column + 3])
            # Land (and, for ocean fields, anything off the ocean grid) is NaN.
            ax.set_facecolor("0.85")
            contour = ax.contourf(
                field["lon"],
                field["lat"],
                field,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            label = f"{name} anomaly" if anomaly else name
            ax.set_title(f"{label} -- lead +{day} days", fontsize=10)
            if column == 0:
                ax.set_ylabel(f"{title}\nLatitude", fontsize=9)
            if row == len(CONTOUR_ROWS) - 1:
                ax.set_xlabel("Longitude", fontsize=9)
            if column == N_COLUMNS - 1:
                fig.colorbar(contour, ax=ax, label=units)

    # Global series: the coupled system should evolve smoothly, with the
    # ocean stepping once per 5 day cycle and the atmosphere every 6 hours.
    series = (
        ("Global mean 2 m temperature (K)", global_mean(forecast["t2m"])),
        ("Global mean SST (K)", global_mean(forecast["sst"])),
    )
    for index, (label, values) in enumerate(series):
        ax = fig.add_subplot(grid[3, 6 * index : 6 * index + 6])
        ax.plot(lead_days, values, linewidth=1.2)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SamudrACE {N_CYCLES} coupled cycles ({NSTEPS} atmosphere steps, "
        f"120 days) from CM4 piControl {INITIAL_TIME}",
        fontsize=13,
    )
    fig.savefig(OUTPUT_PATH, dpi=110, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


def main() -> None:
    """Run the forecast and write the sanity-check figure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SamudrACE on {device}")

    forecast = run_forecast(device)
    print(f"Forecast variables: {sorted(forecast.data_vars)}")
    print(f"Lead times: {forecast.sizes['lead_time']}")
    for name in RUN_VARIABLES:
        field = forecast[name]
        finite = np.isfinite(field.values)
        print(
            f"  {name}: min {np.nanmin(field.values):.3f} "
            f"max {np.nanmax(field.values):.3f} "
            f"finite {finite.sum() / finite.size:.1%}"
        )
        if not finite.any():
            raise RuntimeError(f"{name} is entirely non-finite")

    plot_forecast(forecast)


if __name__ == "__main__":
    main()
