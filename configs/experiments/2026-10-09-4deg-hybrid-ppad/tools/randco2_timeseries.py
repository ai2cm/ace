# ruff: noqa
# mypy: ignore-errors
"""Spencer's random-CO2 evaluator-run figures (explore2 spencerc/2026-09-28-CM4-like-AM4-random-CO2-evaluation-runs,
2026-09-28 / 2026-10-14 notebooks) for the ACE-before/after-coupled-FT atmospheres: per CO2 climate, the 6-hourly
global-mean CO2 (log), T_0 and T2m of each model vs the CM4-like AM4 target on ic_0003, validation year shaded, held-out
R^2 in the legend; plus the held-out summary table (bias, RMSE, correlation, R^2). Series come from the evaluator result
datasets (mean_diagnostics.nc, cached by randco2_compare.py) instead of wandb history; CO2 from the v2 4deg store on GCS.
"""

import os
import textwrap
import warnings

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotting
from cftime import DatetimeNoLeap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.environ.get("RANDCO2_OUT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DATASET_ROOT = "gs://vcm-ml-intermediate/2026-10-11-CM4-like-AM4-random-CO2-4deg"
INITIAL_TIME, CALENDAR = "0152-10-01T06", "noleap"
PLOT_START = DatetimeNoLeap(153, 1, 1, 6)
VALIDATION_START, VALIDATION_END = DatetimeNoLeap(155, 1, 1), DatetimeNoLeap(156, 1, 1)
REFERENCE_CO2 = 284.317
VARIABLES = ["air_temperature_0", "TMP2m"]
CLIMATES = ["1xCO2", "2xCO2", "4xCO2"]
MODELS = {  # legend label -> atmosphere name in randco2_compare.csv
    "ACE2S-ft rs0 (1pctCO$_2$ 46–125, piC 156–235; never saw random CO$_2$)": "ace2sft-rs0 (never saw randCO2), before",
    "ACE2S-ft rs0 after coupled FT (jointens epoch 1, lifted out)": "ace2sft-rs0, after coupled FT ep1 (lifted)",
    "random-CO$_2$ rs2, epoch 11 (Spencer's v2 1pct_46to125 arm), before coupling": "rco2 rs2 ep11, before",
    "random-CO$_2$ rs2 after coupled FT (lifted out)": "rco2 rs2, after coupled FT (lifted)",
}
COLORS = {m: f"C{i}" for i, m in enumerate(MODELS)}
YLIMITS = {"air_temperature_0": (210, 240), "TMP2m": (282, 301)}
YLABELS = {"air_temperature_0": "Global mean T$_0$ [K]", "TMP2m": "Global mean T2m [K]"}

df = pd.read_csv(os.path.join(HERE, "randco2_compare.csv")).dropna(
    subset=["results_ds"]
)
time = xr.date_range(
    INITIAL_TIME, periods=7668, freq="6h", calendar=CALENDAR, use_cftime=True
)


def load(ds_id):
    m = xr.open_dataset(os.path.join(HERE, "cache", ds_id, "mean_diagnostics.nc"))
    out = {}
    for src in ("gen", "target"):
        out[src] = xr.Dataset(
            {
                v: ("time", m[f"weighted_mean_{src}-{v}"].values[: len(time)])
                for v in VARIABLES
            },
            coords={"time": time[: m.sizes["forecast_step"]]},
        )
    return out


gen, target = {}, {}
for label, atm in MODELS.items():
    for _, r in df[df.atmosphere == atm].iterrows():
        d = load(r.results_ds)
        clim = f"{int(r.xCO2)}xCO2"
        gen[(label, clim)] = d["gen"].sel(time=slice(PLOT_START, None))
        target[clim] = d["target"].sel(
            time=slice(PLOT_START, None)
        )  # same target for every model
present = [m for m in MODELS if any(k[0] == m for k in gen)]
end_time = min(g.time.values[-1] for g in gen.values())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    co2 = (
        xr.concat(
            [
                xr.open_zarr(
                    f"{DATASET_ROOT}/random-CO2-{c}-ic_0003.zarr",
                    decode_timedelta=False,
                ).carbon_dioxide
                for c in CLIMATES
            ],
            dim=pd.Index(CLIMATES, name="climate"),
        )
        .sel(time=slice(PLOT_START, end_time))
        .load()
    )
if "lat" in co2.dims:
    co2 = co2.mean(("lat", "lon"))


def held_out(da):
    v = (da.time >= VALIDATION_START) & (da.time < VALIDATION_END)
    return da.where(~v, drop=True)


def r_squared(g, t):
    return 1 - ((g - t) ** 2).sum("time") / ((t - t.mean("time")) ** 2).sum("time")


def plot_time_series(climate):
    plotting.configure_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 5.6), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.30, hspace=0.12)
    co2.sel(climate=climate).plot(
        ax=ax1, yscale="log", yincrease=False, color="k", ls="--", lw=1
    )
    ax1.set_ylim(16 * REFERENCE_CO2, 0.25 * REFERENCE_CO2)
    ax1.set_yticks(
        [0.25 * REFERENCE_CO2, REFERENCE_CO2, 4 * REFERENCE_CO2, 16 * REFERENCE_CO2]
    )
    ax1.set_yticklabels(["0.25xCO$_2$", "1xCO$_2$", "4xCO$_2$", "16xCO$_2$"])
    handles = []
    for ax, v in zip((ax2, ax3), VARIABLES):
        for m in present:
            if (m, climate) in gen:
                gen[(m, climate)][v].plot(ax=ax, color=COLORS[m], lw=0.8)
        target[climate][v].plot(ax=ax, color="k", ls="--", lw=0.8)
        ax.set_ylim(YLIMITS[v])
        ax.set_ylabel(textwrap.fill(YLABELS[v], 25))
    for m in present:
        if (m, climate) not in gen:
            continue
        r2 = [
            f"{s} {r_squared(held_out(gen[(m, climate)][v]), held_out(target[climate][v])).item():.2f}"
            for v, s in zip(VARIABLES, ("T$_0$", "T2m"))
        ]
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[m],
                label=f"ACE (1 seed): {m}\nR$^2$: {', '.join(r2)}",
            )
        )
    handles += [
        Line2D([0], [0], color="k", ls="--", label="CM4-like AM4 (v2 store)"),
        Patch(color="k", alpha=0.1, lw=0, label="Held out from training/validation"),
    ]
    start, end = co2.time.values[0], co2.time.values[-1]
    for ax in (ax1, ax2, ax3):
        ax.axvspan(start, VALIDATION_START, color="k", alpha=0.1, lw=0)
        ax.axvspan(VALIDATION_END, end, color="k", alpha=0.1, lw=0)
        ax.set_xlim(start, end)
        ax.set_title("")
        ax.set_xlabel("")
    ax1.set_ylabel(
        textwrap.fill(f"Global mean CO$_2$ (1xCO$_2$ = {REFERENCE_CO2} ppm)", 25)
    )
    ax1.set_title(
        f"Inference with random CO$_2$ forcing centered about {climate.replace('CO2', 'CO$_2$')}, ic_0003"
    )
    for ax, lab in zip((ax1, ax2, ax3), "abc"):
        ax.annotate(
            f"({lab})",
            xy=(0, 1),
            xytext=(2, -2),
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.0),
        ncol=1,
        fontsize=7,
    )
    return fig


for c in CLIMATES:
    f = plot_time_series(c)
    f.savefig(os.path.join(FIG, f"random-co2-{c}-ic_0003-time-series.png"), dpi=200)
    plt.close(f)

records = []
for m in present:
    for c in CLIMATES:
        if (m, c) not in gen:
            continue
        for v in VARIABLES:
            g, t = held_out(gen[(m, c)][v]), held_out(target[c][v])
            e = g - t
            records.append(
                {
                    "variable": v,
                    "climate": c,
                    "model": MODELS[m],
                    "bias": e.mean().item(),
                    "rmse": float(np.sqrt((e**2).mean())),
                    "correlation": xr.corr(g, t, dim="time").item(),
                    "r_squared": r_squared(g, t).item(),
                }
            )
summary = pd.DataFrame(records).set_index(["variable", "climate", "model"]).sort_index()
summary.to_csv(os.path.join(HERE, "randco2_heldout_summary.csv"))
pd.set_option("display.width", 200)
print(
    summary.round({"bias": 2, "rmse": 2, "correlation": 3, "r_squared": 3}).to_string()
)
