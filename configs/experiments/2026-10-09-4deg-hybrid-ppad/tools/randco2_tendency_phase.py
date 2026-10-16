# ruff: noqa
# mypy: ignore-errors
"""Spencer's T0-tendency phase-space figure (explore2 spencerc/2025-12-10-CM4-like-AM4-random-co2,
2026-01-12-random-CO2-inference-stochastic.ipynb) for the before/after-coupled-FT atmospheres: the global-mean
top-layer temperature tendency (K/day) binned by next-step CO2 (log2-spaced, 11 edges from 1/4x to 16x) and the
global-mean T0 at the start of the step (210-230 K), pooled over the 1x/2x/4x ic_0003 members with the validation
year 155 excluded. Top row: CM4-like AM4 target then each model; bottom row: model minus target."""

import os, textwrap, warnings
import numpy as np, pandas as pd, xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cftime import DatetimeNoLeap
import plotting

HERE = os.environ.get("RANDCO2_OUT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DATASET_ROOT = "gs://vcm-ml-intermediate/2026-10-11-CM4-like-AM4-random-CO2-4deg"
INITIAL_TIME, CALENDAR = "0152-10-01T06", "noleap"
CLIMATES = ["1xCO2", "2xCO2", "4xCO2"]
VALIDATION_YEAR = 155
REFERENCE_CO2 = 284.317
N_BINS = 11
CO2_BINS = REFERENCE_CO2 * np.logspace(-2, 4, N_BINS, base=2)
T0_BINS = np.linspace(210, 230, N_BINS)
TIMESTEP_DAYS = 0.25
MODELS = {
    "ACE2S-ft rs0\n(never saw random CO$_2$)": "ace2sft-rs0 (never saw randCO2), before",
    "random-CO$_2$ rs2\nbefore coupling": "rco2 rs2 ep11, before",
}
df = pd.read_csv(os.path.join(HERE, "randco2_compare.csv")).dropna(
    subset=["results_ds"]
)
for a in sorted(
    {a for a in df.atmosphere if str(a).startswith("rco2 rs2, after")},
    key=lambda a: int(a.split("ep")[1].split()[0]),
):
    MODELS[f"random-CO$_2$ rs2\nafter coupled FT ep {a.split('ep')[1].split()[0]}"] = a
time = xr.date_range(
    INITIAL_TIME, periods=7668, freq="6h", calendar=CALENDAR, use_cftime=True
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    co2 = {
        c: xr.open_zarr(
            f"{DATASET_ROOT}/random-CO2-{c}-ic_0003.zarr", decode_timedelta=False
        )
        .carbon_dioxide.sel(time=time)
        .load()
        for c in CLIMATES
    }
for c in CLIMATES:
    if "lat" in co2[c].dims:
        co2[c] = co2[c].mean(("lat", "lon"))


def series(ds_id, src):
    m = xr.open_dataset(os.path.join(HERE, "cache", ds_id, "mean_diagnostics.nc"))
    return xr.DataArray(
        m[f"weighted_mean_{src}-air_temperature_0"].values,
        coords={"time": time[: m.sizes["forecast_step"]]},
        dims="time",
    )


def binned_tendency(t0_by_climate):
    """mean dT0/dt (K/day) over step i->i+1, binned by CO2 at i+1 and T0 at i, pooled over climates, validation year out."""
    num = np.zeros((N_BINS - 1, N_BINS - 1))
    den = np.zeros_like(num)
    for c, t0 in t0_by_climate.items():
        n = t0.sizes["time"]
        tend = (t0.values[1:] - t0.values[:-1]) / TIMESTEP_DAYS
        x = t0.values[:-1]
        y = co2[c].values[1:n]
        keep = np.array([t.year != VALIDATION_YEAR for t in t0.time.values[1:]])
        h_num, _, _ = np.histogram2d(
            y[keep], x[keep], bins=[CO2_BINS, T0_BINS], weights=tend[keep]
        )
        h_den, _, _ = np.histogram2d(y[keep], x[keep], bins=[CO2_BINS, T0_BINS])
        num += h_num
        den += h_den
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


target = binned_tendency(
    {
        f"{int(r.xCO2)}xCO2": series(r.results_ds, "target")
        for _, r in df[
            df.atmosphere == MODELS["random-CO$_2$ rs2\nbefore coupling"]
        ].iterrows()
    }
)
gens = {}
for label, atm in MODELS.items():
    sub = df[df.atmosphere == atm]
    if len(sub) == 0:
        continue
    gens[label] = binned_tendency(
        {f"{int(r.xCO2)}xCO2": series(r.results_ds, "gen") for _, r in sub.iterrows()}
    )

plotting.configure_style()
n = len(gens) + 1
fig, axes = plt.subplots(2, n, figsize=(2.3 * n + 1.2, 5.3), sharex=True, sharey=True)
fig.subplots_adjust(
    left=0.09, right=0.88, top=0.84, bottom=0.11, wspace=0.12, hspace=0.18
)
X, Y = np.meshgrid(T0_BINS, CO2_BINS)
kw = dict(cmap="RdBu_r", vmin=-1, vmax=1, shading="flat")
p1 = axes[0, 0].pcolormesh(X, Y, target, **kw)
axes[0, 0].set_title("CM4-like AM4")
axes[1, 0].set_visible(False)
for j, (label, g) in enumerate(gens.items(), start=1):
    p1 = axes[0, j].pcolormesh(X, Y, g, **kw)
    axes[0, j].set_title(label)
    p2 = axes[1, j].pcolormesh(X, Y, g - target, **kw)
for ax in axes.flat:
    ax.set_yscale("log")
    ax.invert_yaxis() if not ax.yaxis_inverted() else None
    ax.set_xticks([210, 215, 220, 225, 230])
    for lvl in (REFERENCE_CO2, 4 * REFERENCE_CO2):
        ax.axhline(lvl, color="gray", ls="--", lw=0.8)
for ax in axes[1, 1:]:
    ax.set_xlabel("Global mean T$_0$ [K]")
axes[0, 0].set_xlabel("Global mean T$_0$ [K]")
axes[0, 0].set_ylabel("CO$_2$ concentration\n[ppm]")
axes[1, 1].set_ylabel("CO$_2$ concentration\n[ppm]")
axes[1, 1].yaxis.set_tick_params(labelleft=True)
cax1 = fig.add_axes([0.895, 0.505, 0.012, 0.335])
cax2 = fig.add_axes([0.895, 0.11, 0.012, 0.335])
fig.colorbar(
    p1, cax=cax1, label=textwrap.fill("Global mean T$_0$ tendency [K/day]", 22)
)
fig.colorbar(
    p2, cax=cax2, label=textwrap.fill("Bias (ACE minus CM4-like AM4) [K/day]", 22)
)
fig.suptitle(
    "Random-CO$_2$ ic_0003 (1x, 2x, 4x pooled, validation year excluded): T$_0$ tendency vs CO$_2$ and T$_0$",
    y=0.985,
)
fig.savefig(os.path.join(FIG, "random-co2-T0-tendency-phase-space.png"), dpi=200)
print(
    "wrote",
    os.path.join(FIG, "random-co2-T0-tendency-phase-space.png"),
    "models:",
    list(gens),
)
