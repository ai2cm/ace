# ruff: noqa
# mypy: ignore-errors
"""One-figure trajectory of the random-CO2 ACE through the coupled fine-tune: per epoch (0 = before coupling), the
pooled CO2-response slopes (OLR and top-layer T as ratio to the AM4 target, T2m in K per doubling), the held-out R^2
of the global-mean T2m and T0 series (mean over the 1x/2x/4x members), and the coupled run's own inline whole-system
channel mean (mean of piC and 1pct, from wandb). The never-trained ACE2S is drawn as a dashed reference level."""

import os, re
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
import plotting

HERE = os.environ.get("RANDCO2_OUT", os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
COUPLED_RUN = os.environ.get("COUPLED_RUN", "ai2cm/ace-samudra-coupled-cm4/3geasxik")
BEFORE, CONTROL = "rco2 rs2 ep11, before", "ace2sft-rs0 (never saw randCO2), before"


def epoch_of(name):
    if name == BEFORE:
        return 0
    m = re.search(r"after coupled FT ep(\d+)", name)
    return int(m.group(1)) if m else None


slopes = pd.read_csv(os.path.join(HERE, "randco2_slopes.csv"))
held = pd.read_csv(os.path.join(HERE, "randco2_heldout_summary.csv"))
for d in (slopes,):
    d["epoch"] = d["atmosphere"].map(epoch_of)
held["epoch"] = held["model"].map(epoch_of)
ep = sorted(e for e in slopes.epoch.dropna().unique())


def slope_series(field, col):
    s = slopes[slopes.field == field].set_index("epoch")[col]
    c = slopes[(slopes.field == field) & (slopes.atmosphere == CONTROL)][col].iloc[0]
    t = slopes[slopes.field == field]["slope target"].iloc[0]
    return s.reindex(ep), c, t


def r2_series(var):
    h = held[held.variable == var]
    s = h.dropna(subset=["epoch"]).groupby("epoch")["r_squared"].mean().reindex(ep)
    c = h[h.model == CONTROL]["r_squared"].mean()
    return s, c


api = wandb.Api()
r = api.run(COUPLED_RUN)
hist = r.history(
    keys=[
        "epoch",
        "inference_piC/time_mean_norm/rmse/channel_mean",
        "inference_1pct/time_mean_norm/rmse/channel_mean",
    ],
    pandas=True,
)
hist = hist.dropna(subset=["inference_piC/time_mean_norm/rmse/channel_mean"]).set_index(
    "epoch"
)
coupled = (
    hist["inference_piC/time_mean_norm/rmse/channel_mean"]
    + hist["inference_1pct/time_mean_norm/rmse/channel_mean"]
) / 2

plotting.configure_style(fontsize=8)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.17, top=0.86, wspace=0.3)
C = ["#2a78d6", "#eb6834", "#1baf7a"]
ax = axes[0]
for (field, lab), c in zip([("ULWRFtoa", "OLR"), ("air_temperature_0", "T$_0$")], C):
    s, ctrl, _ = slope_series(field, "ratio")
    ax.plot(s.index, s.values, "o-", color=c, label=f"{lab} slope / target")
    ax.axhline(ctrl, color=c, ls="--", lw=1)
ax.axhline(1, color="k", lw=0.8)
ax.set_ylabel("slope / AM4 target")
ax.set_title("OLR and T$_0$ response per CO$_2$ doubling")
ax.legend(frameon=False)
ax = axes[1]
s, ctrl, t = slope_series("TMP2m", "slope gen")
ax.plot(s.index, s.values, "o-", color=C[2], label="rs2 through coupled FT")
ax.axhline(ctrl, color=C[2], ls="--", lw=1, label="never-trained ACE2S")
ax.axhline(t, color="k", lw=0.8, label="AM4 target")
ax.set_ylabel("K per doubling")
ax.set_title("T2m response per CO$_2$ doubling (spurious)")
ax.legend(frameon=False)
ax = axes[2]
for (var, lab), c in zip([("TMP2m", "T2m"), ("air_temperature_0", "T$_0$")], C):
    s, ctrl = r2_series(var)
    ax.plot(s.index, s.values, "o-", color=c, label=f"R² {lab}")
    ax.axhline(ctrl, color=c, ls="--", lw=1)
ax.set_ylim(0, 1.02)
ax.set_ylabel("held-out R² (mean of 1x, 2x, 4x)")
ax.set_title("Global-mean series vs AM4, ic_0003")
ax.legend(frameon=False)
ax = axes[3]
ax.plot(coupled.index, coupled.values, "o-", color="#8e5bd6")
ax.set_ylabel("channel mean (mean of piC, 1pct)")
ax.set_title("Coupled inline score of the rs2 + ppad pair")
for ax in axes:
    ax.set_xlabel("coupled-FT epoch (0 = before)")
    ax.set_xticks(range(0, 9))
    ax.grid(alpha=0.25)
fig.suptitle(
    "Random-CO$_2$ ACE2S rs2 through coupled fine-tuning with the ppad ocean: CO$_2$ response (ACE-only, prescribed SST) vs coupled skill; dashed = never-trained ACE2S",
    fontsize=9,
)
fig.savefig(os.path.join(FIG, "random-co2-epoch-summary.png"), dpi=200)
print("wrote", os.path.join(FIG, "random-co2-epoch-summary.png"), "epochs", ep)
