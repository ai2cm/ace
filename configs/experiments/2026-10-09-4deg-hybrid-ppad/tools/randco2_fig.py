# ruff: noqa: E501
# mypy: ignore-errors
"""Figure: CO2 response of each ACE atmosphere on the random-CO2 ic_0003 members.
Pooled over the 1x/2x/4x members: monthly global-mean anomalies (member mean + mean annual
cycle removed) of OLR, top-layer T and T2m against log2 CO2, target vs generated, with the
fitted slope (W/m2 or K per doubling). Reads cache/<results_ds>/monthly_mean_*.nc via
randco2_compare.csv.
"""

import os

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.environ.get("RANDCO2_OUT", os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(HERE, "randco2_compare.csv")).dropna(
    subset=["results_ds"]
)
df = df[df.atmosphere.str.contains("before|after")]  # ours (have monthly files)
ORDER = [
    "ace2sft-rs0 (never saw randCO2), before",
    "ace2sft-rs0, after coupled FT ep1 (lifted)",
    "rco2 rs2 ep11, before",
]
ORDER += sorted(
    [a for a in set(df.atmosphere) if a.startswith("rco2 rs2, after")],
    key=lambda a: int(a.split("ep")[1].split()[0]),
)
_pal = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#8e5bd6",
    "#d6498f",
    "#5b6b7a",
    "#0f8fa8",
    "#a0522d",
    "#6b8e23",
    "#444444",
]
COL = {a: _pal[i % len(_pal)] for i, a in enumerate(ORDER)}
FIELDS = [
    ("ULWRFtoa", "OLR anomaly (W/m²)"),
    ("air_temperature_0", "top-layer T anomaly (K)"),
    ("TMP2m", "2 m T anomaly (K)"),
]


def anoms(ds_id, f):
    T = xr.open_dataset(os.path.join(HERE, "cache", ds_id, "monthly_mean_target.nc"))
    G = xr.open_dataset(
        os.path.join(HERE, "cache", ds_id, "monthly_mean_predictions.nc")
    )
    w = np.cos(np.deg2rad(T.lat))

    def gm(D, v):
        return D[v].weighted(w).mean(("lat", "lon")).squeeze().values

    co2 = gm(T, "carbon_dioxide")
    lc = np.log2(co2)
    lc -= lc.mean()
    out = {}
    for src, D in (("tgt", T), ("gen", G)):
        x = gm(D, f)
        mon = np.arange(len(x)) % 12
        out[src] = x - pd.Series(x).groupby(mon).transform("mean").values
    return lc, out


rows = []
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (f, lab) in zip(axes, FIELDS):
    first_t = True
    for atm in [a for a in ORDER if a in set(df.atmosphere)]:
        X, Yg, Yt = [], [], []
        for _, r in df[df.atmosphere == atm].iterrows():
            lc, a = anoms(r.results_ds, f)
            X.append(lc)
            Yg.append(a["gen"])
            Yt.append(a["tgt"])
        X, Yg, Yt = map(np.concatenate, (X, Yg, Yt))
        sg, st = np.polyfit(X, Yg, 1)[0], np.polyfit(X, Yt, 1)[0]
        n_mem = (df.atmosphere == atm).sum()
        rows.append(
            {
                "atmosphere": atm,
                "field": f,
                "members": n_mem,
                "slope gen": sg,
                "slope target": st,
                "ratio": sg / st,
                "resid rmse": np.sqrt(np.mean((Yg - Yt) ** 2)),
            }
        )
        if first_t:
            ax.scatter(X, Yt, s=9, c="#9a9893", label="CM4-like AM4 target", zorder=1)
            xx = np.array([X.min(), X.max()])
            ax.plot(xx, st * xx, c="#52514e", lw=2, zorder=2)
            first_t = False
        xx = np.array([X.min(), X.max()])
        ax.plot(
            xx, sg * xx, c=COL[atm], lw=2, label=f"{atm}: {sg:+.2f}/doubling", zorder=3
        )
    ax.set_xlabel("log₂ CO₂ (member mean removed)")
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25)
    ax.set_title(f"target slope {st:+.2f} per doubling", fontsize=10)
axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle(
    "ACE-only rollouts on random-CO2 ic_0003 (1x, 2x, 4x pooled): monthly global-mean anomaly vs CO2",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "randco2_response.png"), dpi=150)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(HERE, "randco2_slopes.csv"), index=False)
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
