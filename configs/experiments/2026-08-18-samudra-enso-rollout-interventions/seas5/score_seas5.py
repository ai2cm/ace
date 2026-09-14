"""Score SEAS5 ensemble-mean Nino3.4 vs our 99-IC protocol.

Protocol match: init months 2002-01..2010-03; anomalies vs SEAS5's own
lead-dependent 1993-2016 hindcast climatology (standard drift removal);
verified against the same replay-derived truth series used for the model
scores (scratchpad cpl40fix/scores.npz tru/inits), leads 1-6.
"""

import numpy as np
import xarray as xr

SEAS5 = "seas5_data"
SCORES = (
    "/tmp/claude-1002/-home-troya-reports/72506aa0-f488-4bce-a119-3cb27f1b358f/"
    "scratchpad/cpl40fix/scores.npz"
)

z = np.load(SCORES)
inits, tru = z["inits"], z["tru"]

frames = []
for y in range(1993, 2017):
    frames.append(xr.open_dataset(f"{SEAS5}/seas5_sst_{y}.nc"))
ds = xr.concat(frames, dim="forecast_reference_time")
sst = ds["sst"] if "sst" in ds else ds[list(ds.data_vars)[0]]
# ens-mean nino34 box mean per (init, lead)
box = sst.sel(latitude=slice(5, -5), longitude=slice(-170, -120)).mean(
    ("latitude", "longitude")
)
ens = box.mean("number")
# climatology per (init calendar month, lead) over 1993-2016
ref = ens.groupby("forecast_reference_time.month").mean()
anom = ens.groupby("forecast_reference_time.month") - ref


def ym(t):
    t = np.datetime64(t, "M")
    return int(str(t)[:4]) * 12 + int(str(t)[5:7]) - 1


rows = {ym(t): i for i, t in enumerate(anom["forecast_reference_time"].values)}
NL = 6
P = np.full((len(inits), NL), np.nan)
for i, iym in enumerate(inits):
    j = rows.get(int(iym))
    if j is None:
        continue
    P[i, :] = anom.isel(forecast_reference_time=j).values[:NL]
acc = []
for k in range(NL):
    m = np.isfinite(P[:, k]) & np.isfinite(tru[:, k])
    x, y = P[m, k], tru[m, k]
    acc.append(np.corrcoef(x - np.mean(x), y - np.mean(y))[0, 1])
print("SEAS5 ens-mean ACC leads 1-6:", " ".join(f"{a:.2f}" for a in acc))
np.save("seas5_acc.npy", np.array(acc))
