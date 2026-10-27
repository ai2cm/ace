# ruff: noqa: E501
# mypy: ignore-errors
"""Compare ACE atmospheres on the random-CO2 ic_0003 evaluations.

For each (tag, CO2 level) evaluator run: wandb time-mean RMSE/bias of key fields and,
from the result dataset, the per-step global-mean gen/target series (mean_diagnostics.nc)
and the monthly files (carbon_dioxide forcing) -> CO2 response regression.
Usage: python randco2_compare.py  (edit RUNS below)
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import wandb
import xarray as xr

OUT = os.environ.get("RANDCO2_OUT", os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
PROJECT = "ai2cm/ace2-cm4-atmos-noLSM"
# RANDCO2_RES=1deg scores the 1deg twin (configs/experiments/2026-10-22-1deg-randco2-replay: rs1 atmosphere, watchdog
# tags rco2rs1-1deg-{alt,alt-replay}-epN, wandb group ace1deg-randco2-before-after-coupled-ft); default = the 4deg set
RES = os.environ.get("RANDCO2_RES", "4deg")
PREFIX = f"ace{RES}-randco2"
# tag -> wandb run-name pattern with {x} for the CO2 multiplier
# add "rco2 rs2, after coupled FT (lifted)": "ace4deg-randco2-{x}xCO2-ic3-rco2rs2-after-jointens-ep<N>"
# once the rco2rs2 coupled FT has a checkpoint (tools/lift_atmosphere_and_eval.sh)
RUNS = {
    "ace2sft-rs0 (never saw randCO2), before": "ace4deg-randco2-{x}xCO2-ic3-ace2sft-rs0-before",
    "rco2 rs2 ep11, before": "ace4deg-randco2-{x}xCO2-ic3-rco2rs2-before",
    "rco2 rs2 ep3 (Spencer)": "rco2v2_ic12_1pct_46to125_piC_156to235-ace2s_origLR_4deg-ft-ep3-eval_best_inference_ckpt-randco2-{x}xCO2-ic3-rs2",
    "rco2 rs0 ep3 (Spencer)": "rco2v2_ic12_1pct_46to125_piC_156to235-ace2s_origLR_4deg-ft-ep3-eval_best_inference_ckpt-randco2-{x}xCO2-ic3-rs0",
}
FIELDS = [
    "TMP2m",
    "ULWRFtoa",
    "USWRFtoa",
    "PRATEsfc",
    "h500",
    "TMP850",
    "air_temperature_0",
    "air_temperature_7",
    "specific_total_water_7",
    "ULWRFsfc",
    "DLWRFsfc",
]

api = wandb.Api()
# per-epoch "after" atmospheres from the watchdog (tools/watch_coupled_ft_epochs.sh)
import re as _re

_after = {}
_LABEL_1DEG = {
    "rco2rs1-1deg-alt": "rco2 rs1, alternating coupled FT ep{ep} (lifted, EMA)",
    "rco2rs1-1deg-alt-replay": "rco2 rs1, alternating coupled FT + random-CO2 replay ep{ep} (lifted, EMA)",
}
_LABEL = {
    "rco2rs2-after-jointens": "rco2 rs2, after coupled FT ep{ep} (lifted, EMA)",
    "rco2rs2-replay": "rco2 rs2, coupled FT + random-CO2 replay ep{ep} (lifted, EMA)",
    "rco2rs2-replay-oceanonly": "rco2 rs2, ocean-only coupled FT + random-CO2 replay ep{ep} (lifted, EMA)",
    "rco2rs2-resid-replay": "rco2 rs2, hybrid-residual ocean, alternating coupled FT + random-CO2 replay ep{ep} (lifted, EMA)",
    "rco2rs2-spb-rco2rs2": "rco2 rs2, alternating coupled FT ep{ep} (lifted, EMA)",
    "rco2rs2-spb-rco2rs2-l2sp1e-2": "rco2 rs2, alternating coupled FT + weight anchor 0.01 ep{ep} (lifted, EMA)",
    "rco2rs2-spb-rco2rs2-l2sp1": "rco2 rs2, alternating coupled FT + weight anchor 1 ep{ep} (lifted, EMA)",
    "rco2rs2-spb-rco2rs2-l2sp100": "rco2 rs2, alternating coupled FT + weight anchor 100 ep{ep} (lifted, EMA)",
    "rco2rs2-fta-lr1e-5": "rco2 rs2, FTA after FTO lr 1e-5 ep{ep} (lifted, EMA)",
    "rco2rs2-fta-lr7e-6": "rco2 rs2, FTA after FTO lr 7e-6 ep{ep} (lifted, EMA)",
    "rco2rs2-fta-lr1e-5-1yr": "rco2 rs2, FTA after FTO lr 1e-5, 1 yr per scenario ep{ep} (lifted, EMA)",
    "rco2rs2-fta-lr1e-5-10yr": "rco2 rs2, FTA after FTO lr 1e-5, 10 yr per scenario ep{ep} (lifted, EMA)",
    "rco2rs2-fta-encdec": "rco2 rs2, FTA after FTO, encoder+decoder only ep{ep} (lifted, EMA)",
    "rco2rs2-fta-lastblock": "rco2 rs2, FTA after FTO, last block+decoder only ep{ep} (lifted, EMA)",
    "rco2rs2-fta-firstblock": "rco2 rs2, FTA after FTO, encoder+first block only ep{ep} (lifted, EMA)",
}
if RES == "1deg":
    _LABEL = _LABEL_1DEG
    RUNS = {
        "ace2sft-rs0 (never saw randCO2), before": PREFIX
        + "-{x}xCO2-ic3-ace2sft-rs0-before",
        "rco2 rs1 (1deg), before": PREFIX + "-{x}xCO2-ic3-rco2rs1-before",
    }
for _r in api.runs(
    PROJECT,
    filters={"group": f"{PREFIX}-before-after-coupled-ft"},
    order="-created_at",
):
    _m = _re.match(
        PREFIX
        + r"-(\d)xCO2-ic3-(("
        + "|".join(sorted(_LABEL, key=len, reverse=True))
        + r")-(ep\d+|best))$",
        _r.name,
    )
    if _m:
        _ep = _m.group(4)
        _after[(_m.group(3), 99 if _ep == "best" else int(_ep[2:]))] = (
            _m.group(2),
            _ep,
        )
for _kind, _order in sorted(_after):
    _pat, _ep = _after[(_kind, _order)]
    _lab = (
        _LABEL[_kind].replace("ep{ep}", "selected ckpt")
        if _ep == "best"
        else _LABEL[_kind].format(ep=_ep[2:])
    )
    RUNS[_lab] = PREFIX + "-{x}xCO2-ic3-" + _pat
runs = {}
for r in api.runs(
    PROJECT, filters={"jobType": "inference"}, order="-created_at", per_page=600
):
    # newest first: keep the newest FINISHED run of a name (preempted evaluators restart under a new run id)
    if r.name not in runs or (
        runs[r.name].state != "finished" and r.state == "finished"
    ):
        runs[r.name] = r


def beaker_result(run):
    # the job's results dataset: look it up from the beaker experiment with the same name
    out = subprocess.run(
        [
            "beaker",
            "experiment",
            "get",
            f"{run.config.get('_beaker_user','troya') if False else ''}{run.name}",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        for user in ("troya", "spencerc"):
            out = subprocess.run(
                [
                    "beaker",
                    "experiment",
                    "get",
                    f"{user}/{run.name}",
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                break
    if out.returncode != 0:
        return None
    e = json.loads(out.stdout)[0]
    return e["jobs"][-1]["result"]["beaker"]


def fetch(ds, prefix):
    d = os.path.join(CACHE, ds)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, prefix)
    if not os.path.exists(p):
        subprocess.run(
            ["beaker", "dataset", "fetch", ds, "--prefix", prefix, "-o", d],
            check=True,
            capture_output=True,
        )
    return p if os.path.exists(p) else None


rows, series = [], {}
for tag, pat in RUNS.items():
    for x in (1, 2, 4):
        name = pat.format(x=x)
        r = runs.get(name)
        if r is None or r.state != "finished":
            print(f"missing/unfinished: {name} ({None if r is None else r.state})")
            continue
        s = r.summary
        row = {
            "atmosphere": tag,
            "xCO2": x,
            "channel_mean": s.get("inference/time_mean_norm/rmse/channel_mean"),
        }
        for f in FIELDS:
            row[f"rmse {f}"] = s.get(f"inference/time_mean/rmse/{f}")
            row[f"bias {f}"] = s.get(f"inference/time_mean/bias/{f}")
        ds = beaker_result(r)
        row["results_ds"] = ds
        if ds:
            p = fetch(ds, "mean_diagnostics.nc")
            if p:
                m = xr.open_dataset(p)
                series[(tag, x)] = m
                # anomaly correlation of global-mean series (daily means of 6-h steps)
                for f in (
                    "ULWRFtoa",
                    "TMP2m",
                    "air_temperature_0",
                    "air_temperature_7",
                ):
                    g = m[f"weighted_mean_gen-{f}"].values
                    t = m[f"weighted_mean_target-{f}"].values
                    n = (len(g) // 4) * 4
                    g = g[:n].reshape(-1, 4).mean(1)
                    t = t[:n].reshape(-1, 4).mean(1)
                    # remove the mean annual cycle crudely: 365-day running anomalies via day-of-year mean
                    doy = np.arange(len(g)) % 365
                    ga = g - pd.Series(g).groupby(doy).transform("mean").values
                    ta = t - pd.Series(t).groupby(doy).transform("mean").values
                    row[f"anom corr {f}"] = np.corrcoef(ga, ta)[0, 1]
                    row[f"anom std ratio {f}"] = ga.std() / ta.std()
            # monthly files: carbon_dioxide in the target file -> regression on log2 CO2
            pm_t = fetch(ds, "monthly_mean_target.nc")
            pm_g = fetch(ds, "monthly_mean_predictions.nc")
            if pm_t and pm_g:
                T = xr.open_dataset(pm_t)
                G = xr.open_dataset(pm_g)
                if "carbon_dioxide" in T:
                    w = np.cos(np.deg2rad(T.lat))
                    co2 = (
                        T["carbon_dioxide"]
                        .weighted(w)
                        .mean(("lat", "lon"))
                        .squeeze()
                        .values
                    )
                    for f in ("ULWRFtoa", "TMP2m", "air_temperature_0"):
                        if f not in G:
                            continue
                        gt = G[f].weighted(w).mean(("lat", "lon")).squeeze().values
                        tt = T[f].weighted(w).mean(("lat", "lon")).squeeze().values
                        mon = np.arange(len(gt)) % 12
                        ga = gt - pd.Series(gt).groupby(mon).transform("mean").values
                        ta = tt - pd.Series(tt).groupby(mon).transform("mean").values
                        lc = np.log2(co2 / co2.mean())
                        row[f"dOLR/dlog2CO2 gen {f}"] = np.polyfit(lc, ga, 1)[0]
                        row[f"dOLR/dlog2CO2 tgt {f}"] = np.polyfit(lc, ta, 1)[0]
        rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
df.to_csv(os.path.join(OUT, "randco2_compare.csv"), index=False)
cols = [
    "atmosphere",
    "xCO2",
    "channel_mean",
    "rmse TMP2m",
    "bias TMP2m",
    "rmse ULWRFtoa",
    "bias ULWRFtoa",
    "bias air_temperature_0",
    "rmse PRATEsfc",
    "anom corr ULWRFtoa",
    "anom corr air_temperature_0",
    "anom std ratio ULWRFtoa",
]
print(
    df[[c for c in cols if c in df]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"
    )
)
reg = [c for c in df.columns if c.startswith("dOLR")]
if reg:
    print(
        df[["atmosphere", "xCO2"] + reg].to_string(
            index=False, float_format=lambda v: f"{v:.3g}"
        )
    )
