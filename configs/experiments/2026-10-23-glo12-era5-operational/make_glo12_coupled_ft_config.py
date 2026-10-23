# ruff: noqa: E501
#!/usr/bin/env python
"""Derive the GLO12 operational coupled fine-tune config from the best UFS coupled system's config.

Base: ../2026-10-14-ufs4deg-era5-coupling/learnedheat-ftcm4-ufsstats-era5-coupledft-spb.yaml (alternating coupled FT,
both components trained, EnsembleLoss, one component per batch). Changes for GLO12 (Troy 2026-10-23: fine-tune the best
coupled SamudrACE-UFS with ERA5 on GLO12, leave 2025 out of training, alternating updates):
- ocean store: the derived GLO12 4deg 5-day store (zero placeholder sfdsi/UI/VI), atmosphere: ERA5 4deg daily relabelled
  -24 h with GLO12 fractions; stamps 00Z.
- training 2022-06-03 .. 2024-12-31 (189 five-day states), validation 2025 (loss logged only), inline inference inside the
  training years (2023-07 and 2024-01 starts, 72 coupled steps) for monitoring only; every epoch checkpoint is saved so
  2025 hindcasts can pick the epoch afterwards without 2025 touching the selection.
- the five channels GLO12 cannot supervise (hfds_total_area, tauuo, tauvo, wfo, unaccounted_heating) stay in out_names
  (parameter_init cannot shrink the output layer: 'Dest parameter has size 83 ... loaded 88'), are zero-filled in the derived
  store and get an ocean loss weight of 0; the surface-flux correction overwrites hfds_total_area with the ERA5 rebuild and
  the heat-content corrector uses a constant (zero) unaccounted heating instead of the generated channel.
- stats: GLO12 2022-06..2024-12 ocean stats (+ the UFS bundle's placeholder entries), ERA5 atmosphere stats unchanged.
- epochs 30 at lr 7e-5 (189 samples/epoch, ~23 optimizer steps; warm-up over the first epoch).
"""

import pathlib

import fsspec
import numpy as np
import xarray as xr
import yaml

HERE = pathlib.Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "2026-10-14-ufs4deg-era5-coupling"
    / "learnedheat-ftcm4-ufsstats-era5-coupledft-spb.yaml"
)
OUT = HERE / "glo12-era5-coupledft-spb.yaml"
OCEAN_DIR, OCEAN_ZARR = (
    "/glo12",
    "2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025-cm4vars-v2.zarr",
)
ATM_DIR, ATM_ZARR = (
    "/era5glo",
    "2026-10-23-era5-4deg-8layer-daily-glo12-coupled-2022-2025.zarr",
)
DROP = ["hfds_total_area", "tauuo", "tauvo", "wfo", "unaccounted_heating"]

c = yaml.safe_load(open(BASE))
times = xr.open_zarr(
    fsspec.get_mapper(
        "gs://vcm-ml-intermediate/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr"
    ),
    consolidated=True,
).time.values


def first_at_or_after(s):
    t = times[times >= np.datetime64(s)][0]
    return str(t)[:19]


t_train0, t_train1 = str(times[0])[:19], first_at_or_after("2024-12-25")
t_val0 = str(times[times > np.datetime64(t_train1)][0])[:19]
inl = [first_at_or_after("2023-07-01"), first_at_or_after("2024-01-01")]
print(
    "train",
    t_train0,
    "..",
    t_train1,
    "| validation from",
    t_val0,
    "| inline starts",
    inl,
    "| n train states",
    int((times <= np.datetime64(t_train1)).sum()),
)


def ds_block(sub=None):
    o: dict = {"data_path": OCEAN_DIR, "file_pattern": OCEAN_ZARR, "engine": "zarr"}
    a: dict = {"data_path": ATM_DIR, "file_pattern": ATM_ZARR, "engine": "zarr"}
    if sub:
        o["subset"] = dict(sub)
        a["subset"] = dict(sub)
    return {"ocean": o, "atmosphere": a}


c["max_epochs"] = 30
c["save_best_inference_epoch_checkpoints"] = True
c["checkpoint_save_epochs"] = {"start": 0, "stop": None, "step": 1}
c["train_loader"]["dataset"] = {
    "concat": [ds_block({"start_time": t_train0, "stop_time": t_train1})]
}
c["validation"] = [
    {
        "name": "val_2025",
        "evaluate_all_steps": False,
        "weight": 1.0,
        "loader": {
            "batch_size": 8,
            "num_data_workers": 2,
            "prefetch_factor": 2,
            "dataset": ds_block({"start_time": t_val0}),
        },
    }
]
c["inference"] = [
    {
        "name": "inference_1yr_insample",
        "weight": 1.0,
        "n_coupled_steps": 72,
        "coupled_steps_in_memory": 2,
        "loader": {
            "num_data_workers": 2,
            "start_indices": {"times": inl},
            "dataset": ds_block(),
        },
        "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
    }
]
c["optimization"]["scheduler"]["kwargs"]["total_iters"] = 23
oc = c["stepper"]["ocean"]["stepper"]["step"]["config"]
assert oc["out_names"][-5:] == DROP, oc["out_names"][-6:]
ol = c["stepper_training"]["ocean"]["loss"]
ol["weights"] = {**ol.get("weights", {}), **{v: 0.0 for v in DROP}}
ohc = oc["corrector"]["config"]["ocean_heat_content_correction"]
ohc["unaccounted_heating_source"] = "constant"
ohc.pop("unaccounted_heating_name", None)
for k in ("next_step_forcing_names",):
    if k in oc:
        oc[k] = [v for v in oc[k] if v not in DROP]
assert c["stepper_training"]["optimize_single_component_per_batch"] is True
header = (
    "# glo12-era5-coupledft-spb: ALTERNATING coupled fine-tune (both components trained, one per batch, EnsembleLoss) of the best\n"
    "# coupled SamudrACE-UFS system (ocean + atmosphere lifted from samudra-ufs4deg-era5-learnedheat-ftcm4-ufsstats-era5-coupledft-spb,\n"
    "# ckpt 01M5DXQST7WCYTXYXKMEF3KQ13) on the GLO12 operational analysis (4deg, 5-day) with ERA5, training 2022-06..2024-12,\n"
    "# 2025 held out for operational-forecast verification. Generated by make_glo12_coupled_ft_config.py (see its docstring).\n"
)
OUT.write_text(header + yaml.safe_dump(c, sort_keys=False, width=120))
print(
    "wrote",
    OUT,
    "| ocean out",
    len(oc["out_names"]),
    "in",
    len(oc["in_names"]),
    "| zero-weight:",
    [v for v, w in ol["weights"].items() if w == 0],
)
