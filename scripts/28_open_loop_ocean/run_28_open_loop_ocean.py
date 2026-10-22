# ruff: noqa: E501
# mypy: ignore-errors
"""Issue 28: standalone Samudra forced by ACE's saved surface fluxes (arms O_CM4, O_A, O_T, O_C).

    python -I run_28_open_loop_ocean.py run <arm> <j> <ckpt> <out.nc> [--n-forward-steps N] [--config-dir D]
    python -I run_28_open_loop_ocean.py check <j> <O_CM4.nc> <out.yaml>

W = ace-exper-worktrees/exper-2026-10-21-coupled-sst-09-pilot/. `run` executes fme.ace.inference.evaluator.main on
V/ocean/evaluator-config-1pct-09stage2-<j>-O.yaml (V = configs/experiments/cm4_eval), changed only in data paths
(/climate-default -> gs://vcm-ml-intermediate), experiment_dir (local temp), wandb (off) and, for a smoke,
n_forward_steps. Stepper.step is wrapped: at ocean call tau (1-based) each phi in PHI_H is replaced in args.input
(the step + 1 slot of a next-step forcing) and args.next_step_input_data (read by the corrector) by

    g_X,phi(i, tau) = (1/n_in) sum_{k=1..n_in} phi_X(i, (tau - 1) n_in + k)

from arm X's atmosphere per_step.zarr (generated steps only), as CoupledStepper._get_ocean_forcings builds it.
Arm O_CM4 replaces each phi by the loader's own value at that slot through the same code. Wind stress and frozen
precipitation stay CM4's in every arm. Recorded per (sample, time) beside 09 arm O's per-step names:
Q = _compute_ocean_net_surface_energy_flux(next_step_input_data, input sst), n_prate_lt_frozen, and
rms_dswrf_lag_<l> = RMS over finite cells of g_X,DSWRFsfc(tau) - DSWRFsfc_CM4(tau + l).
`check` writes max |sst_OCM4 - sst_O| at tau = 1 per sample and per tau, against 09 arm O's per_step.zarr.
"""

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import xarray as xr
import yaml

CLIMATE_DEFAULT = ("/climate-default", "gs://vcm-ml-intermediate")
STAGE2 = "gs://vcm-ml-experiments/jamesd/2026-10-20-2117-coupled-ace-sst-representation/09-stage2"
STORES = {
    "O_A": "A/{j}/per_step.zarr",
    "O_T": "T/{j}/per_step.zarr",
    "O_C": "C/{j}/atmosphere/per_step.zarr",
}
O_STORE = "O/{j}/per_step.zarr"
PHI_H = [
    "DSWRFsfc",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "PRATEsfc",
]
FROZEN = "total_frozen_precipitation_rate"
LAGS = (-1, 0, 1)


def _open(path: str) -> xr.Dataset:
    import gcsfs

    return xr.open_zarr(gcsfs.GCSFileSystem().get_mapper(path.removeprefix("gs://")))


def _lead_ns(ds: xr.Dataset) -> np.ndarray:
    return ds["time"].values.astype("timedelta64[ns]").astype(np.int64)


def _init_times(ds: xr.Dataset) -> list[str]:
    return [t.strftime("%Y-%m-%dT%H:%M:%S") for t in ds["init_time"].values]


def _adapt(cfg: dict, exp_dir: str, n_forward_steps: int | None) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(cfg).replace(*CLIMATE_DEFAULT))
    cfg.update(experiment_dir=exp_dir)
    cfg["logging"].update(log_to_wandb=False, log_to_file=False)
    if n_forward_steps is not None:
        cfg.update(
            n_forward_steps=n_forward_steps,
            forward_steps_in_memory=min(
                n_forward_steps, cfg.get("forward_steps_in_memory", n_forward_steps)
            ),
        )
        cfg["aggregator"]["log_step_means"] = [{"step": n_forward_steps}]
    return cfg


def _window_means(arm: str, j: str, cfg: dict) -> np.ndarray:
    """g_X (tau, phi, sample, lat, lon), float32, for tau = 1 .. n_forward_steps."""
    n = cfg["n_forward_steps"]
    a = _open(f"{STAGE2}/{STORES[arm].format(j=j)}")
    o = _open(f"{STAGE2}/{O_STORE.format(j=j)}")
    n_in, rem = divmod(a.sizes["time"], o.sizes["time"])
    assert rem == 0 and n_in == 20, (a.sizes["time"], o.sizes["time"])
    assert _init_times(a) == cfg["loader"]["start_indices"]["times"], _init_times(a)
    lead_a, lead_o = _lead_ns(a), _lead_ns(o)
    assert (
        lead_a[n_in - 1 :: n_in] == lead_o
    ).all()  # window tau ends at ocean lead tau
    g = (
        a[PHI_H]
        .isel(time=slice(0, n * n_in))
        .coarsen(time=n_in)
        .mean()
        .transpose("time", "sample", "lat", "lon")
        .to_array("phi")
        .transpose("time", "phi", "sample", "lat", "lon")
        .astype("float32")
        .values
    )
    assert g.shape[0] == n, g.shape
    return g


def _wrap(g: np.ndarray | None, rec: dict) -> None:
    import torch

    from fme.ace.stepper import single_module
    from fme.core.corrector.ocean import _compute_ocean_net_surface_energy_flux
    from fme.core.step.args import StepArgs

    orig = single_module.Stepper.step

    def step(self, a, wrapper=lambda x: x):
        tau = len(rec["Q"]) + 1
        inp, nxt = dict(a.input), dict(a.next_step_input_data)
        cm4_dswrf = inp["DSWRFsfc"].detach().cpu().numpy()
        for p, phi in enumerate(PHI_H):
            src = inp[phi] if g is None else g[tau - 1, p]
            for d in (inp, nxt):
                if phi in d:
                    d[phi] = torch.as_tensor(
                        src, dtype=d[phi].dtype, device=d[phi].device
                    ).clone()
        a = StepArgs(
            input=inp,
            next_step_input_data=nxt,
            labels=a.labels,
            data_mask=a.data_mask,
            stepper_state=a.stepper_state,
        )
        forcing = {**inp, **nxt}
        rec["Q"].append(
            _compute_ocean_net_surface_energy_flux(forcing, inp["sst"])
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        rec["n_prate_lt_frozen"].append(
            (forcing["PRATEsfc"] < forcing[FROZEN]).sum(dim=(-2, -1)).cpu().numpy()
        )
        rec["dswrf_injected"].append(forcing["DSWRFsfc"].detach().cpu().numpy())
        rec["dswrf_cm4"].append(cm4_dswrf)
        print(f"28: ocean step tau={tau} injected", file=sys.stderr, flush=True)
        return orig(self, a, wrapper)

    single_module.Stepper.step = step


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(x**2)))


def run(
    arm: str,
    j: str,
    ckpt: str,
    out: str,
    n_forward_steps: int | None,
    config_dir: str,
) -> None:
    config = f"{config_dir}/ocean/evaluator-config-1pct-09stage2-{j}-O.yaml"
    exp_dir = tempfile.mkdtemp(prefix=f"28-{arm}-{j}-")
    cfg = _adapt(yaml.safe_load(open(config)), exp_dir, n_forward_steps)
    cfg["checkpoint_path"] = ckpt
    n = cfg["n_forward_steps"]
    g = None if arm == "O_CM4" else _window_means(arm, j, cfg)
    rec = {k: [] for k in ("Q", "n_prate_lt_frozen", "dswrf_injected", "dswrf_cm4")}
    _wrap(g, rec)
    cfg_path = pathlib.Path(exp_dir) / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    from fme.ace.inference.evaluator import main
    from fme.core.distributed import Distributed

    with Distributed.context():
        main(str(cfg_path))
    assert len(rec["Q"]) == n, (len(rec["Q"]), n)
    ds = xr.open_zarr(f"{exp_dir}/per_step.zarr").load()
    names = cfg["data_writer"]["files"][0]["names"]
    ds = ds[[v for v in names if v in ds]]
    assert ds.sizes["time"] == n, ds.sizes
    ds["Q"] = (
        ("sample", "time", "lat", "lon"),
        np.stack(rec["Q"], axis=1),
        {
            "units": "W/m**2",
            "long_name": "_compute_ocean_net_surface_energy_flux on injected forcing",
        },
    )
    ds["n_prate_lt_frozen"] = (
        ("sample", "time"),
        np.stack(rec["n_prate_lt_frozen"], axis=1),
        {"long_name": f"cells with PRATEsfc < {FROZEN}"},
    )
    inj, cm4 = np.stack(rec["dswrf_injected"]), np.stack(rec["dswrf_cm4"])
    for lag in LAGS:
        ds[f"rms_dswrf_lag_{lag}"] = (
            ("sample", "time"),
            np.array(
                [
                    [
                        _rms(inj[t, s] - cm4[t + lag, s])
                        if 0 <= t + lag < n
                        else np.nan
                        for t in range(n)
                    ]
                    for s in range(inj.shape[1])
                ]
            ),
            {
                "units": "W/m**2",
                "long_name": f"RMS of injected DSWRFsfc(tau) - CM4 DSWRFsfc(tau{lag:+d})",
            },
        )
    ds.attrs.update(
        arm=arm,
        j=j,
        config=config,
        checkpoint=ckpt,
        source=STORES.get(arm, "CM4 stores (loader)").format(j=j),
        script=__file__,
    )
    for v in ds.variables.values():
        v.encoding.clear()
    ds.to_netcdf(out)


def check(j: str, ocm4_nc: str, out_yaml: str) -> None:
    oc = xr.open_dataset(ocm4_nc)["sst"]
    o = _open(f"{STAGE2}/{O_STORE.format(j=j)}")["sst"].isel(
        time=slice(0, oc.sizes["time"])
    )
    eps = np.abs(oc.values - o.transpose(*oc.dims).values)
    per_tau = np.nanmax(eps, axis=(0, 2, 3))
    doc = {
        "definition": "eps_O = sst_OCM4 - sst_O (09 arm O per_step.zarr), max |.| over finite cells",
        "j": j,
        "max_abs_eps_O_tau1_per_sample": [
            float(x) for x in np.nanmax(eps[:, 0], axis=(1, 2))
        ],
        "max_abs_eps_O_per_tau": [float(x) for x in per_tau],
        "nan_mismatch": int((np.isnan(oc.values) != np.isnan(o.values)).sum()),
    }
    pathlib.Path(out_yaml).write_text(yaml.safe_dump(doc, sort_keys=False, width=120))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("arm", choices=["O_CM4", *STORES])
    r.add_argument("j", choices=["j01", "j04", "j07", "j10"])
    r.add_argument("ckpt")
    r.add_argument("out")
    r.add_argument("--n-forward-steps", type=int)
    r.add_argument("--config-dir", default="configs/experiments/cm4_eval")
    c = sub.add_parser("check")
    c.add_argument("j")
    c.add_argument("ocm4_nc")
    c.add_argument("out_yaml")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.arm, a.j, a.ckpt, a.out, a.n_forward_steps, a.config_dir)
    else:
        check(a.j, a.ocm4_nc, a.out_yaml)
