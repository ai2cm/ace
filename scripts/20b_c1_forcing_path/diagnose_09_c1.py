# ruff: noqa: E501
# mypy: ignore-errors
"""Issue 20b: C1 forcing-path diagnosis — record and compare the atmosphere stepper inputs of arms C and T.

    python diagnose_09_c1.py omega <omega.nc>                                    # local, ace-tools env
    PYTHONPATH=<copy of W/fme at f7718ac> python diagnose_09_c1.py run <arm> <config.yaml> <ckpt> <out.nc> \
        [--swap-from <C.nc> --swap-forcing v,.. --swap-ic v,..]                  # james-vm, fme env, CPU
    PYTHONPATH=... python diagnose_09_c1.py compare <C.nc> <T.nc> <omega.nc> <out.yaml> [<label>=<T_swap.nc> ...]

W = ace-exper-worktrees/exper-2026-10-21-coupled-sst-09-pilot/. `run` executes fme's own evaluator (arm T:
fme.ace.inference.evaluator.main; arm C: fme.coupled.inference.evaluator.main) on a stage-1 config, changed only in
length (`--n-windows` ocean steps), checkpoint and data paths, writers and wandb; `Stepper.step` is wrapped to record each
atmosphere call: `in__<v>` (args.input), `next__<v>` (args.next_step_input_data), `out__<v>` (output), dim `k`
(k = 1 is the first inner step). Swaps replace T's inputs by C's recorded ones: forcing names at every k,
IC names at the first inner step of each window
(where C's `_prescribe_ic_sst` acts).
"""

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import xarray as xr
import yaml

HERE = pathlib.Path(__file__).parent
CLIMATE_DEFAULT = ("/climate-default", "gs://vcm-ml-intermediate")
N_INNER = 20  # ocean 5D / atmosphere 6h (train-config.yaml `timedelta`); checked against the recorded call count
ATMOS_KEY = "PRESsfc"  # present in atmosphere inputs only


def _adapt(cfg: dict, arm: str, ckpt: str, exp_dir: str, n_windows: int) -> dict:
    """Stage-1 config -> n_windows ocean steps, local checkpoint, gs:// data, no writers, no wandb."""
    cfg = yaml.safe_load(yaml.safe_dump(cfg).replace(*CLIMATE_DEFAULT))
    cfg.update(experiment_dir=exp_dir, checkpoint_path=ckpt)
    cfg["logging"].update(log_to_wandb=False, log_to_file=False)
    if arm == "C":
        cfg.update(n_coupled_steps=n_windows, coupled_steps_in_memory=n_windows)
        for realm in ("ocean", "atmosphere"):
            cfg["data_writer"][realm]["files"] = []
    else:
        cfg.update(
            n_forward_steps=n_windows * N_INNER,
            forward_steps_in_memory=n_windows * N_INNER,
        )
        cfg["data_writer"]["files"] = []
    return cfg


def _record(swap: dict | None):
    import torch

    from fme.ace.stepper import single_module
    from fme.core.step.args import StepArgs

    rec: list[dict] = []
    orig = single_module.Stepper.step

    def step(self, a, wrapper=lambda x: x):
        if ATMOS_KEY not in a.input:
            return orig(self, a, wrapper)
        k = len(rec)
        if swap is not None:
            src = swap["ds"]
            inp, nxt = dict(a.input), dict(a.next_step_input_data)
            names = list(swap["forcing"]) + (
                list(swap["ic"]) if k % N_INNER == 0 else []
            )  # window starts
            for v in names:
                for d, pre in ((inp, "in__"), (nxt, "next__")):
                    if v in d:
                        d[v] = torch.as_tensor(
                            src[pre + v].isel(k=k).values,
                            dtype=d[v].dtype,
                            device=d[v].device,
                        )
            a = StepArgs(
                input=inp,
                next_step_input_data=nxt,
                labels=a.labels,
                data_mask=a.data_mask,
                stepper_state=a.stepper_state,
            )
        result = orig(self, a, wrapper)
        rec.append(
            {
                **{"in__" + n: t for n, t in a.input.items()},
                **{"next__" + n: t for n, t in a.next_step_input_data.items()},
                **{"out__" + n: t for n, t in result.output.items()},
            }
        )
        print(
            f"20b: atmosphere step k={len(rec)} recorded", file=sys.stderr, flush=True
        )
        return result

    single_module.Stepper.step = step
    return rec


def run(
    arm: str,
    config: str,
    ckpt: str,
    out: str,
    swap_from: str | None,
    swap_forcing: str,
    swap_ic: str,
    n_windows: int,
) -> None:
    cfg = _adapt(
        yaml.safe_load(open(config)),
        arm,
        ckpt,
        tempfile.mkdtemp(prefix=f"20b-{arm}-"),
        n_windows,
    )
    swap = None
    if swap_from:
        swap = {
            "ds": xr.open_dataset(swap_from),
            "forcing": [v for v in swap_forcing.split(",") if v],
            "ic": [v for v in swap_ic.split(",") if v],
        }
    rec = _record(swap)
    cfg_path = pathlib.Path(cfg["experiment_dir"]) / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    if arm == "C":
        from fme.coupled.inference.evaluator import main
    else:
        from fme.ace.inference.evaluator import main
    main(str(cfg_path))
    assert len(rec) == n_windows * N_INNER, (len(rec), n_windows * N_INNER)
    names = sorted(rec[0])
    ds = xr.Dataset(
        {
            n: (
                ("k", "sample", "lat", "lon"),
                np.stack([r[n].detach().cpu().float().numpy() for r in rec]),
            )
            for n in names
        },
        coords={"k": np.arange(1, len(rec) + 1)},
    )
    ds.attrs.update(
        arm=arm,
        config=config,
        checkpoint=ckpt,
        swap_from=swap_from or "",
        swap_forcing=swap_forcing,
        swap_ic=swap_ic,
    )
    ds.to_netcdf(out)


def omega(out: str) -> None:
    sys.path.insert(0, str(HERE))
    import _common as c
    from check_09_stage1_verdict import masks

    s = c.open_store("5d")
    on = {"lat": s["lat"].values, "lon": s["lon"].values}
    area, om = masks(on)
    land = c.open_store("ocean")["land_fraction"].load().assign_coords(on)
    xr.Dataset(
        {
            "omega": om.astype("int8"),
            "areacello": area,
            "land_fraction_ocean_store": land,
        }
    ).to_netcdf(out)


def _update_yaml(path: str, key: str, value) -> None:
    p = pathlib.Path(path)
    doc = yaml.safe_load(p.read_text()) if p.exists() else {}
    doc[key] = value
    p.write_text(yaml.safe_dump(doc, sort_keys=False, width=120))


def _rms(f: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt((f**2 * w).sum() / w.sum()))


def schedule(
    config_c: str, omega_nc: str, out_yaml: str, half: int = 2 * N_INNER
) -> None:
    """SST schedule on Ω around each IC t_i, 6-h index j relative to t_i (j = 0 is t_i).

    changes: j where max_Ω |ts[j] − ts[j−1]| > 0, per store. hold_window: for the hold covering j = 0 … N_INNER−1,
    a-weighted RMS on Ω of ts_X[j] − mean(ts_CM4[a … b]) for candidate CM4 windows [a, b]. shift: RMS of
    ts_si[j] − ts_ca[j + s]. ic_sst: max_Ω |ts_ca[0] − (5-daily sst at t_i)| (C's window-1 SST is that sst).
    """
    sys.path.insert(0, str(HERE))
    import _common as c

    om = xr.open_dataset(omega_nc)
    w = np.where(om["omega"].values == 1, om["areacello"].fillna(0).values, 0.0)
    on = w > 0
    times = yaml.safe_load(open(config_c))["loader"]["start_indices"]["times"]
    stores = {
        "ts_ca": c.open_store("ca"),
        "ts_si": c.open_store("si"),
        "ts_cm4": c.open_store("atmos"),
    }
    sst5 = c.open_store("5d")["sst"]
    out = {
        "definition": schedule.__doc__.strip(),
        "j_range": [-half, half],
        "n_inner": N_INNER,
        "ics": {},
    }
    cands = {
        "[-20,-1]": (-N_INNER, -1),
        "[-19,0]": (-N_INNER + 1, 0),
        "[0,19]": (0, N_INNER - 1),
        "[1,20]": (1, N_INNER),
        "[0,0]": (0, 0),
        "[-20,-20]": (-N_INNER, -N_INNER),
        "[20,20]": (N_INNER, N_INNER),
    }
    for t in times:
        i0 = int(
            np.flatnonzero(
                stores["ts_ca"]["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").values == t
            )[0]
        )
        ts = {
            k: s["surface_temperature"]
            .isel(time=slice(i0 - half, i0 + half + 1))
            .values
            for k, s in stores.items()
        }
        j = np.arange(-half, half + 1)
        rec = {"changes": {}, "hold_window": {}, "shift": {}}
        for k, v in ts.items():
            d = np.abs(np.diff(v, axis=0))[:, on].max(axis=1)
            changed = j[1:][d > 0]
            rec["changes"][k] = (
                "every j" if len(changed) == len(d) else [int(x) for x in changed]
            )
        for k in ("ts_ca", "ts_si"):
            rec["hold_window"][k] = {
                name: _rms(
                    np.mean(
                        [
                            ts[k][half + jj]
                            - ts["ts_cm4"][half + a : half + b + 1].mean(0)
                            for jj in range(N_INNER)
                        ],
                        axis=0,
                    ),
                    w,
                )
                for name, (a, b) in cands.items()
            }
        for s in (-N_INNER, 0, N_INNER):
            rec["shift"][str(s)] = _rms(
                np.mean(
                    [
                        ts["ts_si"][half + jj] - ts["ts_ca"][half + jj + s]
                        for jj in range(N_INNER)
                    ],
                    axis=0,
                ),
                w,
            )
        s0 = sst5.sel(time=stores["ts_ca"]["time"].values[i0]).values
        s0 = s0 + 273.15 if np.nanmax(s0[on]) < 200 else s0
        rec["ic_sst_max_abs_ts_ca0_minus_sst5d_on_omega"] = float(
            np.abs(ts["ts_ca"][half] - s0)[on].max()
        )
        out["ics"][t] = rec
    _update_yaml(out_yaml, "sst_schedule", out)


FLUX_IN = [
    "DSWRFsfc",
    "USWRFsfc",
    "DLWRFsfc",
    "ULWRFsfc",
    "LHTFLsfc",
    "SHTFLsfc",
    "total_frozen_precipitation_rate",
]


def _flux(ds: xr.Dataset) -> np.ndarray:
    """F (k, sample, lat, lon): fme AtmosphereData.net_surface_energy_flux of the recorded outputs, into the surface."""
    import torch

    from fme.core.atmosphere_data import AtmosphereData

    data = {
        n: torch.as_tensor(ds["out__" + n].values) for n in FLUX_IN if "out__" + n in ds
    }
    return AtmosphereData(data).net_surface_energy_flux.numpy()


def _regions(t: xr.Dataset, om: xr.Dataset) -> dict[str, np.ndarray]:
    land = t["in__land_fraction"].isel(k=0, sample=0).values
    omega_ = om["omega"].values == 1
    return {
        "omega": omega_,
        "open_off_omega": (land == 0) & ~omega_,
        "partial_land": (land > 0) & (land < 1),
        "land": land == 1,
    }


def _diff_table(
    c: xr.Dataset, t: xr.Dataset, prefix: str, reg: dict, coslat: np.ndarray
) -> dict:
    out = {}
    for name in sorted(n for n in c.data_vars if n.startswith(prefix) and n in t):
        a, b = c[name].values, t[name].values
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)  # NaN in one arm only counts as different
        per_k = {}
        for kk, k in enumerate(c["k"].values):
            dk = d[kk]
            if not (dk > 0).any():
                continue
            per_k[int(k)] = {
                r: {
                    "max_abs": float(dk[:, m].max()) if m.any() else 0.0,
                    "area_fraction_nonzero": float(
                        ((dk > 0).any(0) * m * coslat).sum() / (m * coslat).sum()
                    ),
                }
                for r, m in reg.items()
            }
        out[name.removeprefix(prefix)] = {
            "first_k": min(per_k) if per_k else None,
            "per_k": per_k,
            "nan_count_C_T_k1": [int(np.isnan(a[0]).sum()), int(np.isnan(b[0]).sum())],
        }
    return out


def compare(
    c_nc: str, t_nc: str, omega_nc: str, out_yaml: str, swaps: list[str]
) -> None:
    c, t = xr.open_dataset(c_nc), xr.open_dataset(t_nc)
    om = xr.open_dataset(omega_nc)
    lat = om["lat"].values
    coslat = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], om["omega"].shape)
    reg = _regions(t, om)
    w = np.where(reg["omega"], om["areacello"].fillna(0).values, 0.0)
    k = c["k"].values
    inputs = _diff_table(c, t, "in__", reg, coslat)
    nxt = _diff_table(c, t, "next__", reg, coslat)
    outs = _diff_table(c, t, "out__", reg, coslat)
    differing = {n: v["first_k"] for n, v in inputs.items() if v["first_k"] is not None}
    k0 = min(differing.values()) if differing else None
    f_c, f_t = _flux(c), _flux(t)
    norm = lambda f: [_rms(np.nan_to_num(f[i].mean(0)), w) for i in range(len(k))]  # noqa: E731  ‖⟨·⟩‖ on Ω
    sst_change = {
        arm: [
            int(kk)
            for kk, d in zip(
                k[1:],
                np.abs(np.diff(ds["in__surface_temperature"].values, axis=0))
                .max(1)[:, reg["omega"]]
                .max(1),
            )
            if d > 0
        ]
        for arm, ds in (("C", c), ("T", t))
    }
    doc = {
        "definition": "k = inner atmosphere step (k = 1 first); in__ = Stepper.step args.input (state at k-1 plus "
        "forcing), next__ = args.next_step_input_data, out__ = output of step k. max_abs, "
        "area_fraction_nonzero (cos-lat weighted) of |C - T| per region; NaN in one arm only = different.",
        "files": {
            "C": c_nc,
            "T": t_nc,
            "C_attrs": dict(c.attrs),
            "T_attrs": dict(t.attrs),
        },
        "first_differing_input": {
            "k": k0,
            "variables": sorted(n for n, v in differing.items() if v == k0),
        },
        "input_first_k": differing,
        "felt_sst_change_k_on_omega": sst_change,
        "inputs": inputs,
        "next_step_inputs": nxt,
        "outputs_first_k": {n: v["first_k"] for n, v in outs.items()},
        "flux": {
            "k": [int(x) for x in k],
            "norm_F_C_minus_F_T_on_omega": norm(f_c - f_t),
            "norm_F_T_on_omega": norm(f_t),
        },
        "swaps": {},
    }
    for item in swaps:
        label, path = item.split("=", 1)
        s = xr.open_dataset(path)
        f_s = _flux(s)
        doc["swaps"][label] = {
            "file": path,
            "attrs": dict(s.attrs),
            "norm_F_Tswap_minus_F_C_on_omega": norm(f_s - f_c),
            "ratio_to_norm_F_T_minus_F_C": [
                a / b if b > 0 else None
                for a, b in zip(norm(f_s - f_c), norm(f_t - f_c))
            ],
            "max_abs_out_Tswap_minus_C_global": [
                float(
                    max(
                        np.nanmax(np.abs(s[n].values[i] - c[n].values[i]))
                        for n in s.data_vars
                        if n.startswith("out__") and n in c
                    )
                )
                for i in range(len(k))
            ],
        }
    _update_yaml(out_yaml, "inputs_C_vs_T", doc)


def launched(
    runs_yaml: str, omega_nc: str, out_yaml: str, n_steps: int = 2 * N_INNER
) -> None:
    """‖⟨F_C − F_T⟩‖ on Ω per inner step k from the stage-1 per-step stores (A/data/09_checks.yaml stage1_runs)."""
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    runs = yaml.safe_load(open(runs_yaml))["stage1_runs"]["arms"]
    om = xr.open_dataset(omega_nc)
    w = np.where(om["omega"].values == 1, om["areacello"].fillna(0).values, 0.0)
    path = {
        "T": runs["T"]["stores"]["."]["path"],
        "C": runs["C"]["stores"]["atmosphere"]["path"],
    }
    f = {
        a: -xr.open_zarr(fs.get_mapper(p.removeprefix("gs://")))[
            "net_energy_flux_sfc_into_atmosphere"
        ]
        .isel(time=slice(0, n_steps))
        .transpose("sample", "time", "lat", "lon")
        .values
        for a, p in path.items()
    }
    d = (f["C"] - f["T"]).mean(0)
    _update_yaml(
        out_yaml,
        "launched_runs",
        {
            "definition": launched.__doc__.strip(),
            "stores": path,
            "k": list(range(1, n_steps + 1)),
            "norm_F_C_minus_F_T_on_omega": [
                _rms(np.nan_to_num(d[i]), w) for i in range(n_steps)
            ],
            "norm_F_T_on_omega": [
                _rms(np.nan_to_num(f["T"].mean(0)[i]), w) for i in range(n_steps)
            ],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("omega")
    o.add_argument("out")
    s = sub.add_parser("schedule")
    s.add_argument("config_c")
    s.add_argument("omega_nc")
    s.add_argument("out_yaml")
    cp = sub.add_parser("compare")
    for n in ("c_nc", "t_nc", "omega_nc", "out_yaml"):
        cp.add_argument(n)
    cp.add_argument("swaps", nargs="*")
    la = sub.add_parser("launched")
    for n in ("runs_yaml", "omega_nc", "out_yaml"):
        la.add_argument(n)
    r = sub.add_parser("run")
    r.add_argument("arm", choices=["C", "T"])
    r.add_argument("config")
    r.add_argument("ckpt")
    r.add_argument("out")
    r.add_argument("--swap-from")
    r.add_argument("--swap-forcing", default="")
    r.add_argument("--swap-ic", default="")
    r.add_argument("--n-windows", type=int, default=2)
    a = p.parse_args()
    if a.cmd == "omega":
        omega(a.out)
    elif a.cmd == "compare":
        compare(a.c_nc, a.t_nc, a.omega_nc, a.out_yaml, a.swaps)
    elif a.cmd == "launched":
        launched(a.runs_yaml, a.omega_nc, a.out_yaml)
    elif a.cmd == "schedule":
        schedule(a.config_c, a.omega_nc, a.out_yaml)
    elif a.cmd == "run":
        run(
            a.arm,
            a.config,
            a.ckpt,
            a.out,
            a.swap_from,
            a.swap_forcing,
            a.swap_ic,
            a.n_windows,
        )
