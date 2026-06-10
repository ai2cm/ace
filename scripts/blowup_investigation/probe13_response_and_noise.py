"""Probes 1+3: finite-amplitude response curves and noise-injection rate.

From a saved-rollout snapshot, runs one batched ensemble rollout whose members
are 8 unperturbed replicates (probe 3: per-step variance injection into each
global-mean mode from the stochastic stepping) plus a ladder of uniform
additive global-mean perturbations to one variable, 3 replicates each
(probe 1: relaxation/amplification of finite displacements, the restoring
timescale tau, and the basin boundary).

Amplitudes are in units of the variable's full-field training std. Usage:

  python probe13_response_and_noise.py --step 1500 --var specific_total_water_0

Outputs to output/probe13_<var>_step<step>/.
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import dacite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.inference.inference import InferenceConfig, get_initial_condition
from fme.core.cli import prepare_config

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS_PATH = os.path.join(
    HERE, "output", "run_best_ckpt", "autoregressive_predictions.nc"
)
CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")

N_UNPERT = 8
AMPLITUDES = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]  # x full-field std
N_REPL = 3
NOISE_FIT_DAYS = 30  # fit variance growth over this initial window


def load_norm_stats(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = ckpt["stepper"]["config"]["step"]["config"]["normalization"]["network"]
    return dict(norm["means"]), dict(norm["stds"])


def area_weights(nlat, device):
    lat = np.linspace(-90, 90, nlat)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device).reshape(1, 1, -1, 1)


def gm(field, w):
    return (field.double() * w).mean(dim=(-2, -1)).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1500)
    parser.add_argument("--var", default="specific_total_water_0")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    n_members = N_UNPERT + len(AMPLITUDES) * N_REPL
    out_dir = os.path.join(HERE, "output", f"probe13_{args.var}_step{args.step}")
    os.makedirs(out_dir, exist_ok=True)

    # Snapshot state and its valid time from the saved rollout
    preds = xr.open_dataset(PREDS_PATH)
    snap_idx = args.step - 1  # preds index k = state after step k+1
    vt = np.asarray(preds["valid_time"].values).squeeze()
    snap_time = pd.Timestamp(vt[snap_idx])
    print(f"Snapshot: saved index {snap_idx}, valid_time {snap_time}")

    config_data = prepare_config(os.path.join(HERE, "inference.yaml"))
    config_data["n_forward_steps"] = args.days
    config_data["forward_steps_in_memory"] = 1
    config_data["experiment_dir"] = "./output/probe13_tmp"
    config_data["n_ensemble_per_ic"] = n_members
    config_data["initial_condition"]["start_indices"]["times"] = [
        snap_time.strftime("%Y-%m-%dT%H:%M:%S")
    ]
    config = dacite.from_dict(
        data_class=InferenceConfig, data=config_data, config=dacite.Config(strict=True)
    )
    os.makedirs(config.experiment_dir, exist_ok=True)

    stepper_config = config.load_stepper_config()
    req = stepper_config.get_forcing_window_data_requirements(
        n_forward_steps=config.forward_steps_in_memory
    )
    ic = get_initial_condition(
        config.initial_condition.get_dataset(),
        stepper_config.prognostic_names,
        labels=config.labels,
        n_ensemble=config.n_ensemble_per_ic,
    )
    stepper = config.load_stepper()
    stepper.set_eval()
    data = get_forcing_data(
        config=config.forcing_loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=req,
        initial_condition=ic,
        surface_temperature_name=stepper.surface_temperature_name,
        ocean_fraction_name=stepper.ocean_fraction_name,
        label_override=config.labels,
    )

    prog_names = sorted(stepper_config.prognostic_names)
    means, stds = load_norm_stats(CKPT)
    std_v = float(stds[args.var])

    ic_bd = data.initial_condition.as_batch_data()
    device = next(iter(ic_bd.data.values())).device
    nlat = ic_bd.data[prog_names[0]].shape[-2]
    w = area_weights(nlat, device)

    # Replace IC data with the saved snapshot, broadcast to all members,
    # then add per-member offsets to the perturbed variable.
    offsets = np.concatenate([np.zeros(N_UNPERT), np.repeat(AMPLITUDES, N_REPL)])
    offsets_t = torch.tensor(offsets * std_v, dtype=torch.float32, device=device)
    state_data = {}
    for v in prog_names:
        snap = (
            torch.from_numpy(preds[v].isel(time=slice(snap_idx, snap_idx + 1)).values)
            .float()
            .to(device)
        )  # (1, 1, lat, lon)
        field = snap.expand(n_members, -1, -1, -1).clone()
        if v == args.var:
            field = field + offsets_t.view(-1, 1, 1, 1)
        state_data[v] = field
    preds.close()

    state = PrognosticState(
        BatchData(
            data=state_data,
            time=ic_bd.time,
            labels=ic_bd.labels,
            horizontal_dims=ic_bd.horizontal_dims,
            epoch=ic_bd.epoch,
            n_ensemble=ic_bd.n_ensemble,
            data_mask=ic_bd.data_mask,
        )
    )

    # Rollout, accumulating global means of every prognostic variable
    gms = np.zeros((args.days, n_members, len(prog_names)))
    t0 = time.time()
    loader = iter(data.loader)
    for step in range(args.days):
        forcing = next(loader)
        with torch.no_grad():
            _, state = stepper.predict_paired(
                state, forcing=forcing, compute_derived_variables=False
            )
        sd = state.as_batch_data().data
        for vi, v in enumerate(prog_names):
            gms[step, :, vi] = gm(sd[v][:, -1:], w).squeeze().cpu().numpy()
        if (step + 1) % 10 == 0:
            print(f"  step {step + 1}/{args.days} ({time.time() - t0:.0f}s)")

    np.savez(
        os.path.join(out_dir, "global_means.npz"),
        gms=gms,
        offsets=offsets,
        variables=np.array(prog_names),
        snapshot_step=args.step,
    )

    days = np.arange(1, args.days + 1)
    vi = prog_names.index(args.var)
    unpert = gms[:, :N_UNPERT, :]

    # --- Probe 3: noise-injection rate per global-mean mode ---
    lines = [
        f"# Probes 1+3 at snapshot step {args.step} ({snap_time:%Y-%m-%d}), "
        f"perturbed var {args.var}",
        "",
        f"{n_members} members: {N_UNPERT} unperturbed, amplitudes "
        f"{AMPLITUDES} x std ({std_v:.3e}) x {N_REPL} replicates.",
        "",
        "## Probe 3: per-step noise injection (unperturbed members)",
        "",
        "sigma_step = sqrt(slope of across-member variance over first "
        f"{NOISE_FIT_DAYS} days), in full-field-std units per sqrt(day).",
        "",
        "| variable | sigma_step (std/sqrt(day)) | stationary std for tau=100d |",
        "|---|---|---|",
    ]
    for vj, v in enumerate(prog_names):
        var_t = unpert[:, :, vj].var(axis=1, ddof=1)
        slope = np.polyfit(days[:NOISE_FIT_DAYS], var_t[:NOISE_FIT_DAYS], 1)[0]
        sigma_step = np.sqrt(max(slope, 0.0)) / float(stds[v])
        stationary = sigma_step * np.sqrt(100 / 2)
        lines.append(f"| {v} | {sigma_step:.2e} | {stationary:.2e} |")

    # --- Probe 1: normalized response curves ---
    lines += [
        "",
        "## Probe 1: response to finite global-mean displacements of " f"{args.var}",
        "",
        "response(t) = (gm_pert - gm_unpert_mean) / delta; 1.0 = fully "
        "retained (neutral), exp(-t/tau) = relaxing, >1 = amplifying.",
        "",
        "| amplitude (std) | retained @30d | retained @60d | retained @90d |",
        "|---|---|---|---|",
    ]
    base = unpert[:, :, vi].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ai, amp in enumerate(AMPLITUDES):
        m0 = N_UNPERT + ai * N_REPL
        resp = (gms[:, m0 : m0 + N_REPL, vi].mean(axis=1) - base) / (amp * std_v)
        ax.plot(days, resp, lw=1.0, label=f"{amp:+.2g} std")
        r = [
            resp[min(d, args.days) - 1] if args.days >= d else np.nan
            for d in (30, 60, 90)
        ]
        lines.append(f"| {amp:+.2g} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axhline(0.0, color="gray", ls="-", lw=0.5)
    ax.set_xlabel("days since perturbation")
    ax.set_ylabel(f"retained fraction of gm {args.var} displacement")
    ax.set_title(f"Probe 1: response curves at step {args.step}")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "response_curves.png"), dpi=150)
    plt.close(fig)

    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nSaved to {out_dir} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
