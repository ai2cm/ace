"""Compute Jacobians of T0, q0, u0 at a specific step of the baseline rollout.

Loads the prognostic state directly from saved predictions (deterministic) and
iterates the data loader (no model calls) to get the correct forcing window.
"""

import os
import time

import matplotlib
import numpy as np
import torch
import xarray as xr

matplotlib.use("Agg")
import dacite
import matplotlib.pyplot as plt

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.inference.inference import InferenceConfig, get_initial_condition
from fme.core.cli import prepare_config
from fme.core.distributed.distributed import Distributed

TARGET_STEP = 2250
PREDS_PATH = "./output/run_best_ckpt/autoregressive_predictions.nc"
VARS = ["air_temperature_0", "specific_total_water_0", "eastward_wind_0"]
SHORT = {
    "air_temperature_0": "T0",
    "specific_total_water_0": "q0",
    "eastward_wind_0": "u0",
}
EPS = {
    "air_temperature_0": 0.01,
    "specific_total_water_0": 1e-8,
    "eastward_wind_0": 0.01,
}
OUT = "./output/jacobians_step2250"


def area_weights(nlat, device):
    lat = np.linspace(-90, 90, nlat)
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device).reshape(1, 1, -1, 1)


def gm(field, w):
    return (field.double() * w).mean(dim=(-2, -1)).float()


def make_state(template_bd, new_data):
    return PrognosticState(
        BatchData(
            data=new_data,
            time=template_bd.time,
            labels=template_bd.labels,
            horizontal_dims=template_bd.horizontal_dims,
            epoch=template_bd.epoch,
            n_ensemble=template_bd.n_ensemble,
            data_mask=template_bd.data_mask,
        )
    )


def main():
    config_data = prepare_config(
        "inference.yaml",
        override=[
            f"n_forward_steps={TARGET_STEP}",
            "forward_steps_in_memory=1",
            "experiment_dir=./output/jacobian_tmp",
        ],
    )
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    os.makedirs(config.experiment_dir, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

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

    predict = stepper.predict_paired
    prog_names = stepper_config.prognostic_names
    print(f"Prognostic vars ({len(prog_names)}): {sorted(prog_names)}")

    t0 = time.time()

    # --- Iterate data loader (no model calls) to get forcing at step TARGET ---
    print(f"Advancing data loader to step {TARGET_STEP} (data only, no model)...")
    loader = iter(data.loader)
    for i in range(TARGET_STEP - 1):
        next(loader)
    forcing = next(loader)
    print(f"  Forcing ready in {time.time() - t0:.1f}s")

    # --- Load prognostic state from saved predictions ---
    # predictions[time=k] = output of step k+1; input to step N = predictions[time=N-2]
    time_idx = TARGET_STEP - 2
    print(f"Loading state from {PREDS_PATH} at time index {time_idx}...")
    preds = xr.open_dataset(PREDS_PATH)
    ic_bd = data.initial_condition.as_batch_data()
    device = next(iter(ic_bd.data.values())).device

    state_data: dict[str, torch.Tensor] = {}
    for var in prog_names:
        val = preds[var].isel(time=time_idx).values  # (sample=1, lat, lon)
        state_data[var] = torch.from_numpy(val).unsqueeze(1).float().to(device)
    preds.close()

    state = make_state(ic_bd, state_data)
    print(f"  State loaded. Shape: {list(state_data.values())[0].shape}")

    nlat = next(iter(state_data.values())).shape[-2]
    nlon = next(iter(state_data.values())).shape[-1]
    w = area_weights(nlat, device)

    # ---- Baseline forward pass ----
    with torch.no_grad():
        base_out, _ = predict(state, forcing=forcing, compute_derived_variables=False)
    base_p = base_out.prediction

    print("\nBaseline global means at step output:")
    for v in VARS:
        print(f"  {SHORT[v]}: {gm(base_p[v][:, -1:], w).item():.6f}")

    # ---- Finite differences: d(output_2D)/d(gm_input) ----
    fd_maps: dict[str, dict[str, np.ndarray]] = {v: {} for v in VARS}
    scalar_jac: dict[str, dict[str, float]] = {v: {} for v in VARS}

    for in_v in VARS:
        eps = EPS[in_v]
        pd = {k: v.clone() for k, v in state_data.items()}
        pd[in_v] = pd[in_v] + eps
        ps = make_state(ic_bd, pd)
        with torch.no_grad():
            po, _ = predict(ps, forcing=forcing, compute_derived_variables=False)
        for out_v in VARS:
            diff = (po.prediction[out_v] - base_p[out_v]) / eps
            fd_maps[out_v][in_v] = diff[0, -1].cpu().numpy()
            scalar_jac[out_v][in_v] = gm(diff[:, -1:], w).item()

    # ---- Autograd: d(gm_output)/d(input_2D) ----
    grad_maps: dict[str, dict[str, np.ndarray]] = {v: {} for v in VARS}
    for out_v in VARS:
        gd = {}
        gt = {}
        for k, tensor in state_data.items():
            if k in VARS:
                t = tensor.detach().clone().requires_grad_(True)
                gd[k] = t
                gt[k] = t
            else:
                gd[k] = tensor.detach()
        gs = make_state(ic_bd, gd)
        try:
            out, _ = predict(gs, forcing=forcing, compute_derived_variables=False)
            gm_out = gm(out.prediction[out_v][:, -1:], w)
            gm_out.backward()
            for in_v in VARS:
                g = gt[in_v].grad
                grad_maps[out_v][in_v] = (
                    g[0, -1].cpu().numpy() if g is not None else np.zeros((nlat, nlon))
                )
        except Exception as e:
            print(f"Autograd failed for {SHORT[out_v]}: {e}")
            for in_v in VARS:
                grad_maps[out_v][in_v] = np.zeros((nlat, nlon))

    # ---- Print scalar Jacobians ----
    print(f"\n=== Scalar Jacobians at step {TARGET_STEP}: d(gm_out) / d(gm_in) ===")
    header = f"{'':>15s}"
    for in_v in VARS:
        header += f"  d(gm_{SHORT[in_v]:>3s})_in"
    print(header)
    for out_v in VARS:
        row = f"d(gm_{SHORT[out_v]:>3s})_out "
        for in_v in VARS:
            row += f"  {scalar_jac[out_v][in_v]:12.4e}"
        print(row)

    # ---- Plot: d(2D_output)/d(gm_input) ----
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, out_v in enumerate(VARS):
        for j, in_v in enumerate(VARS):
            ax = axes[i, j]
            d = fd_maps[out_v][in_v]
            vm = max(abs(d.max()), abs(d.min()), 1e-10)
            im = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"d({SHORT[out_v]}[x]) / d(gm_{SHORT[in_v]})", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
            if j == 0:
                ax.set_ylabel("lat")
            if i == 2:
                ax.set_xlabel("lon")
    fig.suptitle(
        f"Jacobian of 2D output field w.r.t. global-mean input (step {TARGET_STEP})"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jac_2d_out_vs_gm_in.png"), dpi=150)
    plt.close(fig)

    # ---- Plot: d(gm_output)/d(2D_input) ----
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, out_v in enumerate(VARS):
        for j, in_v in enumerate(VARS):
            ax = axes[i, j]
            d = grad_maps[out_v][in_v]
            vm = max(abs(d.max()), abs(d.min()), 1e-10)
            im = ax.pcolormesh(lon, lat, d, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ax.set_title(f"d(gm_{SHORT[out_v]}) / d({SHORT[in_v]}[x])", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
            if j == 0:
                ax.set_ylabel("lat")
            if i == 2:
                ax.set_xlabel("lon")
    fig.suptitle(
        f"Jacobian of global-mean output w.r.t. 2D input field (step {TARGET_STEP})"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jac_gm_out_vs_2d_in.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {OUT}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with Distributed.context():
        main()
