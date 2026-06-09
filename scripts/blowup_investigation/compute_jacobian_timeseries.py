"""Compute the 3x3 global-mean Jacobian of T0, q0, u0 at each of the first N steps."""

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

N_STEPS = 2922
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
OUT = "./output/jacobian_timeseries_full"


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
            f"n_forward_steps={N_STEPS}",
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

    # Preload predictions into GPU memory
    print(f"Loading predictions for steps 1-{N_STEPS}...")
    preds = xr.open_dataset(PREDS_PATH)
    ic_bd = data.initial_condition.as_batch_data()
    device = next(iter(ic_bd.data.values())).device

    pred_tensors = {}
    for var in prog_names:
        pred_tensors[var] = (
            torch.from_numpy(preds[var].isel(time=slice(0, N_STEPS)).values)
            .float()
            .to(device)
        )  # (1, N_STEPS, 45, 90)
    preds.close()

    nlat = pred_tensors[VARS[0]].shape[-2]
    w = area_weights(nlat, device)

    # Storage for 3x3 Jacobians at each step
    jac = np.zeros((N_STEPS, len(VARS), len(VARS)))

    t0 = time.time()
    print(f"Computing Jacobians for {N_STEPS} steps...")
    loader = iter(data.loader)

    for step in range(N_STEPS):
        forcing = next(loader)

        # State: IC for step 1 (step=0), predictions[time=step-1] for later
        if step == 0:
            state_data = dict(ic_bd.data)
        else:
            state_data = {
                var: pred_tensors[var][:, step - 1 : step] for var in prog_names
            }
        state = make_state(ic_bd, state_data)

        # Base forward pass
        with torch.no_grad():
            base_out, _ = predict(
                state, forcing=forcing, compute_derived_variables=False
            )

        # Finite differences for each input variable
        for j, in_v in enumerate(VARS):
            eps = EPS[in_v]
            pd = {k: v.clone() for k, v in state_data.items()}
            pd[in_v] = pd[in_v] + eps
            ps = make_state(ic_bd, pd)
            with torch.no_grad():
                po, _ = predict(ps, forcing=forcing, compute_derived_variables=False)
            for i, out_v in enumerate(VARS):
                diff = po.prediction[out_v] - base_out.prediction[out_v]
                jac[step, i, j] = gm(diff[:, -1:], w).item() / eps

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (N_STEPS - step - 1) / rate
            print(f"  step {step + 1}/{N_STEPS}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Save raw data
    np.save(os.path.join(OUT, "jacobian_timeseries.npy"), jac)

    # Plot
    days = np.arange(1, N_STEPS + 1)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for i, out_v in enumerate(VARS):
        for j, in_v in enumerate(VARS):
            ax = axes[i, j]
            ax.plot(days, jac[:, i, j], lw=0.8)
            ax.set_title(f"d(gm_{SHORT[out_v]}) / d(gm_{SHORT[in_v]})", fontsize=10)
            ax.axhline(0, color="gray", ls=":", lw=0.5)
            if i == j:
                ax.axhline(1, color="red", ls="--", lw=0.5, alpha=0.5)
                ax.axhline(-1, color="red", ls="--", lw=0.5, alpha=0.5)
            if i == 2:
                ax.set_xlabel("step (days)")
    fig.suptitle(f"Global-mean scalar Jacobians over first {N_STEPS} days", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jacobian_timeseries.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {OUT}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    with Distributed.context():
        main()
