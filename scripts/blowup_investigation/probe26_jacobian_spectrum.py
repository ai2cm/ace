"""Probes 2+6: seed-controlled full global-mean Jacobian spectrum.

Extends the 3x3 global-mean Jacobian to all prognostic variables, with the RNG
seeded identically before the base and each perturbed forward pass so the
stochastic model's noise is common across the finite difference (the prior
3x3 timeseries drew fresh noise per pass, polluting the off-diagonals).

For each of three saved-rollout snapshots (early / mid / near-onset) and each
of several noise seeds, computes J[i, j] = d(gm out_i) / d(gm in_j) via a
uniform additive perturbation eps_j = EPS_FRAC * std_j to input j. Reports the
Jacobian in per-sigma units (Jn = J * std_j / std_i), its diagonal, its
eigenvalue spectrum, and the spread across seeds.

Outputs to output/probe26_jacobian_spectrum/.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import dacite
import matplotlib.pyplot as plt
import numpy as np
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
OUT = os.path.join(HERE, "output", "probe26_jacobian_spectrum")

# Snapshot steps: early (on-manifold), mid (drift), near-onset (step 2113 is
# the q0 OOS onset for this rollout).
TARGET_STEPS = [200, 1500, 2050]
SEEDS = [101, 202, 303]
EPS_FRAC = 1e-3  # eps_j = EPS_FRAC * full-field training std of variable j


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


def seeded_predict(predict_fn, state, forcing, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out, _ = predict_fn(state, forcing=forcing, compute_derived_variables=False)
    return out


def main():
    max_step = max(TARGET_STEPS)
    config_data = prepare_config(
        os.path.join(HERE, "inference.yaml"),
        override=[
            f"n_forward_steps={max_step + 1}",
            "forward_steps_in_memory=1",
            "experiment_dir=./output/probe26_tmp",
        ],
    )
    config = dacite.from_dict(
        data_class=InferenceConfig, data=config_data, config=dacite.Config(strict=True)
    )
    os.makedirs(config.experiment_dir, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

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
    prog_names = sorted(stepper_config.prognostic_names)
    n = len(prog_names)
    print(f"{n} prognostic variables")

    means, stds = load_norm_stats(CKPT)
    std = np.array([float(stds[v]) for v in prog_names])
    eps = EPS_FRAC * std

    ic_bd = data.initial_condition.as_batch_data()
    device = next(iter(ic_bd.data.values())).device

    preds = xr.open_dataset(PREDS_PATH)
    pred_tensors = {}
    for var in prog_names:
        pred_tensors[var] = (
            torch.from_numpy(preds[var].isel(time=slice(0, max_step + 1)).values)
            .float()
            .to(device)
        )
    preds.close()

    nlat = pred_tensors[prog_names[0]].shape[-2]
    w = area_weights(nlat, device)

    # jac[state, seed, out_i, in_j] in raw units
    jac = np.zeros((len(TARGET_STEPS), len(SEEDS), n, n))

    t0 = time.time()
    loader = iter(data.loader)
    n_forwards = 0
    for step in range(max_step + 1):
        forcing = next(loader)
        if step not in TARGET_STEPS:
            continue
        si = TARGET_STEPS.index(step)
        print(f"step {step} ({time.time() - t0:.0f}s)")

        if step == 0:
            state_data = dict(ic_bd.data)
        else:
            state_data = {
                var: pred_tensors[var][:, step - 1 : step] for var in prog_names
            }

        for ki, seed in enumerate(SEEDS):
            base_out = seeded_predict(
                predict, make_state(ic_bd, state_data), forcing, seed
            )
            n_forwards += 1
            base_gm = {
                v: gm(base_out.prediction[v][:, -1:], w).item() for v in prog_names
            }
            for j, in_v in enumerate(prog_names):
                pd = {k: v.clone() for k, v in state_data.items()}
                pd[in_v] = pd[in_v] + eps[j]
                po = seeded_predict(predict, make_state(ic_bd, pd), forcing, seed)
                n_forwards += 1
                for i, out_v in enumerate(prog_names):
                    d = gm(po.prediction[out_v][:, -1:], w).item() - base_gm[out_v]
                    jac[si, ki, i, j] = d / eps[j]
            print(
                f"  seed {seed} done ({time.time() - t0:.0f}s, {n_forwards} forwards)"
            )

    np.save(os.path.join(OUT, "jacobian.npy"), jac)
    with open(os.path.join(OUT, "variables.txt"), "w") as f:
        f.write("\n".join(prog_names))

    # Per-sigma normalized Jacobian: response of gm_i (in std_i units) to a
    # std_j-sized displacement of gm_j.
    jn = jac * std[None, None, None, :] / std[None, None, :, None]
    jn_mean = jn.mean(axis=1)  # over seeds
    jn_spread = jn.std(axis=1)

    lines = ["# Probes 2+6: seed-controlled global-mean Jacobian spectrum", ""]
    lines.append(f"Variables ({n}): see variables.txt. eps = {EPS_FRAC} * std.")
    lines.append(f"Seeds: {SEEDS}. Values below are seed-means (per-sigma units).")
    for si, step in enumerate(TARGET_STEPS):
        m = jn_mean[si]
        sp = jn_spread[si]
        eig = np.linalg.eigvals(m)
        order = np.argsort(-np.abs(eig))
        lines.append("")
        lines.append(f"## Snapshot step {step}")
        lines.append("")
        lines.append(
            f"Top |eigenvalues|: "
            + ", ".join(
                f"{np.abs(eig[k]):.3f}" + ("" if np.isreal(eig[k]) else "(c)")
                for k in order[:8]
            )
        )
        diag = np.diag(m)
        dorder = np.argsort(-np.abs(diag))
        lines.append("")
        lines.append("Largest |diagonal| entries (var: J_ii, seed spread):")
        for k in dorder[:12]:
            lines.append(
                f"- {prog_names[k]}: {diag[k]:+.3f} (+/- {np.diag(sp)[k]:.3f})"
            )
        off = m - np.diag(diag)
        oorder = np.argsort(-np.abs(off), axis=None)
        lines.append("")
        lines.append("Largest |off-diagonal| entries (out <- in: Jn, seed spread):")
        for flat in oorder[:12]:
            i, j = np.unravel_index(flat, off.shape)
            lines.append(
                f"- {prog_names[i]} <- {prog_names[j]}: {off[i, j]:+.3f} "
                f"(+/- {sp[i, j]:.3f})"
            )
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)

    # Heatmaps
    fig, axes = plt.subplots(1, len(TARGET_STEPS), figsize=(7 * len(TARGET_STEPS), 6))
    for si, step in enumerate(TARGET_STEPS):
        ax = axes[si]
        im = ax.imshow(jn_mean[si], cmap="RdBu_r", vmin=-1.2, vmax=1.2)
        ax.set_title(f"step {step}")
        ax.set_xlabel("input var index")
        ax.set_ylabel("output var index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Per-sigma global-mean Jacobian (seed mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jacobian_heatmaps.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n)
    for si, step in enumerate(TARGET_STEPS):
        ax.errorbar(
            x,
            np.diag(jn_mean[si]),
            yerr=np.diag(jn_spread[si]),
            marker="o",
            ms=3,
            lw=0.8,
            label=f"step {step}",
        )
    ax.axhline(1.0, color="red", ls="--", lw=0.8)
    ax.axhline(0.0, color="gray", ls=":", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(prog_names, rotation=90, fontsize=7)
    ax.set_ylabel("J_ii (per-sigma)")
    ax.legend()
    ax.set_title("Global-mean Jacobian diagonal: values near 1 are neutral modes")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jacobian_diagonal.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {OUT}")
    print(f"Total: {n_forwards} forwards in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
