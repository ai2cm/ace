"""Probe 4: spatial structure of the neutral moisture mode.

Two parts:

1. Response to *patterned* displacements: apply gm-equivalent displacements
   of specific_total_water_0 that are uniform, tropics-only (|lat| < 30), or
   extratropics-only, each scaled to the same area-weighted global mean, and
   compare 90-day retention of the global mean. If retention depends on the
   pattern, a uniform-offset training perturbation underconstrains the mode.

2. Where the bias lives: the time-mean (model - ERA5) q0 map over the drift
   window from the saved rollout, showing the spatial pattern of the secular
   dry bias measured by probe 7.

Outputs to output/probe4_spatial/.
"""

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
ERA5_PATH = os.path.join(HERE, "data", "era5_4deg_blowup_slice.nc")
CKPT = os.path.join(HERE, "checkpoint", "best_inference_ckpt.tar")
OUT = os.path.join(HERE, "output", "probe4_spatial")

VAR = "specific_total_water_0"
SNAPSHOT_STEP = 1500
N_DAYS = 90
N_UNPERT = 8
N_REPL = 3
AMPLITUDES = [-1.0, 1.0]  # x full-field std, as the area-weighted global mean
PATTERNS = ["uniform", "tropics", "extratropics"]
DRIFT_WINDOW = (1500, 2100)  # steps for the bias map


def load_norm_stats(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm = ckpt["stepper"]["config"]["step"]["config"]["normalization"]["network"]
    return dict(norm["means"]), dict(norm["stds"])


def pattern_field(name, lat, lon_n, delta):
    """(lat, lon) field with area-weighted global mean delta."""
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    if name == "uniform":
        mask = np.ones_like(lat)
    elif name == "tropics":
        mask = (np.abs(lat) < 30).astype(float)
    elif name == "extratropics":
        mask = (np.abs(lat) >= 30).astype(float)
    else:
        raise ValueError(name)
    gm_mask = (mask * w).mean()
    field = np.repeat((mask * delta / gm_mask)[:, None], lon_n, axis=1)
    return field


def main():
    os.makedirs(OUT, exist_ok=True)
    means, stds = load_norm_stats(CKPT)
    std_v = float(stds[VAR])

    preds = xr.open_dataset(PREDS_PATH)
    snap_idx = SNAPSHOT_STEP - 1
    vt_all = pd.DatetimeIndex(np.asarray(preds["valid_time"].values).squeeze())
    snap_time = vt_all[snap_idx]
    lat = preds["lat"].values
    lon_n = preds.sizes["lon"]
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()

    # --- Part 2 first (cheap): drift-phase bias map vs ERA5 ---
    era5 = xr.open_dataset(ERA5_PATH)
    era5_idx = pd.DatetimeIndex(era5["time"].values).get_indexer(
        vt_all[slice(*DRIFT_WINDOW)]
    )
    bias_map = (
        preds[VAR].isel(time=slice(*DRIFT_WINDOW)).values.squeeze().mean(axis=0)
        - era5[VAR].isel(time=era5_idx).values.mean(axis=0)
    ) / std_v
    era5.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(
        preds["lon"].values, lat, bias_map, cmap="RdBu_r", vmin=-3, vmax=3
    )
    fig.colorbar(im, ax=ax, label="(model - ERA5) / std")
    ax.set_title(
        f"{VAR} bias map, drift phase (steps {DRIFT_WINDOW[0]}-{DRIFT_WINDOW[1]})"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bias_map.png"), dpi=150)
    plt.close(fig)

    zonal = (bias_map * w[:, None]).mean(axis=1)
    trop = np.abs(lat) < 30
    trop_mean = (bias_map[trop] * w[trop, None]).mean() / w[trop].mean()
    extrop_mean = (bias_map[~trop] * w[~trop, None]).mean() / w[~trop].mean()
    summary_part2 = [
        "## Part 2: spatial structure of the drift-phase bias (vs ERA5)",
        "",
        f"global mean: {(bias_map * w[:, None]).mean():+.3f} sigma",
        f"tropics (|lat|<30) mean: {trop_mean:+.3f} sigma",
        f"extratropics mean: {extrop_mean:+.3f} sigma",
        f"zonal-mean range: {zonal.min():+.3f} (lat {lat[zonal.argmin()]:.0f}) to "
        f"{zonal.max():+.3f} (lat {lat[zonal.argmax()]:.0f})",
        "",
    ]

    # --- Part 1: patterned displacement response ---
    config_data = prepare_config(os.path.join(HERE, "inference.yaml"))
    config_data["n_forward_steps"] = N_DAYS
    config_data["forward_steps_in_memory"] = 1
    config_data["experiment_dir"] = "./output/probe4_tmp"
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
    # Single-sample IC; see probe13 for why (forcing double-broadcast bug).
    ic = get_initial_condition(
        config.initial_condition.get_dataset(),
        stepper_config.prognostic_names,
        labels=config.labels,
        n_ensemble=1,
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
    n_members = N_UNPERT + len(PATTERNS) * len(AMPLITUDES) * N_REPL
    ic_bd = data.initial_condition.as_batch_data().broadcast_ensemble(n_members)
    device = next(iter(ic_bd.data.values())).device
    w_t = torch.tensor(w, dtype=torch.float32, device=device).reshape(1, 1, -1, 1)

    # member -> (pattern, amplitude); first N_UNPERT are unperturbed
    member_specs: list[tuple[str, float] | None] = [None] * N_UNPERT
    for p in PATTERNS:
        for a in AMPLITUDES:
            member_specs += [(p, a)] * N_REPL

    state_data = {}
    for v in prog_names:
        snap = (
            torch.from_numpy(preds[v].isel(time=slice(snap_idx, snap_idx + 1)).values)
            .float()
            .to(device)
        )
        field = snap.expand(n_members, -1, -1, -1).clone()
        if v == VAR:
            for mi, spec in enumerate(member_specs):
                if spec is None:
                    continue
                p, a = spec
                pf = torch.from_numpy(
                    pattern_field(p, lat, lon_n, a * std_v).astype(np.float32)
                ).to(device)
                field[mi] = field[mi] + pf
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

    gms = np.zeros((N_DAYS, n_members))
    t0 = time.time()
    loader = iter(data.loader)
    for step in range(N_DAYS):
        forcing = next(loader)
        with torch.no_grad():
            _, state = stepper.predict_paired(
                state, forcing=forcing, compute_derived_variables=False
            )
        sd = state.as_batch_data().data
        gm = (sd[VAR][:, -1:].double() * w_t).mean(dim=(-2, -1))
        gms[step] = gm.squeeze().cpu().numpy()
        if (step + 1) % 30 == 0:
            print(f"  step {step + 1}/{N_DAYS} ({time.time() - t0:.0f}s)")

    np.savez(
        os.path.join(OUT, "global_means.npz"),
        gms=gms,
        member_specs=np.array(
            [f"{s[0]}:{s[1]}" if s else "unpert" for s in member_specs]
        ),
    )

    base = gms[:, :N_UNPERT].mean(axis=1)
    days = np.arange(1, N_DAYS + 1)
    lines = [
        "# Probe 4: spatial structure of the neutral moisture mode",
        "",
        f"Snapshot step {SNAPSHOT_STEP} ({snap_time:%Y-%m-%d}), {VAR},",
        f"gm displacement +/-1 std ({std_v:.3e}), {N_REPL} replicates each.",
        "",
        "## Part 1: retained gm fraction by displacement pattern",
        "",
        "| pattern | amplitude | @30d | @60d | @90d |",
        "|---|---|---|---|---|",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    mi = N_UNPERT
    for p in PATTERNS:
        for a in AMPLITUDES:
            resp = (gms[:, mi : mi + N_REPL].mean(axis=1) - base) / (a * std_v)
            ax.plot(days, resp, lw=1.0, label=f"{p} {a:+.0f} std")
            lines.append(
                f"| {p} | {a:+.0f} | {resp[29]:.3f} | {resp[59]:.3f} | "
                f"{resp[89]:.3f} |"
            )
            mi += N_REPL
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axhline(0.0, color="gray", ls="-", lw=0.5)
    ax.set_xlabel("days since perturbation")
    ax.set_ylabel(f"retained fraction of gm {VAR} displacement")
    ax.set_title("Probe 4: response by displacement pattern")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pattern_response.png"), dpi=150)
    plt.close(fig)

    lines += [""] + summary_part2
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nSaved to {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
