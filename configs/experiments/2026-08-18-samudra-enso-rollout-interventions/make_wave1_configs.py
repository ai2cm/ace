#!/usr/bin/env python3
"""Generate the wave-1 coupled fine-tune arms of the ENSO rollout interventions.

Every arm is the baseline coupled fine-tune (`baseline-ft-run-config.yaml`,
the archived run config of troya/cm4-1pct-coupled-ft-atmos-nino-ocean-6595 --
the config that actually ran, which differs from the repo yamls) with exactly
one change, so effects are attributable. All arms share one new seed; the
control is that same config with no change, so arm-vs-control isolates the
treatment and control-vs-original-baseline measures run-to-run variance.

Arms (see the intervention matrix in the reports repo,
troya/2026-08-18-samudrace-enso-rollout-interventions):

  ctrl     no change (seed only)                         -> run variance
  wint5    ocean loss weights: thetao_1..18 + zos x5     -> 1a
  wint20   ocean loss weights: thetao_1..18 + zos x20    -> 1a
  hzn12    n_coupled_steps 4 -> 12 (60 days); ocean rollout-length outcomes
           extended to {0,1,2,4,8,12}                    -> 1b (pilot memory
           first: activations scale ~3x per sample)
  ohc      ocean corrector: ocean_heat_content_correction
           (scaled_temperature), everything else identical -> 1c(ii)
  resid    OCEAN PRETRAIN from scratch with residual (tendency) prediction,
           from the archived pretrain run config              -> 2b, stage 1;
           the coupled FT on its checkpoint is stage 2, generated once the
           pretrain finishes

Deliberately NOT generated here: the James-checkpoint arm (1c(i)). Partial
parameter init loads a positional initial slice, and that checkpoint's channel
ordering differs from this lineage's, so a safe version means adopting his
ocean stepper config wholesale -- a multi-change arm to be built separately if
wave 1 motivates it.

Usage: python make_wave1_configs.py [--seed 1] [--out-dir wave1_configs]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline-ft-run-config.yaml"
BASELINE_PRETRAIN = HERE / "baseline-pretrain-run-config.yaml"

INTERIOR = [f"thetao_{k}" for k in range(1, 19)] + ["zos"]


NINO_CHANNELS = [f"nino34_lead_{k:02d}" for k in range(1, 13)]
NINO_LABELS_ZARR = "2026-07-14-cm4-1pctCO2-140yr-ocean-no-smoothing-nino-leads.zarr"


def load_baseline() -> dict:
    with open(BASELINE) as f:
        return yaml.safe_load(f)


def strip_nino_channels(c: dict) -> dict:
    """Drop the 12 Nino3.4 readout channels from the ocean, in every arm.

    Three reasons, all load-bearing. The channels were a probe and the probe's
    work is done (diagnosis report). Their training labels are the stale
    5-month-smoothed build (the diagnosis's original verification bug), so
    retraining them would train against known-bad targets. And current main
    removed output_masking_exclude_names: the automatic output masker would
    match nino34_lead_01 to the level-1 bathymetry mask via the _NN suffix, so
    the channels cannot be exempted from masking anymore.

    The channels are the last 12 out_names, so the pretrained checkpoint's
    output-head tensors truncate to the new width by initial slice with no
    information loss for the remaining 86 channels (trim_ocean_ckpt.py).
    Applied to every arm including the control, so internal comparisons are
    matched; comparability to the original baseline runs through the control.
    """
    step = c["stepper"]["ocean"]["stepper"]["step"]["config"]
    assert step["out_names"][-12:] == NINO_CHANNELS, "nino channels not last 12"
    step["out_names"] = step["out_names"][:-12]
    # n_vector_outputs was the feature branch's pooled-MLP readout head that
    # produced these 12 channels; main's Samudra has no such head. The conv
    # trunk is 86-wide either way, so weights load without slicing once the
    # head's parameters are dropped from the checkpoint (trim_ocean_ckpt.py).
    step["builder"]["config"].pop("n_vector_outputs", None)
    c["stepper"]["ocean"]["stepper"].pop("output_masking_exclude_names", None)

    def drop_labels_zarr(node):
        if isinstance(node, dict):
            for v in node.values():
                drop_labels_zarr(v)
            if "merge" in node and isinstance(node["merge"], list):
                node["merge"] = [
                    m
                    for m in node["merge"]
                    if m.get("file_pattern") != NINO_LABELS_ZARR
                ]
        elif isinstance(node, list):
            for v in node:
                drop_labels_zarr(v)

    drop_labels_zarr(c)
    return c


def arm_ctrl(c: dict) -> dict:
    return c


def _weights(c: dict, factor: float) -> dict:
    loss = c["stepper_training"]["ocean"]["loss"]
    assert "weights" not in loss, "baseline ocean loss unexpectedly has weights"
    loss["weights"] = {name: factor for name in INTERIOR}
    return c


def arm_wint5(c: dict) -> dict:
    return _weights(c, 5.0)


def arm_wint20(c: dict) -> dict:
    return _weights(c, 20.0)


def arm_hzn12(c: dict) -> dict:
    assert c["stepper_training"]["n_coupled_steps"] == 4
    c["stepper_training"]["n_coupled_steps"] = 12
    # Extend the ocean's rollout-length distribution to the new window while
    # keeping short lengths in the mix; atmosphere outcomes (up to 41 of its
    # own steps) already fit inside the enlarged window and are unchanged.
    c["stepper_training"]["ocean"]["n_steps"] = {
        "outcomes": [
            {"steps": 0, "probability": 0.05},
            {"steps": 1, "probability": 0.15},
            {"steps": 2, "probability": 0.20},
            {"steps": 4, "probability": 0.20},
            {"steps": 8, "probability": 0.20},
            {"steps": 12, "probability": 0.20},
        ]
    }
    return c


def arm_ohc(c: dict) -> dict:
    step = c["stepper"]["ocean"]["stepper"]["step"]["config"]
    corr = step["corrector"]["config"]
    assert "ocean_heat_content_correction" not in corr
    # scaled_temperature is the method the corrected sibling fine-tunes use.
    # The budget's flux term comes from the ocean's own predicted
    # hfds_total_area, which is already in out_names.
    corr["ocean_heat_content_correction"] = {"method": "scaled_temperature"}
    return c


ARMS = {
    "ctrl": arm_ctrl,
    "wint5": arm_wint5,
    "wint20": arm_wint20,
    "hzn12": arm_hzn12,
    "ohc": arm_ohc,
}


def strip_nino_channels_pretrain(c: dict) -> dict:
    """Same strip as strip_nino_channels, for the ocean-only pretrain layout."""
    step = c["stepper"]["step"]["config"]
    assert step["out_names"][-12:] == NINO_CHANNELS, "nino channels not last 12"
    step["out_names"] = step["out_names"][:-12]
    step["builder"]["config"].pop("n_vector_outputs", None)
    c["stepper"].pop("output_masking_exclude_names", None)

    def drop_labels_zarr(node):
        if isinstance(node, dict):
            for v in node.values():
                drop_labels_zarr(v)
            if "merge" in node and isinstance(node["merge"], list):
                node["merge"] = [
                    m
                    for m in node["merge"]
                    if m.get("file_pattern") != NINO_LABELS_ZARR
                ]
        elif isinstance(node, list):
            for v in node:
                drop_labels_zarr(v)

    drop_labels_zarr(c)
    return c


def make_resid_pretrain(seed: int) -> dict:
    """Ocean pretraining from scratch with residual (tendency) prediction.

    Slow interior fields change little per 5-day step, so after full-field
    normalization the tendency signal the loss sees is tiny; residual
    prediction re-poses the problem as increments. It changes what the network
    output means, so this cannot initialize from the existing checkpoint: it
    pretrains from scratch on the archived pretrain run config, minus the nino
    channels, plus residual_prediction. The coupled FT on the resulting
    checkpoint is stage 2.
    """
    with open(BASELINE_PRETRAIN) as f:
        c = yaml.safe_load(f)
    c = strip_nino_channels_pretrain(c)
    step = c["stepper"]["step"]["config"]
    assert not step.get("residual_prediction")
    step["residual_prediction"] = True
    c["seed"] = seed
    c["experiment_dir"] = "/results"
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="common seed for all wave-1 arms (baseline ran seed 0)",
    )
    ap.add_argument("--out-dir", type=Path, default=HERE / "wave1_configs")
    ap.add_argument("--arms", nargs="+", default=sorted(ARMS), choices=sorted(ARMS))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        c = ARMS[arm](strip_nino_channels(copy.deepcopy(load_baseline())))
        c["seed"] = args.seed
        c["experiment_dir"] = "/results"  # run name comes from WANDB_NAME env
        path = args.out_dir / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        print(f"wrote {path}")

    rp = args.out_dir / "resid-pretrain.yaml"
    rp.write_text(yaml.safe_dump(make_resid_pretrain(args.seed), sort_keys=False))
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
