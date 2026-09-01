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

# ENSO-active upper ocean: temperature AND velocities from the surface down
# through the thermocline band, plus SSH (the thermocline-displacement
# signature). Layer interfaces put levels 0..8 at 0-450 m; level 9 starts at
# 450 m and the column continues to 6.5 km, well below anything ENSO touches.
# The 450 m cutoff matches the diagnosis's subsurface-LIM evidence, which also
# stopped at 450 m. Velocities are included on the currents-arm result
# (prescribing them recovered lead-12 skill to 0.79).
ENSO_LEVELS = list(range(0, 9))  # 0-450 m
ENSO_ACTIVE = (
    [f"thetao_{k}" for k in ENSO_LEVELS]
    + [f"uo_{k}" for k in ENSO_LEVELS]
    + [f"vo_{k}" for k in ENSO_LEVELS]
    + ["ssu", "ssv", "zos"]
)


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


CORRECTED_PRETRAIN = HERE / "corrected-pretrain-run-config.yaml"
# Ocean init for all FT arms: the corrected from-scratch Samudra pretrain
# (troya/cm4-samudra-1pct-ocean-train-using-ufs-var-subset-ohc-hdfs-correctors,
# Beaker dataset 01KW2BQ83EGZ90WZ74CZ4TJATN). Its in/out names match the
# stripped nino lineage exactly, including order, so weights drop in with no
# checkpoint surgery, and it was pretrained WITH the heat-content and
# surface-flux corrections on.


def adopt_corrected_lineage(c: dict) -> dict:
    """Carry the corrected pretrain's constraints into the coupled FT.

    Sets the FT ocean corrector to the pretrain's exactly (full salinity
    positivity list, sea-ice fix, surface_energy_flux_correction and
    ocean_heat_content_correction), so the constraint the ocean was trained
    under stays active through fine-tuning and inference.
    """
    with open(CORRECTED_PRETRAIN) as f:
        pre = yaml.safe_load(f)

    def find(d):
        if isinstance(d, dict):
            if "out_names" in d and "in_names" in d:
                return d
            for v in d.values():
                r = find(v)
                if r:
                    return r
        return None

    step = c["stepper"]["ocean"]["stepper"]["step"]["config"]
    pre_step = find(pre)
    assert step["out_names"] == pre_step["out_names"]
    assert step["in_names"] == pre_step["in_names"]
    step["corrector"] = copy.deepcopy(pre_step["corrector"])
    return c


def _weights(c: dict, factor: float) -> dict:
    loss = c["stepper_training"]["ocean"]["loss"]
    assert "weights" not in loss, "baseline ocean loss unexpectedly has weights"
    loss["weights"] = {name: factor for name in ENSO_ACTIVE}
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
    # Both OOMs of this arm hit the same allocation site inside the
    # VALIDATION loop (the virtual-temperature computation), not a training
    # step: validation ran at batch 16 with windows that scale with
    # n_coupled_steps, so the 12-step window tripled validation memory while
    # the training batch was never the problem. Validation batch drops to 4
    # (12-step x 4 is ~75% of the baseline's 4-step x 16 validation memory);
    # the training batch stays at the baseline 8, keeping the effective batch
    # matched with the other arms.
    c["validation"]["loader"]["batch_size"] = 4
    return c


def arm_tendloss(c: dict) -> dict:
    """Give the ocean the tendency-scaled loss the atmosphere already has.

    In this coupled FT the atmosphere's normalization carries a `residual`
    block (loss errors on prognostics measured against per-step tendency
    stds) while the ocean's loss runs in plain full-field z-score units. In
    those units a variable's per-step dynamics contribute at
    (sigma_tend/sigma_field)^2: measured from the stats files, 1/900 for sst,
    1/2500 for thetao_8, 1/180000 for thetao_18 -- the ocean is trained
    overwhelmingly to hold still, and the diagnosis found it fails precisely
    at evolving its interior. This arm mirrors the atmosphere's residual loss
    block onto the ocean, rescaling the prognostic loss to the physically
    correct per-variable magnitude (what wint5/wint20's flat x5/x20 nudges
    were off by orders of magnitude from reaching). Network I/O normalization
    is unchanged; forward path is unchanged; this is loss-only.
    """
    norm = c["stepper"]["ocean"]["stepper"]["step"]["config"]["normalization"]
    assert "residual" not in norm and "loss" not in norm
    norm["residual"] = {
        "global_means_path": "/ocean_stats/ocean/centering.nc",
        "global_stds_path": "/ocean_stats/ocean/scaling-residual.nc",
    }
    return c


def arm_noohc(c: dict) -> dict:
    """Ablation: corrections OFF during fine-tuning (still on in pretraining).

    With the corrected pretrain as everyone's init, the attribution question
    inverts: does keeping the budget constraints active through fine-tuning
    matter, given the dynamics were learned under them?
    """
    corr = c["stepper"]["ocean"]["stepper"]["step"]["config"]["corrector"]["config"]
    assert corr.pop("ocean_heat_content_correction", None) is not None
    assert corr.pop("surface_energy_flux_correction", None) is not None
    # Ops workaround, not science: this arm twice wedged in the cold
    # pre-training inference pass (once a silent hang at window 148/720, once
    # an NCCL watchdog kill during inference-dataloader worker setup). The
    # pre-training evaluation is metrics-only, so skipping it does not affect
    # training; epoch-end evaluations still run.
    c["evaluate_before_training"] = False
    return c


ARMS = {
    "wint5": arm_wint5,
    "wint20": arm_wint20,
    "hzn12": arm_hzn12,
    "noohc": arm_noohc,
    "tendloss": arm_tendloss,
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
    with open(CORRECTED_PRETRAIN) as f:
        c = yaml.safe_load(f)
    step = c["stepper"]["step"]["config"]
    assert not any(n.startswith("nino34_lead") for n in step["out_names"])
    assert not step.get("residual_prediction")
    step["residual_prediction"] = True
    # residual_prediction MUST pair with tendency-scaled loss normalization
    # (the shield-som and 2025-03-17 residual configs both do this). Without
    # it the loss normalizes tendencies by full-field stds; slow interior
    # fields' tendencies are ~1% of field std, the dynamics contribute almost
    # nothing to the loss, and the model learns "predict no change" -- great
    # one-step validation, NaN rollouts. The first resid launch demonstrated
    # exactly that: val loss 0.33 with 'Inference error: nan' at every epoch.
    norm = step["normalization"]
    assert "network" in norm and "residual" not in norm
    norm["residual"] = {
        "global_means_path": "/ocean_stats/ocean/centering.nc",
        "global_stds_path": "/ocean_stats/ocean/scaling-residual.nc",
    }
    c["seed"] = seed
    c["experiment_dir"] = "/results"
    return c


def make_residfix_pretrain(seed: int) -> dict:
    """Residual prediction with the forward path in tendency units.

    The first resid arm demonstrated the structural failure of residual
    prediction as previously wired: the network output is added to the input
    in FULL-FIELD-normalized units, so representing a real 5-day tendency
    requires outputs of 1/30th to 1/430th of a normalized unit, while the
    tendency-scaled loss amplifies errors on the deepest levels by up to
    180,000x. Validation loss fell four orders of magnitude and every one of
    150 epoch-end rollouts was NaN.

    residual_normalized_prediction fixes the forward path: the network's
    prognostic outputs are treated as tendencies in residual-normalized units
    (one unit of output = one std of the true 5-day tendency), added to the
    input in physical units. The loss (residual block) then measures errors
    in the same units end to end.
    """
    c = make_resid_pretrain(seed)
    c["stepper"]["step"]["config"]["residual_normalized_prediction"] = True
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
        c = ARMS[arm](
            adopt_corrected_lineage(strip_nino_channels(copy.deepcopy(load_baseline())))
        )
        c["seed"] = args.seed
        c["experiment_dir"] = "/results"  # run name comes from WANDB_NAME env
        path = args.out_dir / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        print(f"wrote {path}")

    rp = args.out_dir / "resid-pretrain.yaml"
    rp.write_text(yaml.safe_dump(make_resid_pretrain(args.seed), sort_keys=False))
    print(f"wrote {rp}")

    rf = args.out_dir / "residfix-pretrain.yaml"
    rf.write_text(yaml.safe_dump(make_residfix_pretrain(args.seed), sort_keys=False))
    print(f"wrote {rf}")


if __name__ == "__main__":
    main()
