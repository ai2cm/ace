#!/usr/bin/env python
# ruff: noqa
"""Compose the 1deg coupled fine-tune configs for the random-CO2 replay test (Troy 2026-10-22).

Template: Elynn's 1deg alternating coupled fine-tune (Jeremy's one-component-per-batch recipe,
inputs/elynn-alternating-coupled-ft-config.yaml: n_coupled_steps 8, n_ensemble 2, EnsembleLoss, lr 7e-5 with
linear warm-up, 8 epochs, 1deg 2026-07-15 om4 coupled stores). Into it go
  - the ocean stepper block of James's 1deg ocean_ppad_fmean rs0 (wandb ace-samudra-cm4/mnh7jo06,
    inputs/samudra-ocean_ppad_fmean-rs0-config.yaml), corrector warm-up off for the fine-tune;
  - the atmosphere stepper block of Spencer's 1deg random-CO2 ACE2S rs1
    (cm4_randco2v2_ic1and2_1pct_46to125_piC_156to235-atmos_ace2s_origLR-ft-rs1, inputs/ace-randco2v2-origLR-ft-rs1-config.yaml);
  - weights from /ocean_ckpt.tar and /atmos_ckpt.tar (mounted by launch-1deg-coupled-ft.sh).
Arms: alt (the alternating baseline) and alt-replay (a third of batches are random-CO2 atmosphere-only updates,
Spencer's step schedule, from the six 1deg random-CO2 training members of the rs1 fine-tune).
    python make_1deg_configs.py
"""

import copy
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
template = yaml.safe_load(
    (HERE / "inputs/elynn-alternating-coupled-ft-config.yaml").read_text()
)
ocean_cfg = yaml.safe_load(
    (HERE / "inputs/samudra-ocean_ppad_fmean-rs0-config.yaml").read_text()
)
atmos_cfg = yaml.safe_load(
    (HERE / "inputs/ace-randco2v2-origLR-ft-rs1-config.yaml").read_text()
)

STRIP = (
    "loss",
    "optimize_last_step_only",
    "n_ensemble",
    "parameter_init",
    "train_n_forward_steps",
)
ocean = ocean_cfg["stepper"]
atmos = atmos_cfg["stepper"]
for comp in (ocean, atmos):
    for k in STRIP:
        comp.pop(k, None)
corr = ocean["step"]["config"]["corrector"]["config"]
corr["corrector_disabled_epochs"] = (
    0  # the pretrain's warm-up epoch; not for a fine-tune of a trained ocean
)

cfg = copy.deepcopy(template)
cfg["stepper"]["ocean"]["stepper"] = ocean
cfg["stepper"]["atmosphere"]["stepper"] = atmos
# Elynn's template trains with activation checkpointing in both networks; the component configs come from runs that
# did not need it. Without these the 1deg pair OOMs on an 80 GiB GPU in the first validation pass (2026-10-23).
ocean["step"]["config"]["builder"]["config"]["checkpoint_strategy"] = "all"
atmos["step"]["config"]["builder"]["config"]["checkpointing"] = 3
sic = corr["sea_ice_fraction_correction"]["sea_ice_fraction_name"]
cfg["stepper"]["ocean_fraction_prediction"]["sea_ice_fraction_name"] = sic
st = cfg["stepper_training"]
st.pop("parameter_init", None)
st["ocean"]["parameter_init"] = {"weights_path": "/ocean_ckpt.tar"}
st["atmosphere"]["parameter_init"] = {"weights_path": "/atmos_ckpt.tar"}
assert st["optimize_single_component_per_batch"] is True
cfg["logging"]["project"] = "ace-samudra-coupled-cm4"
cfg.pop("resume_results", None)

# replay data: the random-CO2 members of the rs1 fine-tune's training loader (atmosphere + sea_ice stores)
rc = []
for part in atmos_cfg["train_loader"]["dataset"]["merge"]:
    rc.append({"concat": [i for i in part["concat"] if "random-CO2" in i["data_path"]]})
assert all(len(p["concat"]) == 6 for p in rc)
atmos_loader = {
    "batch_size": 8,
    "num_data_workers": 4,
    "prefetch_factor": 4,
    "time_buffer": 10,
    "dataset": {"merge": rc},
}
spencer_steps = atmos_cfg["stepper_training"]["n_forward_steps"]

arms = {
    "alt": (
        cfg,
        [
            "1deg alternating coupled fine-tune (Jeremy's one-component-per-batch recipe, Elynn's template) of Spencer's 1deg",
            "random-CO2 ACE2S rs1 with James's 1deg ocean_ppad_fmean rs0 Samudra: the no-replay baseline (Troy 2026-10-22).",
        ],
    ),
}
rep = copy.deepcopy(cfg)
rep["train_loader_atmosphere_only"] = atmos_loader
rep["stepper_training"]["atmosphere_only"] = {
    "fraction": 0.3333,
    "n_steps": spencer_steps,
    "coupled_batches_update": "both",
}
arms["alt-replay"] = (
    rep,
    [
        "1deg alternating coupled fine-tune of the random-CO2 ACE2S rs1 + ocean_ppad_fmean Samudra WITH RANDOM-CO2 REPLAY:",
        "a third of each epoch's batches come from the 1deg random-CO2 atmosphere+sea-ice stores (ic_0001/0002 at 1x/2x/4x) and",
        "update the atmosphere alone under prescribed SST (Spencer's step schedule); coupled batches keep the 50/50 draw.",
        "Troy 2026-10-22: does replay keep the random-CO2 response at 1deg as it did at 4deg? Code: ace branch",
        "troya/2026-10-20-coupled-ft-atmosphere-replay (train_loader_atmosphere_only + stepper_training.atmosphere_only).",
    ],
)
for tag, (c, header) in arms.items():
    out = HERE / "coupled" / f"1deg-randco2rs1-fmean-{tag}.yaml"
    hdr = "".join(
        f"# {l}\n" for l in header + [f"Launch: ARM={tag} ./launch-1deg-coupled-ft.sh"]
    )
    out.write_text(hdr + yaml.safe_dump(c, sort_keys=False, width=110))
    print("wrote", out)
