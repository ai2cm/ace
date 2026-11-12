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
import os
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
# main (2026-10-22, "Recover wandb metrics lost to preemption") requires a relative metrics_log_dir; Elynn's template
# predates it and says /results/metrics
if str(cfg["logging"].get("metrics_log_dir", "")).startswith("/"):
    cfg["logging"]["metrics_log_dir"] = "metrics"
cfg.pop("resume_results", None)
cfg["checkpoint_every_n_batches"] = (
    250  # a preempted segment loses at most ~1 h of training (was 1000)
)
# the 1deg pair sits at the edge of 80 GiB with 2 samples per GPU: the end-of-epoch validation OOMed at batch 8
# (2026-10-25); one validation sample per GPU (fme requires batch_size % world_size == 0). 2026-10-27: training itself
# OOMed at 2 samples per GPU on 4 GPUs (both arms, ~75 GiB), so the pair now runs on 8 GPUs (N_GPUS=8 ./launch-...),
# one training sample per GPU; the training batch (8) is unchanged.
N_GPUS = int(os.environ.get("N_GPUS", "8"))
for v in cfg["validation"]:
    v["loader"]["batch_size"] = N_GPUS

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
# 2026-10-29 (Troy): the same alternating baseline with the ocean swapped for the 1deg hybrid residual + diagnosed-SST
# pretrain (thetao residual, sst output-only; wandb ace-samudra-cm4/jrgclj2f), the paper-1 recipe candidate. Needs the
# coupled-stepper ocean_ic_names fix and the OceanData sst fallback (both cherry-picked onto this branch).
hyb_ocean = yaml.safe_load(
    (HERE / "inputs/samudra-resid-ohc-ppad-sstdiag-fmean-1deg-config.yaml").read_text()
)["stepper"]
for k in STRIP:
    hyb_ocean.pop(k, None)
hyb_corr = hyb_ocean["step"]["config"]["corrector"]["config"]
hyb_corr["corrector_disabled_epochs"] = 0
hyb_ocean["step"]["config"]["builder"]["config"]["checkpoint_strategy"] = "all"
assert "sst" not in hyb_ocean["step"]["config"]["in_names"]
assert "sst" in hyb_ocean["step"]["config"]["out_names"]
hyb = copy.deepcopy(cfg)
hyb["stepper"]["ocean"]["stepper"] = hyb_ocean
arms["hybridsstdiag-alt"] = (
    hyb,
    [
        "1deg alternating coupled fine-tune of the random-CO2 ACE2S rs1 with the 1deg HYBRID RESIDUAL + DIAGNOSED-SST",
        "Samudra (James's ocean_ppad_fmean rs0 recipe with thetao_0..18 residual and sst output-only; wandb jrgclj2f).",
        "Identical to -alt otherwise (Troy 2026-10-29). Launch with OCEAN_DS=01M626JPB247BQWSE0GN35SAXR.",
    ],
)
# 2026-11-02 (Troy): the hybrid residual + diagnosed-SST ocean WITH random-CO2 replay = the paper-1 1deg candidate.
hyb_rep = copy.deepcopy(rep)
hyb_rep["stepper"]["ocean"]["stepper"] = copy.deepcopy(hyb_ocean)
arms["hybridsstdiag-alt-replay"] = (
    hyb_rep,
    [
        "1deg alternating coupled fine-tune of the random-CO2 ACE2S rs1 with the 1deg HYBRID RESIDUAL + DIAGNOSED-SST",
        "Samudra (wandb jrgclj2f) WITH RANDOM-CO2 REPLAY (a third of batches atmosphere-only on the random-CO2 stores, as",
        "-alt-replay). = -hybridsstdiag-alt + replay (Troy 2026-11-02). Launch with OCEAN_DS=01M626JPB247BQWSE0GN35SAXR.",
    ],
)
# 2026-11-04 (Troy): both replay arms again with the atmosphere swapped for Spencer's 1deg random-CO2 ACE2.5 rs1
# (cm4_randco2v2_ic1and2_1pct_46to125_piC_156to235-atmos_ace2p5-ft-rs1-train, wandb wb3jfw7u, beaker
# 01M6DWYW0AG9N7MCGFC1B3Q7SQ). Same variables, normalization and training data as the ACE2S rs1; the builder adds
# grouped filters and a spectral bottleneck. The replay keeps the ACE2S arms' step schedule so the pairs differ only in
# the atmosphere. Launch with ATMOS_DS=01M6DWYW0RWBN0X33ABS98T0AH.
ace2p5_builder = yaml.safe_load(
    (HERE / "inputs/ace2p5-randco2v2-ft-rs1-config.yaml").read_text()
)["stepper"]["step"]["config"]["builder"]
for tag, base in (
    ("ace2p5-alt-replay", rep),
    ("ace2p5-hybridsstdiag-alt-replay", hyb_rep),
):
    c = copy.deepcopy(base)
    b = c["stepper"]["atmosphere"]["stepper"]["step"]["config"]["builder"]
    assert b["type"] == ace2p5_builder["type"]
    b["config"] = {
        **ace2p5_builder["config"],
        "checkpointing": b["config"]["checkpointing"],
    }
    arms[tag] = (
        c,
        [
            f"= -{tag.removeprefix('ace2p5-')} with the atmosphere swapped for the 1deg random-CO2 ACE2.5 rs1 (wandb",
            "wb3jfw7u), Troy 2026-11-04. Launch with ATMOS_DS=01M6DWYW0RWBN0X33ABS98T0AH"
            + (" OCEAN_DS=01M626JPB247BQWSE0GN35SAXR." if "hybrid" in tag else "."),
        ],
    )
# 2026-11-08 (Troy): James's two-stage coupled FT recipe with the hybrid residual + diagnosed-SST ocean. Stage 1 = his
# FTO (beaker 01M4SZFVA0A9XBQC502PPC217Q, ace 74f046947, inputs/james-1deg-fto-config.yaml): atmosphere frozen, ocean
# trained on 4 coupled steps with the loss on every step, lr 7e-5, 8 epochs, max_grad_norm 1.0, seed 0. Stage 2 = his
# alternating FT of the FTO's coupled best_inference checkpoint (beaker 01M5AR9ZMHXQPCNK2W8N18YQBS, ace e8b3f31f4,
# inputs/james-1deg-ftfto-alt-config.yaml + the optimize_single_component_per_batch=true override): lr 1e-5, 3 epochs,
# plus our random-CO2 replay. Only the ocean (hybrid, wandb jrgclj2f) and the atmosphere (the random-CO2 ACE2S rs1 the
# replay needs; James used the plain ACE2S origLR-ft rs2) differ from James; his corrector_disabled_epochs 1 is kept.
james_fto = yaml.safe_load((HERE / "inputs/james-1deg-fto-config.yaml").read_text())
james_alt = yaml.safe_load(
    (HERE / "inputs/james-1deg-ftfto-alt-config.yaml").read_text()
)
james_alt["stepper_training"]["optimize_single_component_per_batch"] = (
    True  # his launch override
)


def _james_stage(base, james):
    c = copy.deepcopy(base)
    replay = c["stepper_training"].get("atmosphere_only")
    c["stepper_training"] = copy.deepcopy(james["stepper_training"])
    if replay is not None:
        c["stepper_training"]["atmosphere_only"] = replay
    c["optimization"] = {**copy.deepcopy(james["optimization"]), "max_grad_norm": 1.0}
    c["max_epochs"] = james["max_epochs"]
    c["seed"] = 0
    c["stepper"]["ocean"]["stepper"]["step"]["config"]["corrector"]["config"][
        "corrector_disabled_epochs"
    ] = james["stepper"]["ocean"]["stepper"]["step"]["config"]["corrector"]["config"][
        "corrector_disabled_epochs"
    ]
    return c


arms["hybridsstdiag-jfto"] = (
    _james_stage(hyb, james_fto),
    [
        "Stage 1 of James's coupled FT recipe (FTO: atmosphere frozen; beaker 01M4SZFVA0A9XBQC502PPC217Q) with the 1deg",
        "HYBRID RESIDUAL + DIAGNOSED-SST Samudra (wandb jrgclj2f) and the random-CO2 ACE2S rs1, Troy 2026-11-08.",
        "Launch with OCEAN_DS=01M626JPB247BQWSE0GN35SAXR.",
    ],
)
c = _james_stage(hyb_rep, james_alt)
# 2026-11-10 (James, ace PR #1623): the replay batches now read the per-member random-CO2-*-atmosphere.zarr stores, whose
# surface_temperature is blended with the ocean slot's sst as in the piControl/1pct coupled atmosphere stores (the
# -sea_ice.zarr stores carried AM4's own surface temperature). Same variables otherwise; the earlier replay arms keep
# the -sea_ice.zarr stores they were trained on.
for part in c["train_loader_atmosphere_only"]["dataset"]["merge"]:
    for item in part["concat"]:
        item["file_pattern"] = item["file_pattern"].replace(
            "-sea_ice.zarr", "-atmosphere.zarr"
        )
assert c["stepper_training"]["parameter_init"]["checkpoint_path"] == "/ckpt.tar"
arms["hybridsstdiag-jfto-alt-replay"] = (
    c,
    [
        "Stage 2 of James's coupled FT recipe (alternating FT of the FTO checkpoint; beaker 01M5AR9ZMHXQPCNK2W8N18YQBS)",
        "WITH RANDOM-CO2 REPLAY, from the -hybridsstdiag-jfto best_inference checkpoint, Troy 2026-11-08.",
        "Launch with CKPT_DS=<the -hybridsstdiag-jfto results dataset>.",
    ],
)


# 2026-11-12 (Troy): the stage-2 ep3 checkpoint of the recipe above breaks down after ~60 yr in both centennials (a
# regional surface instability off Newfoundland) although its FTO start is stable for 200 yr. Both stages train the
# ocean with the OHC corrector off for their first epoch (corrector_disabled_epochs: 1, copied from James), and the
# corrector matters for OHC in the hybrid-residual ocean. Twins of both stages with the corrector on throughout.
def _corrector_always_on(c):
    c = copy.deepcopy(c)
    c["stepper"]["ocean"]["stepper"]["step"]["config"]["corrector"]["config"][
        "corrector_disabled_epochs"
    ] = 0
    return c


arms["hybridsstdiag-jfto-con"] = (
    _corrector_always_on(arms["hybridsstdiag-jfto"][0]),
    arms["hybridsstdiag-jfto"][1][:2]
    + [
        "Twin with the OHC corrector ON in every epoch (corrector_disabled_epochs: 0), Troy 2026-11-12.",
        "Launch with OCEAN_DS=01M626JPB247BQWSE0GN35SAXR.",
    ],
)
arms["hybridsstdiag-jfto-alt-replay-con"] = (
    _corrector_always_on(arms["hybridsstdiag-jfto-alt-replay"][0]),
    arms["hybridsstdiag-jfto-alt-replay"][1][:2]
    + [
        "Twin with the OHC corrector ON in every epoch (corrector_disabled_epochs: 0), Troy 2026-11-12.",
        "Launch with CKPT_DS=<the -hybridsstdiag-jfto-con results dataset>.",
    ],
)
for tag, (c, header) in arms.items():
    out = HERE / "coupled" / f"1deg-randco2rs1-fmean-{tag}.yaml"
    hdr = "".join(
        f"# {l}\n" for l in header + [f"Launch: ARM={tag} ./launch-1deg-coupled-ft.sh"]
    )
    out.write_text(hdr + yaml.safe_dump(c, sort_keys=False, width=110))
    print("wrote", out)
    # twin that continues a previous submission's results (mounted at /existing-results by the launcher when
    # EXISTING_RESULTS_DS is set): same run, new Beaker job spec (e.g. to add --min-runtime)
    r = copy.deepcopy(c)
    r["resume_results"] = {"existing_dir": "/existing-results", "resume_wandb": True}
    out_r = HERE / "coupled" / f"1deg-randco2rs1-fmean-{tag}-resume.yaml"
    out_r.write_text(
        hdr.replace("Launch:", "Resume-launch (EXISTING_RESULTS_DS=<results dataset>):")
        + yaml.safe_dump(r, sort_keys=False, width=110)
    )
    print("wrote", out_r)
