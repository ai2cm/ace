#!/usr/bin/env python
# ruff: noqa: E501
# mypy: ignore-errors
"""Compose the 1deg UFS + ERA5 coupled fine-tune configs (paper-2 stage 3; Troy 2026-11-05).

Oceans: the two 1deg UFS fine-tunes of the CM4 hybrid-residual + diagnosed-SST ppad Samudra (../2026-11-02-1deg-ufs-ftcm4):
  noohc  corrector-free (the paper-2 line; wandb ace-samudra-cm4/s5j0xr6o, results 01M6H7NBRG4W31DHT2MKD4WBWA)
  ohc    scaled-temperature OHC corrector kept (wandb 75s8184h, results 01M6HWDXRVCZC1W5HFD5WPM19Z), added 2026-11-09 to test
         whether the 1deg corrector-free ocean's coupled cooling is a heat-bookkeeping failure the corrector removes Atmospheres (one arm each, Troy: "find Jeremy's 1deg 6-hourly ACE2S paper model, also FT with the
ACE weights we used before, and compare"):
  ace2s6h   Jeremy McGibbon's 6-hourly ACE2S paper arm, 12-step fine-tune (inputs/ace2s6h-paper-stepper.yaml)
  ace2s512  the ERA5 ACE2S of the summer-2026 SamudrACE-UFS coupled runs (inputs/ace2s512-era5-6h-stepper.yaml)
Recipe (Troy 2026-11-04: every new coupled FT is FTO first, then alternating coupled FT): stage 1 "fto" = ocean trained
against the interactive but frozen atmosphere; stage 2 "fto-then-alt" = Jeremy's alternating one-component-per-batch coupled
fine-tune (Elynn's 1deg template, inputs/elynn-alternating-coupled-ft-config.yaml: 8 coupled steps, 2 members, EnsembleLoss,
lr 7e-5) warm-started from the ocean lifted out of stage 1 and the original atmosphere checkpoint.
Data: ocean = the 1deg UFS 2026-10-06 replay store with the CM4 placeholder fields (weka); atmosphere = the ERA5 1deg 6-hourly
2026-03-19 store (weka) MERGED behind a surface-fraction store derived from the UFS ocean (land_fraction, sea_surface_fraction,
sea_ice_fraction, ocean_fraction; beaker dataset at /era5frac), so the atmosphere sees the ocean's land mask, as the CM4
coupled-atmosphere stores do (ERA5's own land mask differs from MOM6's in 4707 of 64800 cells). ERA5 surface stress is minus the
FV3/CM4 momentum flux the ocean was trained on (corr -0.76 vs the UFS 5-day means): atmosphere_output_rename + scale -1.
UFS protocol: train 1994-2001 + 2012-2021 (the 2002-2011 decade is never touched), validation 2022-2023, inline 10-yr coupled
rollouts (730 ocean steps) from eight 2012-2013 starts (one per GPU), 12 epochs per stage, UFS ocean + ERA5 atmosphere stats.
    python make_configs.py
"""

import copy
import itertools
import os
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
N_GPUS = int(os.environ.get("N_GPUS", "8"))
STRIP = (
    "loss",
    "optimize_last_step_only",
    "n_ensemble",
    "parameter_init",
    "train_n_forward_steps",
)

UFS = {
    "data_path": "/climate-default",
    "file_pattern": "2026-10-06-ufs-replay-ocean-1deg-19level-1994-2023-cm4vars.zarr",
    "engine": "zarr",
}
FRAC = {
    "data_path": "/era5frac",
    "file_pattern": "2026-11-05-era5-1deg-6h-ufs-coupled-fractions-1994-2023.zarr",
    "engine": "zarr",
}
ERA5 = {
    "data_path": "/climate-default",
    "file_pattern": "2026-03-19-era5-1deg-8layer-1940-2025.zarr",
    "engine": "zarr",
}
START, END = (
    "1994-01-05T18:00:00",
    "2023-10-10T18:00:00",
)  # the UFS 5-day stamps (window end - 6 h, 18Z)
INLINE_ICS = [  # UFS stamps, 10-yr windows that stay inside the record and never touch 2002-2011
    "2012-01-01T18:00:00",
    "2012-04-05T18:00:00",
    "2012-07-04T18:00:00",
    "2012-10-02T18:00:00",
    "2013-01-05T18:00:00",
    "2013-04-05T18:00:00",
    "2013-07-04T18:00:00",
    "2013-10-02T18:00:00",
]
assert len(INLINE_ICS) % N_GPUS == 0


def pair(start=None, stop=None):
    # fme's MergeDatasetConfig takes a variable from the FIRST source that holds it: the UFS-derived fractions win.
    # Merged members must share one time axis: training/validation subset both members identically; inference may not
    # subset (the loader clips every member to start at the ocean's first time and run to the member's end, which is why
    # the fraction store ends exactly where the ERA5 store ends).
    sub = {"subset": {"start_time": start, "stop_time": stop}} if start else {}
    return {
        "ocean": UFS | copy.deepcopy(sub),
        "atmosphere": {"merge": [FRAC | copy.deepcopy(sub), ERA5 | copy.deepcopy(sub)]},
    }


template = yaml.safe_load(
    (HERE / "inputs/elynn-alternating-coupled-ft-config.yaml").read_text()
)
OCEAN_FILES = {  # the two 1deg UFS fine-tunes of the CM4 hybrid-residual + diagnosed-SST ppad Samudra (../2026-11-02-1deg-ufs-ftcm4)
    "noohc": "resid-noohc-ppad-sstdiag-fmean-ufsft.yaml",  # corrector dropped (paper-2 line); wandb s5j0xr6o, results 01M6H7NBRG4W31DHT2MKD4WBWA
    "ohc": "resid-ohc-ppad-sstdiag-fmean-ufsft.yaml",  # scaled-temperature OHC corrector kept; wandb 75s8184h, results 01M6HWDXRVCZC1W5HFD5WPM19Z
}
OCEAN_DESC = {
    "noohc": "corrector-free 1deg UFS Samudra (resid-noohc-ppad-sstdiag-fmean-ufsft, /ocean_ckpt.tar)",
    "ohc": "corrector-kept 1deg UFS Samudra (resid-ohc-ppad-sstdiag-fmean-ufsft, scaled-temperature OHC corrector; /ocean_ckpt.tar)",
    "ohc-uhm1p7": "corrector-kept 1deg UFS Samudra with constant_unaccounted_heating -1.7 W/m2 in the OHC corrector (the offset that matched the replay's 30-yr column heat at inference, 2026-11-10; /ocean_ckpt.tar)",
}
# the ohc ocean with a constant unaccounted heating in its corrector DURING coupled training (Troy 2026-11-10: the offset
# experiments "potentially fix this problem"): same checkpoints as "ohc", only the corrector constant differs
OCEAN_FILES["ohc-uhm1p7"] = OCEAN_FILES["ohc"]
OCEAN_UH = {"ohc-uhm1p7": -1.7}


def load_ocean(name):
    ocean_cfg = yaml.safe_load(
        (HERE.parent / "2026-11-02-1deg-ufs-ftcm4" / OCEAN_FILES[name]).read_text()
    )
    ocean = {k: v for k, v in ocean_cfg["stepper"].items() if k not in STRIP}
    corr = ocean["step"]["config"]["corrector"]["config"]
    assert corr["corrector_disabled_epochs"] == 0
    assert ("ocean_heat_content_correction" in corr) == name.startswith("ohc")
    if name in OCEAN_UH:
        corr["ocean_heat_content_correction"]["constant_unaccounted_heating"] = (
            OCEAN_UH[name]
        )
    assert (
        "sst" not in ocean["step"]["config"]["in_names"]
        and "sst" in ocean["step"]["config"]["out_names"]
    )
    ocean["step"]["config"]["builder"]["config"]["checkpoint_strategy"] = (
        "all"  # 1deg pair on 80 GiB GPUs (2026-10-23)
    )
    return ocean


ATMOS_FILES = {
    "ace2s6h": "ace2s6h-paper-stepper.yaml",
    "ace2s512": "ace2s512-era5-6h-stepper.yaml",
}
ATMOS = {
    "ace2s6h": "Jeremy McGibbon's 6-hourly ACE2S paper arm (12-step fine-tune, wandb cerls68n, ema_ckpt_0002)",
    "ace2s512": "the summer-2026 SamudrACE-UFS ERA5 ACE2S (embed 512, no bottleneck; wandb u9ek57d9, best_inference_ckpt)",
}


def base(arm, ocean_name):
    ocean = load_ocean(ocean_name)
    sic = ocean["step"]["config"]["corrector"]["config"]["sea_ice_fraction_correction"][
        "sea_ice_fraction_name"
    ]
    atmos = yaml.safe_load((HERE / "inputs" / ATMOS_FILES[arm]).read_text())["stepper"]
    atmos = {k: v for k, v in atmos.items() if k not in STRIP}
    atmos["step"]["config"]["builder"]["config"]["checkpointing"] = 3
    cfg = copy.deepcopy(template)
    cfg["max_epochs"] = 12
    cfg["checkpoint_every_n_batches"] = (
        50  # ~165 batches per epoch: a preempted segment loses at most ~20 min
    )
    cfg.pop("resume_results", None)
    cfg["inference"] = [
        {
            "name": "inference_10yr",
            "weight": 1.0,
            "n_coupled_steps": 730,
            "coupled_steps_in_memory": 2,
            "loader": {
                "num_data_workers": 2,
                "start_indices": {"times": list(INLINE_ICS)},
                "dataset": pair(),
            },
            "aggregator": {"log_zonal_mean_images": False, "log_histograms": False},
        }
    ]
    cfg["logging"]["project"] = "ace-samudra-coupled-cm4"
    cfg["logging"]["metrics_log_dir"] = "metrics"
    cfg["train_loader"]["dataset"] = {
        "concat": [
            pair(START, "2001-12-31T18:00:00"),
            pair("2012-01-01T18:00:00", "2021-12-31T18:00:00"),
        ]
    }
    cfg["validation"] = [
        {
            "name": "val_2022_2023",
            "evaluate_all_steps": False,
            "weight": 1.0,
            "loader": {
                "batch_size": N_GPUS,  # one validation sample per GPU (the 1deg pair OOMed at 8 per 4 GPUs, 2026-10-25)
                "num_data_workers": 2,
                "prefetch_factor": 2,
                "dataset": pair("2022-01-03T18:00:00", END),
            },
        }
    ]
    # ~165 iterations per epoch on the 18 training years: 300 warm-up iterations ~ 2 epochs (the 4deg UFS recipe)
    cfg["optimization"]["scheduler"]["kwargs"]["total_iters"] = 300
    st = cfg["stepper_training"]
    st.pop("parameter_init", None)
    st["ocean"]["parameter_init"] = {"weights_path": "/ocean_ckpt.tar"}
    st["atmosphere"]["parameter_init"] = {"weights_path": "/atmos_ckpt.tar"}
    assert (
        st["optimize_single_component_per_batch"] is True
        and st["n_coupled_steps"] == 8
        and st["n_ensemble"] == 2
    )
    cfg["stepper"] = {
        "sst_name": "sst",
        "ocean_fraction_prediction": {
            "land_fraction_name": "land_fraction",
            "sea_ice_fraction_name": sic,
            "sea_ice_fraction_name_in_atmosphere": "sea_ice_fraction",
        },
        "atmosphere_output_rename": {
            "eastward_surface_wind_stress": "eastward_surface_stress",
            "northward_surface_wind_stress": "northward_surface_stress",
        },
        "atmosphere_output_scale": {
            "eastward_surface_wind_stress": -1.0,
            "northward_surface_wind_stress": -1.0,
        },
        "ocean": {"timedelta": "5D", "stepper": copy.deepcopy(ocean)},
        "atmosphere": {"timedelta": "6h", "stepper": atmos},
    }
    return cfg


for (oc, odesc), (arm, desc) in itertools.product(OCEAN_DESC.items(), ATMOS.items()):
    alt = base(arm, oc)
    fto = copy.deepcopy(alt)
    st = fto["stepper_training"]
    st["optimize_single_component_per_batch"] = False
    at = st["atmosphere"]
    at["parameter_init"] = {
        "weights_path": "/atmos_ckpt.tar",
        "parameters": [{"frozen": {"include": ["*"]}}],
    }
    at["loss_weight"] = 0.0
    at["n_steps"] = 0
    # FTO with the ocean-only fine-tune's deterministic recipe (MSE, 4-step windows, loss on every step, 1 member) instead of the
    # 2-member CRPS last-step-only recipe: the coupled cooling deep dive's discriminator (2026-11-10). Ocean-only rollouts of the
    # FTO-lifted ocean cool under replay fluxes, so coupled training itself enlarges the ocean's heat sink; this tells whether
    # the loss/window recipe or the ACE forcing does it.
    fto_mse = copy.deepcopy(fto)
    stm = fto_mse["stepper_training"]
    stm["n_coupled_steps"] = 4
    stm["n_ensemble"] = 1
    stm["ocean"]["n_steps"] = 4
    stm["ocean"]["optimize_last_step_only"] = False
    stm["ocean"]["loss"] = {"type": "MSE"}
    stages = {
        "fto-mse": (
            fto_mse,
            [
                f"ufs1deg-{oc}-{arm}-fto-mse: FTO as ufs1deg-{oc}-{arm}-fto but with the ocean-only fine-tune's deterministic recipe",
                "(MSE loss on every step of 4-step windows, 1 member) instead of 2-member CRPS on the last step of 1-8-step windows:",
                "the coupled cooling deep dive's loss-recipe discriminator (2026-11-10).",
                f"Launch: OCEAN={oc} ARM={arm} STAGE=fto-mse ./launch-1deg-ufs-era5-coupled-ft.sh",
            ],
        ),
        "fto": (
            fto,
            [
                f"ufs1deg-{oc}-{arm}-fto: FTO (ocean trained, atmosphere interactive but frozen, every batch an ocean update) of the",
                f"{odesc} with",
                f"{desc} (/atmos_ckpt.tar). Stage 1 of the FTO -> alternating coupled FT recipe (Troy 2026-11-04).",
                f"Launch: OCEAN={oc} ARM={arm} STAGE=fto ./launch-1deg-ufs-era5-coupled-ft.sh",
            ],
        ),
        "fto-then-alt": (
            alt,
            [
                f"ufs1deg-{oc}-{arm}-fto-then-alt: alternating coupled fine-tune (Jeremy's one-component-per-batch recipe, Elynn's 1deg",
                f"template) of the FTO-stage {odesc.split(' (')[0]} (lifted from ufs1deg-{oc}-{arm}-fto, /ocean_ckpt.tar) with {desc}",
                "(/atmos_ckpt.tar, the original checkpoint). Stage 2 of the FTO -> alternating coupled FT recipe (Troy 2026-11-04).",
                f"Launch: OCEAN={oc} ARM={arm} STAGE=fto-then-alt OCEAN_DS=<lifted FTO ocean dataset> ./launch-1deg-ufs-era5-coupled-ft.sh",
            ],
        ),
    }
    for stage, (c, header) in stages.items():
        hdr = "".join(f"# {line}\n" for line in header)
        out = HERE / "coupled" / f"ufs1deg-{oc}-{arm}-{stage}.yaml"
        out.write_text(hdr + yaml.safe_dump(c, sort_keys=False, width=110))
        r = copy.deepcopy(c)
        r["resume_results"] = {
            "existing_dir": "/existing-results",
            "resume_wandb": True,
        }
        out_r = HERE / "coupled" / f"ufs1deg-{oc}-{arm}-{stage}-resume.yaml"
        out_r.write_text(
            hdr.replace(
                "Launch:", "Resume-launch (EXISTING_RESULTS_DS=<results dataset>):"
            )
            + yaml.safe_dump(r, sort_keys=False, width=110)
        )
        print("wrote", out, "and", out_r.name)
