"""Generate multi-step fine-tuning configs for the paper-final var-masking runs.

Each fine-tune config is the run's **exact 1-step pre-training config** (the
config.yaml the checkpoint was trained with, cached under
``pretrain_source_configs/``) with only these changes:

  1. ``stepper_training.n_forward_steps`` is swapped from ``1`` to the multi-step
     probability schedule used by the ERA5 baseline multi-step fine-tune
     (``configs/baselines/era5/ace-train-config-multi-step-finetuning.yaml``):
     {1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}. This schedule is the *only*
     thing taken from the ERA5 baseline recipe.
  2. ``stepper_training.parameter_init.weights_path`` is added so training starts
     from the pre-trained checkpoint (mounted at ``/weights``).
  3. ``max_epochs`` is capped at FT_MAX_EPOCHS (fine-tuning is short; pre-training
     ran 150).
  4. ``stepper.step.config.input_dropout_optimized_steps_only`` is set True so
     input masking applies only on the optimized (last) rollout step, not the
     intermediate no_grad steps (a no-op for the mask0 cells). Without it,
     masking would perturb the rollout trajectory feeding the optimized step
     while inference runs unmasked.
  5. The heavy multi-year diagnostic inferences (INLINE_INFERENCE_DROP) are
     removed from inline inference -- they dominate FT wall-clock and are
     eval-only; run them post-FT via the eval tooling.

Everything else -- training/validation windows, the retained inference entries
(aimip_checkpoint + weather), optimizer (FusedAdam), EnsembleLoss (crps 0.9 /
energy 0.1, no extra weights), EMA, masking level, global-mean-removal, model
architecture -- is copied verbatim from pre-training, so the fine-tune differs
from pre-training only in that it rolls out multiple steps over a short schedule.

Two configs are written per cell, one per entry in FT_VARIANTS -- they are
identical except for which pre-training checkpoint ``parameter_init`` loads:
``-mstepft`` starts from ``best_ckpt.tar`` (lowest validation loss) and
``-mstepftaimip`` from ``best_inference_ckpt.tar`` (lowest inference error on
the weight-1.0 ``aimip_checkpoint`` entry). Pre-training writes both, so which
one to fine-tune from is an open question these two variants answer empirically.

Checkpoint dataset IDs (for the ``/weights`` mount) are resolved from
``wandb_to_beaker_map.json`` (refresh with ``update_beaker_map.py``).

Usage:
    python generate_finetune_configs.py [--source-map PATH] [--existing-only]
"""

import argparse
import copy
import json
import pathlib

import yaml
from generate_masking_configs import CONFIG_PREFIX, RUN_CONFIGS_DIR, WANDB_PREFIX

HERE = pathlib.Path(__file__).parent
DEFAULT_SOURCE_MAP = HERE / "wandb_to_beaker_map.json"
PRETRAIN_CONFIGS_DIR = HERE / "pretrain_source_configs"

# Fine-tune variants, generated for every cell: (config/run-name suffix, source
# checkpoint loaded into parameter_init). The two differ only in which epoch of
# the pre-training run they start from -- pre-training writes both.
#
#   -mstepft       best_ckpt.tar            lowest validation loss; matches the
#                                           ERA5 baseline fine-tuning recipe.
#   -mstepftaimip  best_inference_ckpt.tar  lowest inference error, which for
#                                           these runs means the weight-1.0
#                                           aimip_checkpoint entry -- the same
#                                           criterion the -bestinf evaluations
#                                           report on.
#
# Suffixes must not make eval_checkpoints.is_derived_run_name true, or the
# fine-tune runs are read as evaluations and filtered out of the run -> dataset
# map by update_beaker_map.py.
FT_VARIANTS = (
    ("-mstepft", "training_checkpoints/best_ckpt.tar"),
    ("-mstepftaimip", "training_checkpoints/best_inference_ckpt.tar"),
)

# Fine-tuning is short; cap it well below pre-training's max_epochs (150).
FT_MAX_EPOCHS = 20

# Fine-tunes exist only for the paper-final v5 cells (SELECTED_SOURCES), so
# iter_train_configs yields nothing for the other baseline versions.
FT_VERSION = "v5"

# Heavy multi-year diagnostic inferences dropped from *inline* inference for the
# short FT. Each is weight 0.0 (they do not drive checkpoint selection -- only
# aimip_checkpoint, weight 1.0, does) but costs hundreds-to-thousands of windows
# per inference-epoch (10year: 366, long_46year: 1680), so running them every
# 10 epochs dominated wall-clock. They are the *final* climate diagnostics and
# belong in the post-FT eval pass (generate_eval_configs.py / submit_eval_jobs.py),
# not inline. aimip_checkpoint (selection) and the cheap weather entries stay.
INLINE_INFERENCE_DROP = ("10year", "10year_insample", "long_46year")

# The one thing taken from the ERA5 baseline multi-step fine-tuning config: the
# n_forward_steps probability schedule. Everything else comes from pre-training.
MULTISTEP_SCHEDULE = {
    "outcomes": [
        {"steps": 1, "probability": 0.6},
        {"steps": 2, "probability": 0.2},
        {"steps": 4, "probability": 0.1},
        {"steps": 12, "probability": 0.05},
        {"steps": 20, "probability": 0.05},
    ]
}

# Paper-final source run per (global-mean-removal, masking) cell.
SELECTED_SOURCES = {
    "gmroff-mask0": "ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5",
    "gmroff-mask20": "ace2-var-mask-nc-sfno-era5-gmroff-mask20-seed1-v5",
    "gmron-mask0": "ace2-var-mask-nc-sfno-era5-gmron-mask0-seed1-v5",
    "gmron-mask20": "ace2-var-mask-nc-sfno-era5-gmron-mask20-seed0-v5",
}


def source_run_to_config_stem(source_run_name: str, ft_suffix: str) -> str:
    """Config stem for a source run name.

    ace2-var-mask-nc-sfno-era5-gmroff-mask0-seed1-v5
    -> ace-train-config-4deg-nc-sfno-era5-gmroff-mask0-seed1-v5-mstepft
    """
    suffix = source_run_name.removeprefix(WANDB_PREFIX)
    return f"{CONFIG_PREFIX}{suffix}{ft_suffix}"


def _load_pretrain_config(
    source_run_name: str, beaker_dataset_id: str | None = None
) -> dict:
    """The cached 1-step pre-training config a fine-tune is derived from."""
    pretrain_path = PRETRAIN_CONFIGS_DIR / f"{source_run_name}.yaml"
    if not pretrain_path.exists():
        dataset = beaker_dataset_id if beaker_dataset_id is not None else "<dataset>"
        raise FileNotFoundError(
            f"missing cached pre-training config {pretrain_path.name} in "
            f"{PRETRAIN_CONFIGS_DIR} — fetch it with "
            f"`beaker dataset stream-file {dataset} config.yaml`."
        )
    return yaml.safe_load(pretrain_path.read_text())


def iter_train_configs(version: str) -> list[tuple[str, dict]]:
    """``(name, config)`` for every fine-tune run of ``version``.

    ``name`` is the *fine-tune* config stem, so the eval tooling derives the
    ``-mstepft``/``-mstepftaimip`` run name from it and resolves the
    fine-tune's own checkpoint dataset. ``config`` is the *pre-training*
    config, whose inference suite still holds the multi-year diagnostics that
    INLINE_INFERENCE_DROP prunes from inline fine-tune inference. Those
    diagnostics are the reason the fine-tunes need an eval pass at all, so the
    eval suite is deliberately built from the unpruned suite; see
    generate_eval_configs.py.

    Signature mirrors iter_train_configs in generate_masking_configs.py and
    generate_seed_configs.py so all three families enumerate the same way.
    """
    if version != FT_VERSION:
        return []
    configs: list[tuple[str, dict]] = []
    for source_run_name in SELECTED_SOURCES.values():
        pretrain_cfg = _load_pretrain_config(source_run_name)
        for ft_suffix, _ in FT_VARIANTS:
            stem = source_run_to_config_stem(source_run_name, ft_suffix)
            configs.append((stem, copy.deepcopy(pretrain_cfg)))
    return configs


def _to_finetune(pretrain_cfg: dict, checkpoint_name: str) -> dict:
    """Return a fine-tune config: pre-training config with only the multi-step
    schedule, checkpoint weight-loading, and shorter max_epochs changed.
    """
    cfg = copy.deepcopy(pretrain_cfg)
    st = cfg.setdefault("stepper_training", {})
    st["n_forward_steps"] = copy.deepcopy(MULTISTEP_SCHEDULE)
    st["parameter_init"] = {"weights_path": f"/weights/{checkpoint_name}"}
    cfg["max_epochs"] = FT_MAX_EPOCHS
    # Mask only the optimized (last) step of each rollout, not the intermediate
    # no_grad steps -- otherwise masking perturbs the trajectory feeding the
    # optimized step while inference runs unmasked (train/inference mismatch).
    # No-op for the mask0 cells (max_masked_vars 0); meaningful for mask20.
    cfg["stepper"]["step"]["config"]["input_dropout_optimized_steps_only"] = True
    # Drop the heavy multi-year diagnostic inferences from inline inference (see
    # INLINE_INFERENCE_DROP); they dominate FT wall-clock and are eval-only.
    inference = cfg.get("inference")
    if isinstance(inference, list):
        cfg["inference"] = [
            e for e in inference if e.get("name") not in INLINE_INFERENCE_DROP
        ]
    return cfg


def generate_finetune_config(
    source_run_name: str,
    beaker_dataset_id: str,
    ft_suffix: str,
    checkpoint_name: str,
    existing_only: bool,
) -> None:
    stem = source_run_to_config_stem(source_run_name, ft_suffix)
    out_path = RUN_CONFIGS_DIR / f"{stem}.yaml"
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return

    pretrain_cfg = _load_pretrain_config(source_run_name, beaker_dataset_id)
    cfg = _to_finetune(pretrain_cfg, checkpoint_name)

    header = (
        f"# arg: --dataset {beaker_dataset_id}:/weights\n"
        f"# source pre-training run: {source_run_name}\n"
        "# = that run's 1-step pre-training config, with only: n_forward_steps"
        " swapped for the\n#   multi-step schedule, parameter_init added,"
        f" max_epochs capped at {FT_MAX_EPOCHS}, and\n#"
        "   input_dropout_optimized_steps_only set"
        " (see generate_finetune_configs.py).\n"
        f"# fine-tuning starts from {checkpoint_name}\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}  (weights: {beaker_dataset_id}/{checkpoint_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=str(DEFAULT_SOURCE_MAP),
        help=(
            "JSON mapping pre-training run name -> Beaker dataset ID"
            f" (default: {DEFAULT_SOURCE_MAP})."
        ),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only overwrite fine-tune configs that already exist.",
    )
    args = parser.parse_args()

    source_map = json.loads(pathlib.Path(args.source_map).read_text())

    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    missing = []
    for cell, source_run_name in SELECTED_SOURCES.items():
        dataset_id = source_map.get(source_run_name)
        if dataset_id is None:
            missing.append((cell, source_run_name))
            continue
        for ft_suffix, checkpoint_name in FT_VARIANTS:
            generate_finetune_config(
                source_run_name,
                dataset_id,
                ft_suffix,
                checkpoint_name,
                args.existing_only,
            )

    if missing:
        lines = "\n".join(f"  {cell}: {run}" for cell, run in missing)
        raise SystemExit(
            "No Beaker dataset ID in the source map for:\n"
            f"{lines}\n"
            "Refresh it with update_beaker_map.py (source run must have a "
            "succeeded job)."
        )


if __name__ == "__main__":
    main()
