"""Generate cooldown configs from the var-masking training configs.

For each ``*-mask*.yaml`` training config in ``run_configs/``, produces a
corresponding ``*-cooldown.yaml`` (and ``*-bestinfcooldown.yaml``) that loads a
pre-trained checkpoint and re-runs a short PolynomialLR cooldown with masking
disabled (no ``input_dropout``). This isolates the effect of the cooldown phase
from the input-masking schedule.

Two checkpoint variants are emitted per source config:

  - ``-cooldown``:        loads ``training_checkpoints/ckpt.tar`` (the last
                          training checkpoint, always written by training).
  - ``-bestinfcooldown``: loads ``training_checkpoints/best_inference_ckpt.tar``
                          (always written by training).

The generated configs are written into ``run_configs/`` so that
run-ace-train.sh (which reads from that subdir) can submit them.
"""

import argparse
import copy
import json
import pathlib

import yaml
from generate_masking_configs import (
    BASE_CONFIG_FILENAMES,
    CONFIG_PREFIX,
    RUN_CONFIGS_DIR,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    stem_has_version,
)

HERE = pathlib.Path(__file__).parent
DEFAULT_CHECKPOINT_NAME = "training_checkpoints/ckpt.tar"
BEST_INFERENCE_CHECKPOINT_NAME = "training_checkpoints/best_inference_ckpt.tar"
DEFAULT_EPOCHS = 8
DEFAULT_LR = 0.0001


def source_config_to_run_name(config_filename: str) -> str:
    """Wandb run name for a training config filename (stem ends in -v1/-v2/-v3/-v4)."""
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


def _clear_inference_epochs(cfg: dict) -> None:
    """Run inference every cooldown epoch.

    The base config schedules inference at ``epochs.start: 10`` (step 10), but a
    cooldown only runs a handful of epochs, so that schedule never fires. Drop
    the ``epochs`` key entirely; its default (empty ``Slice``) runs inference on
    every epoch of the cooldown.
    """
    entries = cfg.get("inference", [])
    if isinstance(entries, dict):
        entries = [entries]
    for entry in entries:
        entry.pop("epochs", None)


def _build_scheduler(epochs: int) -> dict:
    return {
        "schedulers": [
            {
                "type": "PolynomialLR",
                "kwargs": {"power": 0.5, "total_iters": epochs},
                "step_each_iteration": False,
            }
        ],
        "milestones": [],
    }


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return {run.name for run in runs}


def _out_path_to_run_name(out_path: pathlib.Path) -> str:
    """Reconstruct the wandb run name for a generated cooldown config.

    Config stem: {CONFIG_PREFIX}{base}-{version}{cooldown_suffix}
    Run name:    {WANDB_PREFIX}{base}-{version}{cooldown_suffix}
    (the version tag is already embedded in the source config's stem, which
    this cooldown config is built from — see generate_cooldown_config).
    """
    stem = out_path.stem
    if CONFIG_PREFIX and stem.startswith(CONFIG_PREFIX):
        stem = stem[len(CONFIG_PREFIX) :]
    return f"{WANDB_PREFIX}{stem}"


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    beaker_dataset_id: str | None,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
) -> None:
    if wandb_run_names is not None:
        if _out_path_to_run_name(out_path) in wandb_run_names:
            if out_path.exists():
                out_path.unlink()
                print(f"Deleted {out_path.name} (run exists in wandb)")
            else:
                print(f"Skipped {out_path.name} (run exists in wandb, no file)")
            return
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    if beaker_dataset_id is not None:
        header = f"# arg: --dataset {beaker_dataset_id}:/checkpoints\n"
    else:
        header = "# arg: --dataset REPLACE_WITH_BEAKER_DATASET_ID:/checkpoints\n"
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_cooldown_config(
    source_path: pathlib.Path,
    source_map: dict[str, str] | None,
    checkpoint_name: str,
    epochs: int,
    lr: float,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
    suffix: str = "-cooldown",
) -> None:
    out_path = RUN_CONFIGS_DIR / f"{source_path.stem}{suffix}.yaml"

    beaker_dataset_id = None
    if source_map is not None:
        beaker_dataset_id = source_map.get(source_config_to_run_name(source_path.name))

    with source_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)

    # No new pre-cooldown checkpoint during this short run.
    cfg["pre_cooldown_checkpoint_epoch"] = None

    if "stepper_training" not in cfg:
        cfg["stepper_training"] = {}
    cfg["stepper_training"]["parameter_init"] = {
        "weights_path": f"/checkpoints/{checkpoint_name}"
    }

    # No masking during the cooldown.
    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg.pop("input_dropout", None)

    cfg["optimization"]["lr"] = lr
    cfg["optimization"]["scheduler"] = _build_scheduler(epochs)
    cfg["max_epochs"] = epochs

    # Run inference every epoch of the shortened cooldown.
    _clear_inference_epochs(cfg)

    _write_config(cfg, out_path, beaker_dataset_id, existing_only, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source_map = str(HERE / "wandb_to_beaker_map.json")
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=default_source_map,
        help=(
            "JSON file mapping training run name → Beaker dataset ID"
            f" (default: {default_source_map})."
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help=(
            "Checkpoint filename inside the dataset mount"
            f" (default: {DEFAULT_CHECKPOINT_NAME})."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Cooldown epoch count (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Base LR for the cooldown (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite cooldown configs that already exist.",
    )
    parser.add_argument(
        "--version",
        "-v",
        choices=sorted(BASE_CONFIG_FILENAMES),
        default=None,
        help="Restrict to source configs of this baseline version (default: all).",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            f"Delete cooldown configs whose run name already exists in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT}."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] | None = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    source_configs = sorted(
        p
        for p in RUN_CONFIGS_DIR.glob("*-mask*.yaml")
        if not p.name.endswith("-finetune.yaml")
        and not p.name.endswith("-cooldown.yaml")
        and not p.name.endswith("-bestinfcooldown.yaml")
        and p.name.startswith(CONFIG_PREFIX)
        and (args.version is None or stem_has_version(p.stem, args.version))
    )

    variants = [
        (args.checkpoint_name, "-cooldown"),
        (BEST_INFERENCE_CHECKPOINT_NAME, "-bestinfcooldown"),
    ]
    for source_path in source_configs:
        for checkpoint_name, suffix in variants:
            generate_cooldown_config(
                source_path,
                source_map,
                checkpoint_name,
                args.epochs,
                args.lr,
                args.existing_only,
                wandb_run_names,
                suffix,
            )


if __name__ == "__main__":
    main()
