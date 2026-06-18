"""Generate cooldown configs from var-masking training configs.

For each *-mask*.yaml training config in this directory, produces a
corresponding *-cooldown.yaml that loads the pre-cooldown checkpoint saved at
epoch 142 (training_checkpoints/pre_cooldown_ckpt.tar) and re-runs the final
8-epoch PolynomialLR cooldown with masking disabled (no input_dropout). This
isolates the effect of the cooldown phase from the input-masking schedule.
"""

import argparse
import copy
import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
WANDB_PROJECT = "VarMasking4"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v4"  # stripped from wandb run names before comparison
CONFIG_PREFIX = (
    "ace-train-config-4deg-AIMIP-"  # stripped from config stems before comparison
)
DEFAULT_CHECKPOINT_NAME = "training_checkpoints/pre_cooldown_ckpt.tar"
BEST_INFERENCE_CHECKPOINT_NAME = "training_checkpoints/best_inference_ckpt.tar"
DEFAULT_EPOCHS = 8
DEFAULT_LR = 0.0001


def source_config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}{WANDB_SUFFIX}"


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


COOLDOWN_SUFFIXES = ("-bestinfcooldown", "-cooldown")


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return {run.name for run in runs}


def _out_path_to_run_name(out_path: pathlib.Path) -> str:
    """Reconstruct the wandb run name for a generated cooldown config.

    Config stem: {CONFIG_PREFIX}{base}{cooldown_suffix}
    Run name:    {WANDB_PREFIX}{base}{WANDB_SUFFIX}{cooldown_suffix}
    (the -v4 version tag sits before the cooldown suffix).
    """
    stem = out_path.stem
    if CONFIG_PREFIX and stem.startswith(CONFIG_PREFIX):
        stem = stem[len(CONFIG_PREFIX) :]
    for cooldown_suffix in COOLDOWN_SUFFIXES:
        if stem.endswith(cooldown_suffix):
            base = stem[: -len(cooldown_suffix)]
            return f"{WANDB_PREFIX}{base}{WANDB_SUFFIX}{cooldown_suffix}"
    return f"{WANDB_PREFIX}{stem}{WANDB_SUFFIX}"


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
    out_path = HERE / f"{source_path.stem}{suffix}.yaml"

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

    _write_config(cfg, out_path, beaker_dataset_id, existing_only, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source_map = str(HERE / "wandb_to_beaker_map.json")
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=default_source_map,
        help=(
            "JSON file mapping config filename → Beaker dataset ID"
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
        for p in HERE.glob("*-mask*.yaml")
        if not p.name.endswith("-finetune.yaml")
        and not p.name.endswith("-cooldown.yaml")
        and not p.name.endswith("-bestinfcooldown.yaml")
        and p.name.startswith("ace-train-config-")
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
