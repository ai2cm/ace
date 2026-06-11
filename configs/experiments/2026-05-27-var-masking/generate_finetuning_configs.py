"""Generate fine-tuning configs from var-masking training configs.

For each *-mask*.yaml config in this directory, produces a corresponding
*-finetune.yaml that loads weights from a finished training run and adapts
the model back to the full-input distribution via a short polynomial-decay
cooldown (no input_dropout).
"""

import argparse
import copy
import json
import pathlib

import yaml

HERE = pathlib.Path(__file__).parent
WANDB_PROJECT = "VarMasking3"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-var-mask-"  # stripped from wandb run names before comparison
WANDB_SUFFIX = "-v3"  # stripped from wandb run names before comparison
CONFIG_PREFIX = (
    "ace-train-config-4deg-AIMIP-"  # stripped from config stems before comparison
)
DEFAULT_CHECKPOINT_NAME = "training_checkpoints/best_ckpt.tar"
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.0001


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
    import wandb  # lazy import: only needed with --skip-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    names = set()
    for run in runs:
        name = run.name
        if WANDB_PREFIX and name.startswith(WANDB_PREFIX):
            name = name[len(WANDB_PREFIX):]
        if WANDB_SUFFIX and name.endswith(WANDB_SUFFIX):
            name = name[: -len(WANDB_SUFFIX)]
        names.add(name)
    return names


def _write_config(
    cfg: dict,
    out_path: pathlib.Path,
    beaker_dataset_id: str | None,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
) -> None:
    if existing_only and not out_path.exists():
        print(f"Skipped {out_path.name}")
        return
    if wandb_run_names is not None:
        stem = out_path.stem
        if CONFIG_PREFIX and stem.startswith(CONFIG_PREFIX):
            stem = stem[len(CONFIG_PREFIX):]
        if stem in wandb_run_names:
            print(f"Skipped {out_path.name} (run exists in wandb)")
            return
    if beaker_dataset_id is not None:
        header = f"# arg: --dataset {beaker_dataset_id}:/checkpoints\n"
    else:
        header = "# arg: --dataset REPLACE_WITH_BEAKER_DATASET_ID:/checkpoints\n"
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def generate_finetune_config(
    source_path: pathlib.Path,
    source_map: dict[str, str] | None,
    checkpoint_name: str,
    epochs: int,
    lr: float,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
) -> None:
    out_path = HERE / f"{source_path.stem}-v2-finetune.yaml"

    beaker_dataset_id = None
    if source_map is not None:
        beaker_dataset_id = source_map.get(source_path.name)

    with source_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)

    cfg["pre_cooldown_checkpoint_epoch"] = None

    if "stepper_training" not in cfg:
        cfg["stepper_training"] = {}
    cfg["stepper_training"]["parameter_init"] = {
        "weights_path": f"/checkpoints/{checkpoint_name}"
    }

    step_cfg = cfg["stepper"]["step"]["config"]
    step_cfg.pop("input_dropout", None)

    cfg["optimization"]["lr"] = lr
    cfg["optimization"]["scheduler"] = _build_scheduler(epochs)
    cfg["max_epochs"] = epochs

    _write_config(cfg, out_path, beaker_dataset_id, existing_only, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source_map = str(HERE / "sources_template.json")
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
        help=f"Fine-tuning epoch count (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Base LR for fine-tuning (default: {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite fine-tuning configs that already exist.",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help=(
            f"Skip configs whose run name already exists in "
            f"{WANDB_ENTITY}/{WANDB_PROJECT}."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] | None = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.skip_wandb:
        print(f"Fetching run names from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    source_configs = sorted(
        p
        for p in HERE.glob("*-mask*.yaml")
        if not p.name.endswith("-finetune.yaml")
        and p.name.startswith("ace-train-config-")
    )

    for source_path in source_configs:
        generate_finetune_config(
            source_path,
            source_map,
            args.checkpoint_name,
            args.epochs,
            args.lr,
            args.existing_only,
            wandb_run_names,
        )


if __name__ == "__main__":
    main()
