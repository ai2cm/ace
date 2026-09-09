"""Generate ERA5 fine-tuning configs for the per-dataset normalization ablation.

The ablation's `fm` cells train jointly on ERA5 and C96 SHiELD under three
normalization arms crossed with module conditioning. This generator takes each
finished `nc-sfno` fm cell and writes a short fine-tuning run which warm-starts
from its last-epoch checkpoint and continues on ERA5 alone.

The question is *specialization*, not transfer: every source model has already
seen ERA5 in the mixture, so what is being asked is whether the normalization
arm a model was pretrained under leaves it better positioned to be specialized
onto ERA5. Note that the fine-tuning phase itself differentiates nothing --
with only `era5` in the train_loader, A2 and A3 both bind their `era5` group
(the same `groups/era5/` constants) and the conditional cells see a constant
one-hot -- so a difference between cells is entirely a difference between the
weights they start from.

Three things about the transformation are load-bearing:

- **The normalization block is copied verbatim.** `parameter_init` loads module
  weights only and the normalizer is rebuilt from this YAML with nothing
  checking the two agree, so re-deriving statistics from the ERA5-only data
  would silently retrain on a shifted input space. It would also collapse the
  experiment: all three arms would then read one identical set of constants.
- **The label vocabulary is taken from the checkpoint**
  (`override_labels_from_weights`). An ERA5-only train_loader shrinks
  `DatasetInfo.all_labels` to `{era5}`, which builds the conditional cells'
  label-dependent weights narrower than the checkpoint's and fails to load.
  It also matters for the unconditional cells, whose checkpoints would
  otherwise record a different vocabulary than their siblings'.
- **The vertical coordinate is taken from the checkpoint**
  (`override_vertical_coordinate_from_weights`), rather than being re-derived
  from the restricted train_loader, so it stays consistent from pretraining
  through fine-tuning and inline inference. Same reasoning as the cooldown
  generator.

Usage:
    python generate_norm_ablation_finetune_configs.py
"""

import argparse
import copy
import json
import pathlib
from typing import Any

import yaml
from generate_norm_ablation_configs import (
    ARMS,
    CONFIG_PREFIX,
    config_name,
    label_for_member,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
WANDB_PREFIX = "ace2-fm-"

#: Only the sfno architecture and the joint-training regime are fine-tuned. The
#: c96 cells have no `era5` normalization group to bind to, and the era5 cell is
#: already an ERA5 specialist.
ARCH = "nc-sfno"
REGIME = "fm"

#: The unmasked cells; the mask10 twins are a separate axis and are not
#: fine-tuned.
MASKING = ""

#: Filename inside the source run's Beaker result dataset. This is the
#: last-epoch checkpoint, the same file submit_eval_jobs.py evaluates as
#: `-lastepoch`, so the fine-tune departs from the weights the ablation's
#: existing figures describe.
CHECKPOINT_NAME = "training_checkpoints/ckpt.tar"

FINETUNE_SUFFIX = "-finetune"
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-5

#: The label identifying the ERA5 members of a loader, per LABEL_MARKERS.
ERA5_LABEL = "era5"


def source_cells() -> list[tuple[str, bool]]:
    """The (arm, conditional) cells to fine-tune, in a stable order."""
    return [(arm, conditional) for arm in ARMS for conditional in (False, True)]


def source_config_name(arm: str, conditional: bool) -> str:
    return config_name(ARCH, REGIME, arm, conditional, MASKING)


def config_to_run_name(config_filename: str) -> str:
    stem = pathlib.Path(config_filename).stem
    return f"{WANDB_PREFIX}{stem.removeprefix(CONFIG_PREFIX)}"


def _build_scheduler(epochs: int) -> dict[str, Any]:
    """A single decay over the whole fine-tune, with no warmup.

    The source checkpoint is already at the end of its own cooldown, so there
    is nothing to warm up to and a monotone decay keeps the per-epoch
    checkpoints an interpretable trajectory rather than a warmup transient.
    """
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


def _dataset_members(dataset: Any) -> list[dict]:
    """The XarrayDataConfig members of a loader's dataset config."""
    if isinstance(dataset, list):
        return dataset
    for key in ("concat", "merge"):
        if key in dataset:
            return dataset[key]
    return [dataset]


def _is_era5(dataset: Any) -> bool:
    """Whether every member of a loader's dataset is ERA5.

    Membership is resolved with the generator's own LABEL_MARKERS rather than a
    bare substring test, so an unrecognized member raises instead of being
    silently treated as non-ERA5.
    """
    labels = {label_for_member(member) for member in _dataset_members(dataset)}
    return labels == {ERA5_LABEL}


def _restrict_train_loader_to_era5(cfg: dict) -> None:
    """Drop the non-ERA5 members of the train_loader concat."""
    dataset = cfg["train_loader"]["dataset"]
    members = dataset["concat"]
    era5_members = [m for m in members if label_for_member(m) == ERA5_LABEL]
    if not era5_members:
        raise ValueError("train_loader has no ERA5 members to fine-tune on.")
    dataset["concat"] = era5_members
    # group_weights partitioned the original concat and is now stale. The fm
    # configs do not set it, but a source config that did would silently
    # mis-weight the restricted concat.
    cfg["train_loader"].pop("group_weights", None)


def _clear_inference_epochs(cfg: dict) -> None:
    """Run every inline-inference entry on every fine-tuning epoch.

    The source configs schedule inference with `{start: 0, step: 10}`, sized
    for a 150-epoch run. That slice indexes the epoch list `[1..max_epochs]`,
    so on a 10-epoch run it selects epoch 1 alone. Dropping the field falls
    back to the default of every epoch.
    """
    for entry in cfg["inference"]:
        entry.pop("epochs", None)


def _select_checkpoints_on_era5(cfg: dict) -> None:
    """Zero the checkpoint-selection weight of every non-ERA5 inference entry.

    The source configs weight both the ERA5 `10year` entry and the C96
    `10year_insample_ensemble_varying_co2` entry at 1.0, so best_inference_ckpt
    would be selected half on the data the fine-tune has just stopped training
    on. The C96 entries stay in the suite as weight-0 forgetting diagnostics.
    """
    for entry in cfg["inference"]:
        if entry.get("weight", 1.0) > 0.0 and not _is_era5(entry["loader"]["dataset"]):
            entry["weight"] = 0.0


def build_config(
    source_cfg: dict, checkpoint_path: str, epochs: int, lr: float
) -> dict:
    cfg = copy.deepcopy(source_cfg)

    cfg["stepper_training"]["parameter_init"] = {
        "weights_path": checkpoint_path,
        "override_labels_from_weights": True,
        "override_vertical_coordinate_from_weights": True,
    }

    _restrict_train_loader_to_era5(cfg)
    _clear_inference_epochs(cfg)
    _select_checkpoints_on_era5(cfg)

    cfg["max_epochs"] = epochs
    cfg["optimization"]["lr"] = lr
    cfg["optimization"]["scheduler"] = _build_scheduler(epochs)
    # One EMA checkpoint per epoch. EMA rather than raw because
    # validate_using_ema is set, so EMA is the weight set every metric and
    # every downstream eval config reads.
    cfg["ema_checkpoint_save_epochs"] = {"start": 1, "step": 1}
    # Run the whole suite on the loaded checkpoint before any gradient step, so
    # the pre-fine-tune baseline is measured under this config's own
    # aggregators rather than pulled from the source run.
    cfg["evaluate_before_training"] = True
    # No new pre-cooldown checkpoint during this short run.
    cfg["pre_cooldown_checkpoint_epoch"] = None

    return cfg


def _write_config(cfg: dict, out_path: pathlib.Path, beaker_dataset_id: str) -> None:
    header = f"# arg: --dataset {beaker_dataset_id}:/checkpoints\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=str(HERE / "wandb_to_beaker_map.json"),
        help="JSON file mapping wandb run name -> Beaker dataset ID.",
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
        help=f"Base LR for the fine-tune (default: {DEFAULT_LR}).",
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    for arm, conditional in source_cells():
        source_name = source_config_name(arm, conditional)
        source_path = RUN_CONFIGS_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"{source_name} not found — run "
                "generate_norm_ablation_configs.py first"
            )
        source_run_name = config_to_run_name(source_name)
        beaker_dataset_id = source_map.get(source_run_name)
        if beaker_dataset_id is None:
            # No result dataset recorded means the source run has not finished,
            # so its last-epoch checkpoint does not exist yet.
            raise ValueError(
                f"No Beaker dataset ID for {source_run_name!r} in "
                f"{args.source_map} — has the source run finished? "
                "Refresh with update_beaker_map.py."
            )
        with source_path.open() as f:
            source_cfg = yaml.safe_load(f)
        cfg = build_config(
            source_cfg,
            f"/checkpoints/{CHECKPOINT_NAME}",
            args.epochs,
            args.lr,
        )
        out_path = RUN_CONFIGS_DIR / f"{source_path.stem}{FINETUNE_SUFFIX}.yaml"
        _write_config(cfg, out_path, beaker_dataset_id)


if __name__ == "__main__":
    main()
