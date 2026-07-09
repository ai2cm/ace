"""Generate cooldown configs from FM training configs.

For each nc-sfno training config in this directory, produces a corresponding
*-cooldown.yaml that loads the pre-cooldown checkpoint saved at epoch 142
(training_checkpoints/pre_cooldown_ckpt.tar) and re-runs the final 8-epoch
PolynomialLR cooldown with masking disabled (no input_dropout). This isolates
the effect of the cooldown phase from the input-masking schedule.

The cooldown also restricts the training data to a single target dataset. The
FM configs train on a mix of datasets (SHiELD AMIP, random-CO2, SOM, and ERA5),
so the cooldown "cools down" from that multi-dataset mixture onto ERA5 alone:
the non-ERA5 members are dropped from the train_loader concat and the
multi-group group_weights are removed. A run trained on a single dataset (e.g.
a pure C96/SHiELD run) instead cools down onto that dataset, keeping all its
members. A run trained on multiple datasets none of which is ERA5 has no
unambiguous target and raises.
"""

import argparse
import copy
import json
import pathlib

import yaml
from _version_select import add_version_arg, stem_matches_version

HERE = pathlib.Path(__file__).parent
WANDB_PROJECT = "FM"
WANDB_ENTITY = "ai2cm"
WANDB_PREFIX = "ace2-fm-"  # stripped from wandb run names before comparison
CONFIG_PREFIX = (
    "ace-train-config-4deg-AIMIP-"  # stripped from config stems before comparison
)
DEFAULT_CHECKPOINT_NAME = "training_checkpoints/pre_cooldown_ckpt.tar"
BEST_INFERENCE_CHECKPOINT_NAME = "training_checkpoints/best_inference_ckpt.tar"
DEFAULT_EPOCHS = 8
DEFAULT_LR = 0.0001
ERA5_MARKER = "era5"  # substring identifying ERA5 members in a concat entry


def source_config_to_run_name(config_filename: str) -> str:
    # The version tag (-v1 / -v2) is already part of the source config stem.
    stem = pathlib.Path(config_filename).stem
    suffix = stem.removeprefix(CONFIG_PREFIX)
    return f"{WANDB_PREFIX}{suffix}"


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


def _train_dataset_identities(cfg: dict) -> set[str]:
    """Distinct training datasets in the train_loader concat, keyed by data_path.

    A dataset is split across multiple concat members by time subset (or by
    ensemble member/file_pattern), so members are grouped by data_path: two
    members sharing a data_path are one dataset. Used to distinguish a
    single-dataset run (e.g. C96/SHiELD) from a multi-dataset FM mixture.
    """
    dataset = cfg["train_loader"]["dataset"]
    if "concat" not in dataset:
        raise ValueError(
            "Expected train_loader.dataset.concat; cannot determine "
            "cooldown target dataset."
        )
    return {entry.get("data_path", "") for entry in dataset["concat"]}


def _restrict_train_loader_for_cooldown(cfg: dict) -> None:
    """Restrict the train_loader concat to the cooldown target dataset.

    Determines the cooldown target from the training datasets:

    - Single dataset (all concat members share one data_path, e.g. a pure
      C96/SHiELD run): cool down onto that dataset; keep all its members.
    - Multiple datasets including ERA5 (the FM mixtures): cool down onto ERA5
      alone; drop the non-ERA5 members.
    - Multiple datasets without ERA5: the cooldown target is ambiguous, so
      raise rather than guess.

    In all cases the now-stale multi-group ``group_weights`` is dropped.
    """
    dataset = cfg["train_loader"]["dataset"]
    identities = _train_dataset_identities(cfg)
    if len(identities) > 1:
        era5_members = [
            entry
            for entry in dataset["concat"]
            if ERA5_MARKER in entry.get("file_pattern", "")
        ]
        if not era5_members:
            raise ValueError(
                "Cooldown target is ambiguous: train_loader mixes multiple "
                f"datasets {sorted(identities)} but none is ERA5. Cannot infer "
                "which dataset to cool down onto."
            )
        dataset["concat"] = era5_members
    # Single-dataset runs keep all members; multi-dataset runs are now ERA5-only.
    # Either way group_weights partitioned the original concat and is now stale.
    cfg["train_loader"].pop("group_weights", None)


def _restrict_inference_for_cooldown(cfg: dict, single_dataset: bool) -> None:
    """Restrict inline-inference entries to the cooldown target's coordinate.

    The cooldown stepper's vertical coordinate comes from its (restricted)
    train_loader. For a multi-dataset FM run cooled onto ERA5, inference entries
    on other datasets (e.g. SHiELD, a different hybrid sigma-pressure coordinate)
    would run against a mismatched coordinate, so only ERA5 entries are kept.

    A single-dataset run keeps its training coordinate unchanged, so all of its
    inline-inference entries remain coordinate-consistent and are kept as-is.
    """
    if single_dataset:
        return
    entries = cfg.get("inference")
    if entries is None:
        return
    if not isinstance(entries, list):
        entries = [entries]
    cfg["inference"] = [
        entry
        for entry in entries
        if ERA5_MARKER in entry["loader"]["dataset"].get("file_pattern", "")
    ]


def _clear_inference_epochs(cfg: dict) -> None:
    """Run every inline-inference entry on every cooldown epoch.

    The FM configs schedule inference with an ``epochs`` slice sized for the
    ~150-epoch training run (e.g. ``{start: 10, step: 10}``). That slice indexes
    into the epoch list ``[1..max_epochs]``, so on the 8-epoch cooldown it can
    select no epochs at all (``[1..8][10::10]`` is empty), meaning inference
    never runs and no inference metrics are logged. Drop the field so each entry
    falls back to the default (every epoch), guaranteeing the same inference
    metrics the base config logs are produced during the cooldown.
    """
    entries = cfg.get("inference")
    if entries is None:
        return
    if not isinstance(entries, list):
        entries = [entries]
    for entry in entries:
        entry.pop("epochs", None)


def _fetch_wandb_run_names() -> set[str]:
    import wandb  # lazy import: only needed with --delete-if-in-wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return {run.name for run in runs}


def _out_path_to_run_name(out_path: pathlib.Path) -> str:
    """Reconstruct the wandb run name for a generated cooldown config.

    Config stem: {CONFIG_PREFIX}{base}{cooldown_suffix}
    Run name:    {WANDB_PREFIX}{base}{cooldown_suffix}
    (the version tag -v1 / -v2 is already part of {base}).
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
    out_path = HERE / f"{source_path.stem}{suffix}.yaml"

    beaker_dataset_id = None
    if source_map is not None:
        source_run_name = source_config_to_run_name(source_path.name)
        beaker_dataset_id = source_map.get(source_run_name)
        if beaker_dataset_id is None:
            # No training result dataset recorded means the training run has not
            # finished, so its cooldown checkpoint does not exist yet and the
            # cooldown would fail. Skip, matching generate_eval_configs.py, and
            # remove any stale config left from an earlier finished state.
            if out_path.exists():
                out_path.unlink()
                print(
                    f"Deleted {out_path.name} (no dataset ID for {source_run_name!r})"
                )
            else:
                print(
                    f"Skipped {out_path.name} (no dataset ID for {source_run_name!r})"
                )
            return

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

    # Cool down onto the training target: the single training dataset for a
    # single-dataset run, or ERA5 for a multi-dataset FM mixture.
    single_dataset = len(_train_dataset_identities(cfg)) <= 1
    _restrict_train_loader_for_cooldown(cfg)

    # Drop inline-inference entries whose coordinate won't match the cooldown
    # stepper. For a single-dataset run the coordinate is unchanged, so all
    # entries are kept; a multi-dataset ERA5 cooldown keeps ERA5 entries only.
    _restrict_inference_for_cooldown(cfg, single_dataset)

    # The base epochs schedule is sized for the full run; on the 8-epoch cooldown
    # it may never fire. Run every inference entry on every cooldown epoch so the
    # base config's inference metrics are logged.
    _clear_inference_epochs(cfg)

    cfg["optimization"]["lr"] = lr
    cfg["optimization"]["scheduler"] = _build_scheduler(epochs)
    cfg["max_epochs"] = epochs

    _write_config(cfg, out_path, beaker_dataset_id, existing_only, wandb_run_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
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
        for p in HERE.glob("*.yaml")
        if p.name.startswith(CONFIG_PREFIX)
        and "nc-sfno" in p.name
        and stem_matches_version(p.stem, args.version)
        and not p.name.endswith("-finetune.yaml")
        and not p.name.endswith("-cooldown.yaml")
        and not p.name.endswith("-bestinfcooldown.yaml")
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
