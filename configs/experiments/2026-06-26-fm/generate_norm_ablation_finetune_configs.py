"""Generate ERA5 fine-tuning configs for the per-dataset normalization ablation.

Two regimes are fine-tuned onto ERA5, and they ask different questions.

**fm** -- *specialization.* The `fm` cells train jointly on ERA5 and C96 SHiELD
under three normalization arms crossed with module conditioning. Every source
model has already seen ERA5 in the mixture, so what is asked is whether the arm
a model was pretrained under leaves it better positioned to be specialized onto
ERA5. The fine-tuning phase itself differentiates nothing -- with only `era5` in
the train_loader, A2 and A3 both bind their `era5` group (the same
`groups/era5/` constants) and the conditional cells see a constant one-hot -- so
a difference between cells is entirely a difference between the weights they
start from.

**c96** -- *transfer.* The `c96` cells never saw ERA5 at all; the ablation's own
figures log NaN for every ERA5 inference entry on those runs. Fine-tuning them
asks how far C96-only pretraining under A1 vs A3 carries to ERA5, and whether
the arm changes the adaptation trajectory. `evaluate_before_training` makes the
epoch-0 point a genuine zero-shot ERA5 measurement for a model that has never
seen the data.

The c96 regime has no `era5` label and no `era5` normalization group -- its
vocabulary is `{amip, ramped, som}` and `GroupedNormalizer` raises on a label
with no group. Rather than growing the vocabulary, **the ERA5 data is labeled
`amip`** (see C96_ERA5_ALIAS). This keeps three properties which growing it
would cost:

- The normalization block is untouched, so ERA5 samples are normalized by the
  constants the model was pretrained with. That is the same continuity rule the
  fm cells rely on, applied to the only group c96 has for the job.
- `overwrite_weights` matches label-dependent weights positionally against
  *sorted* labels. A fourth label named `era5` would sort to
  `[amip, era5, ramped, som]` and silently copy the checkpoint's `ramped`
  column into `era5` and `som` into `ramped`.
- A1 and A3 then differ only in how they were *pretrained*, which is the thing
  under test. Giving A3 an ERA5 group while A1 (which has no `grouped` block)
  stayed on the pooled constants would confound the fine-tune input scale with
  the pretraining difference.

The cost is recorded in per_dataset_norm_plan.md: ERA5's
`specific_total_water_0` lands ~5 sigma off the `amip` mean it is normalized
against, and the conditional cells are told the data is AMIP.

Three things about the transformation are load-bearing:

- **The normalization block is copied verbatim.** `parameter_init` loads module
  weights only and the normalizer is rebuilt from this YAML with nothing
  checking the two agree, so re-deriving statistics from the ERA5-only data
  would silently retrain on a shifted input space. It would also collapse the
  experiment: all three arms would then read one identical set of constants.
- **The label vocabulary is taken from the checkpoint**
  (`override_labels_from_weights`). An ERA5-only train_loader shrinks
  `DatasetInfo.all_labels` to a single label, which builds the conditional
  cells' label-dependent weights narrower than the checkpoint's and fails to
  load. It also matters for the unconditional cells, whose checkpoints would
  otherwise record a different vocabulary than their siblings'.
- **The vertical coordinate is taken from the checkpoint**
  (`override_vertical_coordinate_from_weights`), rather than being re-derived
  from the restricted train_loader, so it stays consistent from pretraining
  through fine-tuning and inline inference. Same reasoning as the cooldown
  generator.

Usage:
    python generate_norm_ablation_finetune_configs.py [--regime {fm,c96}]
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
    REGIME_SOURCES,
    config_name,
    degenerate_reason,
    label_for_member,
    load_base,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
WANDB_PREFIX = "ace2-fm-"

#: Only the sfno architecture is fine-tuned. The era5 cell is skipped in every
#: regime list: it is already an ERA5 specialist.
ARCH = "nc-sfno"

#: Regimes whose cells are fine-tuned onto ERA5, in submission order.
REGIMES = ("fm", "c96")

#: The regime whose base config supplies the ERA5 data. `fm` is the only base
#: holding both sources, so its ERA5 train members, validation and inference
#: entries are what the c96 configs splice in -- which also makes the c96
#: fine-tunes read exactly the same ERA5 data as the fm ones, so the two sets
#: differ only in the pretraining regime.
ERA5_SOURCE_REGIME = "fm"

#: The label the spliced ERA5 data carries in a c96 config. See the module
#: docstring: the c96 vocabulary has no `era5`, and `amip` is both the group
#: those configs already fall back to (`default_group`) and the C96 stream
#: closest to ERA5 physically -- prescribed observed SSTs over 1979-2008.
C96_ERA5_ALIAS = "amip"

#: The unmasked cells; the mask10 twins are a separate axis and are not
#: fine-tuned.
MASKING = ""

#: Filename inside the source run's Beaker result dataset. This is the
#: last-epoch checkpoint, the same file submit_eval_jobs.py evaluates as
#: `-lastepoch`, so the fine-tune departs from the weights the ablation's
#: existing figures describe. Deliberately not `-bestinf`: for a c96 cell that
#: is a checkpoint selected on C96 data, which would bias the transfer
#: measurement by an amount differing per cell.
CHECKPOINT_NAME = "training_checkpoints/ckpt.tar"

FINETUNE_SUFFIX = "-finetune"
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-5

#: The label identifying the ERA5 members of a loader, per LABEL_MARKERS.
#: Resolved from the member's zarr path, so it still identifies ERA5 loaders in
#: a c96 config after they have been relabeled to C96_ERA5_ALIAS.
ERA5_LABEL = "era5"


def source_cells(regime: str) -> list[tuple[str, bool]]:
    """The (arm, conditional) cells to fine-tune for a regime, in a stable order.

    Degenerate cells are skipped for the same reason the base generator skips
    them: no source run exists, because training one would have reproduced a
    cheaper cell. For `c96` that drops A2, which has no `era5` label to separate
    out and so reduces to the A1 control.
    """
    return [
        (arm, conditional)
        for arm in ARMS
        for conditional in (False, True)
        if degenerate_reason(regime, arm, conditional) is None
    ]


def source_config_name(regime: str, arm: str, conditional: bool) -> str:
    return config_name(ARCH, regime, arm, conditional, MASKING)


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
    silently treated as non-ERA5. Note this reads the member's *path*, not its
    `labels`, so it still holds for a c96 config's relabeled ERA5 loaders.
    """
    labels = {label_for_member(member) for member in _dataset_members(dataset)}
    return labels == {ERA5_LABEL}


def _set_labels(dataset: Any, label: str) -> None:
    """Force every member of a loader's dataset to carry exactly `label`.

    Used to relabel spliced ERA5 data as C96_ERA5_ALIAS. Deliberately not
    reusing the base generator's `add_labels_to_dataset`, which resolves the
    label from the path and would write `era5`.
    """
    for member in _dataset_members(dataset):
        member["labels"] = [label]


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


def _replace_train_loader_with_era5(cfg: dict, era5_source: dict, label: str) -> None:
    """Swap a C96-only train_loader's data for the ERA5 members, relabeled.

    Only the `dataset` is replaced; the loader's own settings (batch size,
    workers, time buffers) stay as the source config had them, which is the
    same rule the fm path follows by leaving them untouched.
    """
    dataset = copy.deepcopy(era5_source["train_loader"]["dataset"])
    if not _is_era5(dataset):
        raise ValueError(
            f"{ERA5_SOURCE_REGIME} train_loader is not ERA5-only after "
            "restriction; cannot splice it into a c96 config."
        )
    _set_labels(dataset, label)
    cfg["train_loader"]["dataset"] = dataset
    cfg["train_loader"].pop("group_weights", None)


def _replace_validation_with_era5(cfg: dict, era5_source: dict, label: str) -> None:
    """Swap the validation data for the ERA5 source's, relabeled."""
    dataset = copy.deepcopy(era5_source["validation"]["loader"]["dataset"])
    _set_labels(dataset, label)
    cfg["validation"]["loader"]["dataset"] = dataset


def _splice_era5_inference(cfg: dict, era5_source: dict, label: str) -> None:
    """Prepend the ERA5 source's ERA5 inference entries, relabeled.

    A c96 config's suite is entirely C96, so it has nothing to select a
    checkpoint on once training moves to ERA5. The five ERA5 entries are taken
    verbatim rather than rebuilt: their aggregator variable lists already omit
    `total_water_path`, the one SHiELD diagnostic the c96 entries carry and the
    ERA5 store does not.

    They are prepended so the resulting 13-entry suite is in the same order as
    the fm configs', which makes the two sets diffable side by side.
    """
    existing = {entry["name"] for entry in cfg["inference"]}
    era5_entries = []
    for entry in era5_source["inference"]:
        if not _is_era5(entry["loader"]["dataset"]):
            continue
        if entry["name"] in existing:
            raise ValueError(
                f"ERA5 inference entry {entry['name']!r} collides with an "
                "entry already in the source config."
            )
        entry = copy.deepcopy(entry)
        _set_labels(entry["loader"]["dataset"], label)
        era5_entries.append(entry)
    if not era5_entries:
        raise ValueError(
            f"No ERA5 inference entries found in the {ERA5_SOURCE_REGIME} base."
        )
    cfg["inference"] = era5_entries + cfg["inference"]


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

    The source configs weight the C96 `10year_insample_ensemble_varying_co2`
    entry at 1.0 (and, in the fm regime, the ERA5 `10year` entry too), so
    best_inference_ckpt would be selected on data the fine-tune has just
    stopped training on. The C96 entries stay in the suite as weight-0
    forgetting diagnostics.
    """
    for entry in cfg["inference"]:
        if entry.get("weight", 1.0) > 0.0 and not _is_era5(entry["loader"]["dataset"]):
            entry["weight"] = 0.0


def _apply_era5_data(cfg: dict, regime: str) -> None:
    """Point every loader in the config at ERA5.

    The fm regime already holds ERA5 and only needs the C96 members removed
    from its train_loader; its validation and inference suite are unchanged.
    The c96 regime holds none, so its train and validation data are replaced
    and the ERA5 inference entries are spliced in, all relabeled to
    C96_ERA5_ALIAS.
    """
    if regime == ERA5_SOURCE_REGIME:
        _restrict_train_loader_to_era5(cfg)
        return
    era5_source = load_base(REGIME_SOURCES[ERA5_SOURCE_REGIME])
    _restrict_train_loader_to_era5(era5_source)
    _replace_train_loader_with_era5(cfg, era5_source, C96_ERA5_ALIAS)
    _replace_validation_with_era5(cfg, era5_source, C96_ERA5_ALIAS)
    _splice_era5_inference(cfg, era5_source, C96_ERA5_ALIAS)


def build_config(
    source_cfg: dict, regime: str, checkpoint_path: str, epochs: int, lr: float
) -> dict:
    cfg = copy.deepcopy(source_cfg)

    cfg["stepper_training"]["parameter_init"] = {
        "weights_path": checkpoint_path,
        "override_labels_from_weights": True,
        "override_vertical_coordinate_from_weights": True,
    }

    _apply_era5_data(cfg, regime)
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
    # aggregators rather than pulled from the source run. For the c96 cells
    # this is the zero-shot ERA5 number, which those runs never produced.
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
        "--regime",
        choices=REGIMES,
        help="Only this regime's cells (default: all of them).",
    )
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

    regimes = [args.regime] if args.regime else list(REGIMES)
    for regime in regimes:
        for arm, conditional in source_cells(regime):
            source_name = source_config_name(regime, arm, conditional)
            source_path = RUN_CONFIGS_DIR / source_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"{source_name} not found — run "
                    "generate_norm_ablation_configs.py first"
                )
            source_run_name = config_to_run_name(source_name)
            beaker_dataset_id = source_map.get(source_run_name)
            if beaker_dataset_id is None:
                # No result dataset recorded means the source run has not
                # finished, so its last-epoch checkpoint does not exist yet.
                raise ValueError(
                    f"No Beaker dataset ID for {source_run_name!r} in "
                    f"{args.source_map} — has the source run finished? "
                    "Refresh with update_beaker_map.py."
                )
            with source_path.open() as f:
                source_cfg = yaml.safe_load(f)
            cfg = build_config(
                source_cfg,
                regime,
                f"/checkpoints/{CHECKPOINT_NAME}",
                args.epochs,
                args.lr,
            )
            out_path = RUN_CONFIGS_DIR / f"{source_path.stem}{FINETUNE_SUFFIX}.yaml"
            _write_config(cfg, out_path, beaker_dataset_id)


if __name__ == "__main__":
    main()
