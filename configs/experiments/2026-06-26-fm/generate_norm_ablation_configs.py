"""Generate training configs for the per-dataset normalization ablation.

The ablation asks whether per-group normalization statistics improve ERA5
performance when a model is trained jointly on ERA5 and C96 SHiELD data. The
hypothesis is one of feature-space alignment: pooled statistics leave
`specific_total_water_0` with disjoint marginals between the two sources, and
per-group standardization is exactly the operation that closes a scalar mean
and scale offset like that.

Note this is *not* a test of whether the model can tell the sources apart.
`global_mean_co2` stays pooled (see PINNED_VARIABLES) and remains a perfect
discriminator in every arm.

Three grouping strategies (arms) are crossed with three data regimes and with
module conditioning off/on:

    A1  shared      one group over all data (the control)
    A2  per-source  c96 / era5
    A3  per-config  amip / ramped / som / era5

Conditioning is a separate use of the same labels: with `conditional: true` the
module additionally consumes them through its adaLN/CLN layers, giving the
model an explicit source signal rather than only an aligned input space. Both
uses are independent, hence the cross.

Cells that would train a model identical to a cheaper one are skipped; see
degenerate_reason. That leaves 11 configs per architecture, 22 in total:

    c96 regime   A1 (== A2), A3, each off/on            -> 4
    era5 regime  A1 (== A2 == A3), conditioning a no-op -> 1
    fm regime    A1, A2, A3, each off/on                -> 6

Each config is composed from two base configs: the regime source supplies the
datasets, validation and inference entries, and the architecture source
supplies the module builder and the input ordering it was trained with. For
the two cells whose regime and architecture come from the same base config,
that composition is a no-op. A handful of top-level logging and checkpointing
settings which the swin base omits are then applied uniformly, so all 22 runs
report the same metrics.

Every config carries dataset labels, including the A1 controls, so that the
cells differ only in the two axes under test.

Usage:
    python generate_norm_ablation_configs.py [--include-degenerate]
"""

import argparse
import copy
import pathlib
from typing import Any

import yaml

HERE = pathlib.Path(__file__).parent
BASE_CONFIGS_DIR = HERE / "base_configs"
RUN_CONFIGS_DIR = HERE / "run_configs"

CONFIG_PREFIX = "ace-train-config-4deg-AIMIP-"

# Root of the statistics written by
# scripts/data_process/get_pooled_stats.py from the norm-ablation-*-stats.yaml
# configs. Each regime has its own subdirectory because the regimes' store
# lists do not nest: the `era5` group of the fm regime covers different time
# windows than the era5 regime's own data.
#
# The stats jobs write to gs://vcm-ml-intermediate/alexeyy/norm_ablation_0/
# first, so the per-member subdirectories are reachable from the analysis
# notebooks, and are then copied to weka with scripts/data_process/gcs_to_weka.sh
# before training. This is the weka side of that copy, which is what the
# generated training configs read.
STATS_ROOT = "/climate-default/alexeyy/norm_ablation_0"

# Variables which always use the pooled constants, even in the per-group arms.
#
#   global_mean_co2   Near-constant within a group, so a per-group std would be
#                     ~0 and the normalized input would explode. Splitting it
#                     would also put ERA5 and C96 in disjoint CO2 input spaces,
#                     which is the opposite of the alignment being tested.
#   HGTsfc            Static: the same field every sample. Its spatial
#                     fingerprint survives any scalar normalization, so a
#                     per-group split buys nothing.
#   land_fraction     Static, same reasoning.
#   ocean_fraction    Static, same reasoning.
#   sea_ice_fraction  Not static, but the between-source gap is well inside
#                     ERA5's own temporal variance; there is nothing to align.
#   DSWRFtoa          Pure orbital geometry, identical between sources by
#                     construction.
PINNED_VARIABLES = [
    "global_mean_co2",
    "HGTsfc",
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "DSWRFtoa",
]

# Labels attached to each dataset, keyed by a substring of the member's zarr
# path. These are the finest-grained groups; the coarser A2 groups are built
# from them by the normalization config rather than by relabelling the data,
# so the data blocks are identical across all arms of a regime.
LABEL_MARKERS = [
    ("era5", "era5"),
    ("shield-amip-ensemble", "amip"),
    # The constant-CO2 AMIP stream is a separate zarr used only by the
    # long_43year_constant_co2 inference entry; it is AMIP data and shares
    # AMIP's normalization constants.
    ("shield-amip-constant-co2", "amip"),
    ("ramped-climSST-random-CO2", "ramped"),
    ("shield-som-ensemble", "som"),
]

# Arms: group name -> the labels it contains.
ARMS = {
    "a1": {"all": ["amip", "ramped", "som", "era5"]},
    "a2": {"c96": ["amip", "ramped", "som"], "era5": ["era5"]},
    "a3": {
        "amip": ["amip"],
        "ramped": ["ramped"],
        "som": ["som"],
        "era5": ["era5"],
    },
}

# Regime -> the base config supplying datasets, validation and inference.
REGIME_SOURCES = {
    "c96": "ace-train-config-4deg-AIMIP-nc-sfno-c96-v3.yaml",
    "era5": "ace-train-config-4deg-AIMIP-nc-sfno-v2.yaml",
    "fm": "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-random-v1.yaml",
}

# Regime -> the labels its data actually contains. Used to detect degenerate
# arms and to pick each config's fallback group.
REGIME_LABELS = {
    "c96": ["amip", "ramped", "som"],
    "era5": ["era5"],
    "fm": ["amip", "ramped", "som", "era5"],
}

# Architecture -> the base config supplying the module builder.
ARCH_SOURCES = {
    "nc-sfno": "ace-train-config-4deg-AIMIP-nc-sfno-v2.yaml",
    "nc-swin-v2": "ace-train-config-4deg-AIMIP-nc-swin-v2-fm-random-v1.yaml",
}

# Keys copied from the architecture source into the generated config. Anything
# not listed here comes from the regime source.
#
# in_names is included because the two architectures order their inputs
# differently (the swin config puts global_mean_co2 last), and that ordering is
# baked into a checkpoint's channel layout.
ARCH_STEP_CONFIG_KEYS = ["builder", "residual_prediction", "in_names"]

# Top-level settings the swin base config omits but the sfno ones set. Applied
# to every generated config so all 22 log and checkpoint identically.
SHARED_TOP_LEVEL = {
    "train_aggregator": {"ensemble_metrics": True},
    "ema_checkpoint_save_epochs": {"start": 5, "step": 5},
}


def load_base(filename: str) -> dict:
    with open(BASE_CONFIGS_DIR / filename) as f:
        return yaml.safe_load(f)


def label_for_member(member: dict) -> str:
    """Determine which dataset a train_loader concat member belongs to."""
    path = f"{member.get('data_path', '')}/{member.get('file_pattern', '')}"
    for marker, label in LABEL_MARKERS:
        if marker in path:
            return label
    raise ValueError(f"Could not determine a dataset label for member: {member}")


def add_labels_to_dataset(dataset: dict) -> set[str]:
    """Label every member of a dataset config in place; return the labels used.

    All members of a loader must either set labels or leave them unset, so this
    is applied to every loader in a config, not only the training one.
    """
    labels: set[str] = set()
    if "concat" in dataset:
        members = dataset["concat"]
    elif "merge" in dataset:
        members = dataset["merge"]
    else:
        members = [dataset]
    for member in members:
        label = label_for_member(member)
        member["labels"] = [label]
        labels.add(label)
    return labels


def add_labels(config: dict) -> set[str]:
    """Attach dataset labels to every loader in a training config.

    The train, validation and inference loaders must agree on whether labels
    are in use, so a config either labels all of them or none.
    """
    labels = add_labels_to_dataset(config["train_loader"]["dataset"])
    for validation in config.get("validation_list", [config.get("validation")]):
        if validation is not None:
            add_labels_to_dataset(validation["loader"]["dataset"])
    for inference in config.get("inference", []):
        add_labels_to_dataset(inference["loader"]["dataset"])
    return labels


def build_normalization(
    regime: str, arm: str, groups: dict[str, list[str]]
) -> dict[str, Any]:
    """Build the normalization block for one cell.

    The pooled (`network` and `residual`) constants are unchanged from the
    control in every arm. The `grouped` block layers per-group constants on top
    of them for the network's inputs and outputs only, which leaves the loss,
    global mean removal, spatial masking and the aggregators' normalized
    metrics on the pooled scale so those stay comparable across arms.
    """
    pooled = f"{STATS_ROOT}/{regime}"
    normalization: dict[str, Any] = {
        "network": {
            "global_means_path": f"{pooled}/centering.nc",
            "global_stds_path": f"{pooled}/scaling-full-field.nc",
        },
        "residual": {
            "global_means_path": f"{pooled}/centering.nc",
            "global_stds_path": f"{pooled}/scaling-residual.nc",
        },
    }
    if arm == "a1":
        return normalization
    normalization["grouped"] = {
        "groups": {
            group_name: {
                "labels": labels,
                "normalization": _group_stats_paths(pooled, regime, group_name, labels),
            }
            for group_name, labels in groups.items()
        },
        # A batch with no labels (e.g. standalone inference on an unlabeled
        # dataset) falls back to this group. Named explicitly because an
        # implicit choice would silently normalize against the wrong
        # distribution.
        "default_group": _default_group(regime, groups),
        "pinned_variables": list(PINNED_VARIABLES),
    }
    return normalization


def _group_stats_paths(
    pooled: str, regime: str, group_name: str, labels: list[str]
) -> dict[str, str]:
    """Stats paths for one normalization group.

    A group covering every label in the regime is pooled over exactly the stores
    the root pooled stats cover, so it reads those directly. The stats configs
    write no `groups/{name}/` directory in that case — there would be nothing to
    distinguish it from the root — and pointing at one would fail on a missing
    netCDF. Only the degenerate single-group arms reach this branch; every arm
    the generator writes by default splits the regime into at least two groups.
    """
    if set(labels) == set(REGIME_LABELS[regime]):
        root = pooled
    else:
        root = f"{pooled}/groups/{group_name}"
    return {
        "global_means_path": f"{root}/centering.nc",
        "global_stds_path": f"{root}/scaling-full-field.nc",
    }


def _default_group(regime: str, groups: dict[str, list[str]]) -> str:
    """Pick the fallback group for unlabeled batches.

    ERA5 is the evaluation target, so prefer the group holding it; a regime
    without ERA5 falls back to its first group by name.
    """
    for group_name, labels in sorted(groups.items()):
        if "era5" in labels and "era5" in REGIME_LABELS[regime]:
            return group_name
    return sorted(groups)[0]


def groups_for_cell(regime: str, arm: str) -> dict[str, list[str]]:
    """The arm's groups, restricted to the labels the regime actually has."""
    present = set(REGIME_LABELS[regime])
    groups = {}
    for group_name, labels in ARMS[arm].items():
        restricted = [label for label in labels if label in present]
        if restricted:
            groups[group_name] = restricted
    return groups


def all_cells() -> list[tuple[str, str, str, bool]]:
    """Every (arch, regime, arm, conditional) cell, in a stable order."""
    return [
        (arch, regime, arm, conditional)
        for arch in ARCH_SOURCES
        for regime in REGIME_SOURCES
        for arm in ARMS
        for conditional in (False, True)
    ]


def degenerate_reason(regime: str, arm: str, conditional: bool) -> str | None:
    """Why this cell trains a model identical to a cheaper one, if it does.

    Two independent collapses, both from a regime having too few labels for the
    axis to vary anything:

    - An arm with a single group has nothing to select per sample, so it is the
      A1 control.
    - Conditioning on a single label feeds every sample the same constant
      one-hot, and the resulting constant scale and shift are absorbed by the
      normalization layers' own affine parameters.
    """
    if arm != "a1" and len(groups_for_cell(regime, arm)) < 2:
        return "reduces to the A1 control for this regime"
    if conditional and len(REGIME_LABELS[regime]) < 2:
        return "conditioning on a single label is a no-op for this regime"
    return None


def build_config(arch: str, regime: str, arm: str, conditional: bool) -> dict:
    config = copy.deepcopy(load_base(REGIME_SOURCES[regime]))
    arch_base = load_base(ARCH_SOURCES[arch])

    # Architecture keys override the regime source's own.
    step_config = config["stepper"]["step"]["config"]
    arch_step_config = arch_base["stepper"]["step"]["config"]
    for key in ARCH_STEP_CONFIG_KEYS:
        step_config[key] = copy.deepcopy(arch_step_config[key])
    for key, value in SHARED_TOP_LEVEL.items():
        config[key] = copy.deepcopy(value)

    labels = add_labels(config)
    expected = set(REGIME_LABELS[regime])
    if labels != expected:
        raise ValueError(
            f"{arch}/{regime}: train_loader resolved to labels {sorted(labels)}, "
            f"expected {sorted(expected)}. REGIME_LABELS is out of date with "
            f"{REGIME_SOURCES[regime]}."
        )

    step_config["normalization"] = build_normalization(
        regime, arm, groups_for_cell(regime, arm)
    )
    # Labels always reach the normalizer; `conditional` controls only whether
    # they additionally drive the module's adaLN/CLN conditioning.
    if conditional:
        step_config["builder"]["conditional"] = True
    return config


def config_name(arch: str, regime: str, arm: str, conditional: bool) -> str:
    suffix = "-cond" if conditional else ""
    return f"{CONFIG_PREFIX}{arch}-{regime}-{arm}{suffix}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help=(
            "Also write the arms which reduce to the A1 control for their "
            "regime. Useful only as a seed-variance estimate, and only if the "
            "seeds are then changed."
        ),
    )
    args = parser.parse_args()

    RUN_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    for arch, regime, arm, conditional in all_cells():
        name = config_name(arch, regime, arm, conditional)
        reason = degenerate_reason(regime, arm, conditional)
        if reason is not None and not args.include_degenerate:
            skipped.append((name, reason))
            continue
        config = build_config(arch, regime, arm, conditional)
        out_path = RUN_CONFIGS_DIR / name
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        written.append(name)
        print(f"Wrote {out_path}")

    for name, reason in skipped:
        print(f"Skipped {name}: {reason}")
    print(f"{len(written)} configs written, {len(skipped)} degenerate cells skipped")


if __name__ == "__main__":
    main()
