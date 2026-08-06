"""Generate fixed-variable evaluator suite configs for the FM training runs.

Sibling to generate_orography_configs.py: the same experiment, but instead of
swapping which grid supplies ``HGTsfc`` it holds a single input variable fixed
at a precomputed time mean for the whole rollout. One eval suite is generated
per variable in the run's ``in_names``, so a run's suite set answers "how much
does this model rely on each of its inputs varying?".

Each inference entry gets:

  - ``loader.dataset.constant_field_override`` pointing at a time-mean store,
    so the dataset serves that variable's time mean unchanged at every
    timestep, and
  - ``stepper_override.prescribed_prognostic_names: [name]`` when the variable
    is prognostic, so the stepper overwrites its own prediction with that field
    on every step instead of evolving it freely.

Forcing-only variables (in ``in_names`` but not ``out_names``) get no stepper
override: the model never predicts them, so the dataset's value is all it ever
sees, and ``prescribed_prognostic_names`` rejects names outside ``out_names``.

Two variants of every suite are generated, differing only in *which* dataset's
time mean is served (``--variant`` selects which are written):

``own``
    The time mean of the entry's own dataset. The served field is the right
    field for that dataset, only stripped of its variability, so a metric shift
    measures the loss of variability alone.

``swapped``
    The time mean of the *other* dataset -- ERA5 entries are served the C96
    mean and both C96 entries (AMIP ensemble and constant-CO2) are served the
    ERA5 mean. This deliberately confounds the loss of variability with a
    change of grid and climatology, the way the orography grid swap does; the
    ``own`` suite is the baseline that confound is read against.

Both variants set ``prescribed_prognostic_names`` identically, so a swapped
suite differs from its own-dataset counterpart by exactly one path: a
prognostic variable is pinned to the foreign climatology for the whole rollout
while its metrics are still scored against its own dataset's target.

``land_fraction``, ``ocean_fraction`` and ``HGTsfc`` are already time-invariant
in these datasets, so under ``own`` holding them at their time mean is a
numerical no-op; their suites act as null controls there, where any metric
shift is pipeline noise rather than a real effect. Under ``swapped`` they are
real perturbations, since the two grids genuinely disagree about them -- the
swapped ``HGTsfc`` suite reaches the orography experiment's grid swap by a
different route and cross-checks it.

Generated for the FM (multi-dataset) runs only; the single-dataset runs are not
part of this experiment.
"""

import argparse
import copy
import json
import pathlib

import yaml
from _version_select import add_version_arg
from generate_eval_configs import (
    BASE_CONFIGS_DIR,
    CONFIG_PREFIX,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SOURCE_MAP,
    RUN_CONFIGS_DIR,
    WANDB_PREFIX,
    _build_eval_suite_config,
    _fetch_wandb_finished_summaries,
    _fetch_wandb_run_names,
    _write_config,
    discover_source_configs,
    eval_suite_config_to_run_name,
    source_config_to_run_name,
)
from generate_eval_configs import EVAL_SUITE_CONFIG_PREFIX as _EVAL_SUITE_CONFIG_PREFIX

FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX = f"{_EVAL_SUITE_CONFIG_PREFIX}fixed-"

# Fixed-variable evals only ever run the best-inference checkpoint (not
# besttrain/lastepoch) -- submit_fixed_var_jobs.py sources it directly from the
# corresponding unmodified run's own result dataset, so there is no separate
# "fixed-variable training run" or dataset to record.
FIXED_VAR_CHECKPOINT_SUFFIXES = ("-bestinf",)

# Suite variants, distinguished by whose time mean each inference entry is
# served (see the module docstring). The variant is part of the suite filename
# for every variant but `own`, whose names predate this distinction.
VARIANTS = ("own", "swapped")
DEFAULT_VARIANT = "both"
VARIANT_FILENAME_PARTS = {"own": "", "swapped": "swapped-"}

# Precomputed time-mean store for each inference dataset, on the same weka
# mount the training configs read their normalization statistics from. These
# are the per-source subdirectories (not `combined/`), matching the
# subdirectory each grid's training config already uses for centering/scaling.
# The C96 constant-CO2 dataset has no statistics of its own; it is the same
# model on the same grid as the AMIP ensemble, so it reuses the ic_0001 mean.
_ERA5_TIME_MEAN = (
    "/climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/"
    "2026-03-19-era5-4deg-8layer-1940-2025/time-mean.nc"
)
_C96_TIME_MEAN = (
    "/climate-default/"
    "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-"
    "dataset-stats/ic_0001/time-mean.nc"
)

# Time-mean store per variant for each inference dataset, keyed by the
# (data_path, file_pattern) pair the inference entries use. Holding both
# variants in one table means a new inference dataset cannot be taught to one
# variant and forgotten by the other; `swapped` is the mean of the dataset on
# the other grid, which for both C96 datasets is ERA5's.
TIME_MEAN_PATHS = {
    (
        "/climate-default",
        "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
    ): {"own": _ERA5_TIME_MEAN, "swapped": _C96_TIME_MEAN},
    (
        "/climate-default/2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-"
        "ensemble-dataset",
        "ic_0001.zarr",
    ): {"own": _C96_TIME_MEAN, "swapped": _ERA5_TIME_MEAN},
    (
        "/climate-default/2026-07-01-vertically-resolved-c96-4deg-daily-shield-amip-"
        "constant-co2-dataset",
        "AMIP-constant-CO2.zarr",
    ): {"own": _C96_TIME_MEAN, "swapped": _ERA5_TIME_MEAN},
}


def _is_fm_run(run_name: str) -> bool:
    """True for the multi-dataset (foundation model) training runs."""
    return run_name.removeprefix(WANDB_PREFIX).startswith("nc-sfno-fm")


def _time_mean_path(dataset: dict, variant: str) -> str:
    """Return the `variant` time-mean store for `dataset`'s source."""
    if "data_path" not in dataset:
        raise ValueError(
            f"Dataset {dataset!r} is not a plain data_path-keyed dict; "
            "concat/merge inference datasets are not supported by this script."
        )
    key = (dataset["data_path"], dataset["file_pattern"])
    if key not in TIME_MEAN_PATHS:
        raise ValueError(
            f"No time-mean store recorded for inference dataset {key!r}; "
            f"known datasets: {sorted(TIME_MEAN_PATHS)}."
        )
    return TIME_MEAN_PATHS[key][variant]


def _apply_fixed_var_override(dataset: dict, name: str, variant: str) -> dict:
    """Return `dataset` with `name` served from the `variant` time mean."""
    return {
        **copy.deepcopy(dataset),
        "constant_field_override": {
            "path": _time_mean_path(dataset, variant),
            "names": [name],
        },
    }


def _fixed_var_names(
    train_cfg: dict, requested: list[str] | None, source_run_name: str
) -> list[str]:
    """Input variable names to generate suites for, in `in_names` order."""
    in_names = train_cfg["stepper"]["step"]["config"]["in_names"]
    if requested is None:
        return list(in_names)
    missing = sorted(set(requested) - set(in_names))
    if missing:
        raise ValueError(
            f"Variables {missing} are not inputs of {source_run_name}; "
            f"its in_names are {in_names}."
        )
    return [name for name in in_names if name in set(requested)]


def source_config_to_fixed_var_eval_suite_config(
    config_filename: str, name: str, variant: str = "own"
) -> str:
    suffix = pathlib.Path(config_filename).stem.removeprefix(CONFIG_PREFIX)
    variant_part = VARIANT_FILENAME_PARTS[variant]
    return f"{FIXED_VAR_EVAL_SUITE_CONFIG_PREFIX}{variant_part}{name}-{suffix}.yaml"


def generate_fixed_var_eval_configs(
    source_path: pathlib.Path,
    source_map: dict[str, str],
    variables: list[str] | None,
    variants: tuple[str, ...],
    inference_names: list[str] | None,
    checkpoint_path: str,
    existing_only: bool,
    wandb_run_names: set[str] | None = None,
    wandb_finished_summaries: dict[str, list[set[str]]] | None = None,
) -> None:
    source_run_name = source_config_to_run_name(source_path.name)
    source_dataset_id = source_map.get(source_run_name)
    if source_dataset_id is None:
        # No training result dataset recorded for this run yet (e.g. a config
        # not present in the source map). Skip rather than halt the whole run.
        print(f"Skipped {source_path.name} (no dataset ID for {source_run_name!r})")
        return

    with source_path.open() as f:
        train_cfg = yaml.safe_load(f)

    if not _is_fm_run(source_run_name):
        print(f"Skipped {source_path.name} (not a multi-dataset FM run)")
        return

    out_names = set(train_cfg["stepper"]["step"]["config"]["out_names"])

    for name in _fixed_var_names(train_cfg, variables, source_run_name):
        for variant in variants:
            cfg = _build_eval_suite_config(
                train_cfg=train_cfg,
                inference_names=inference_names,
                checkpoint_path=checkpoint_path,
            )
            for entry in cfg["inferences"]:
                entry_cfg = entry["config"]
                loader = entry_cfg["loader"]
                loader["dataset"] = _apply_fixed_var_override(
                    loader["dataset"], name, variant
                )
                if name in out_names:
                    entry_cfg["stepper_override"] = {
                        "prescribed_prognostic_names": [name]
                    }
            out_path = RUN_CONFIGS_DIR / source_config_to_fixed_var_eval_suite_config(
                source_path.name, name, variant
            )
            _write_config(
                cfg,
                out_path,
                source_run_name,
                source_dataset_id,
                existing_only,
                wandb_run_names,
                eval_run_name_base=eval_suite_config_to_run_name(out_path.name),
                checkpoint_suffixes=FIXED_VAR_CHECKPOINT_SUFFIXES,
                wandb_finished_summaries=wandb_finished_summaries,
            )


def select_source_configs(
    version: str | None, base_configs: list[str] | None
) -> list[pathlib.Path]:
    """Discovered source configs, optionally narrowed to named base configs.

    A base config may be named by filename, stem, or the run-name suffix that
    remains once CONFIG_PREFIX is stripped.
    """
    source_configs = discover_source_configs(version)
    if base_configs is None:
        return source_configs
    by_name: dict[str, pathlib.Path] = {}
    for path in source_configs:
        by_name[path.stem] = path
        by_name[path.stem.removeprefix(CONFIG_PREFIX)] = path
    selected = []
    for name in base_configs:
        stem = pathlib.Path(name).stem
        if stem not in by_name:
            raise ValueError(
                f"Base config {name!r} not found in {BASE_CONFIGS_DIR} among the "
                "configs selected by --version; available: "
                f"{sorted(p.stem for p in source_configs)}."
            )
        selected.append(by_name[stem])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_version_arg(parser)
    parser.add_argument(
        "--base-config",
        nargs="+",
        default=None,
        help=(
            "Training config(s) to generate suites for, named by filename, "
            "stem, or run-name suffix (default: all discovered configs)."
        ),
    )
    parser.add_argument(
        "--variable",
        nargs="+",
        default=None,
        help=(
            "Input variable name(s) to hold fixed, one suite each "
            "(default: every variable in the run's in_names)."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "both"),
        default=DEFAULT_VARIANT,
        help=(
            "Whose time mean each held variable is served from: 'own' for the "
            "entry's own dataset, 'swapped' for the dataset on the other grid "
            f"(default: {DEFAULT_VARIANT})."
        ),
    )
    parser.add_argument(
        "--inference-name",
        nargs="+",
        default=None,
        help="Inline inference entry name(s) to export (default: all entries).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the mounted checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--source-map",
        metavar="PATH",
        default=DEFAULT_SOURCE_MAP,
        help=(
            "JSON file mapping training run name → Beaker dataset ID"
            f" (default: {DEFAULT_SOURCE_MAP})."
        ),
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Only rewrite evaluator configs that already exist.",
    )
    parser.add_argument(
        "--delete-if-in-wandb",
        action="store_true",
        help=(
            "Delete/skip eval suites whose checkpoint runs all already exist "
            "in wandb."
        ),
    )
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Delete/skip eval suites whose checkpoint runs all finished in "
            "wandb with every inference entry logged. Stricter than "
            "--delete-if-in-wandb, which only checks that runs with the "
            "expected names exist."
        ),
    )
    args = parser.parse_args()

    with open(args.source_map) as f:
        source_map: dict[str, str] = json.load(f)

    wandb_run_names: set[str] | None = None
    if args.delete_if_in_wandb:
        print("Fetching run names from wandb...")
        wandb_run_names = _fetch_wandb_run_names()
        print(f"Found {len(wandb_run_names)} existing runs.")

    wandb_finished_summaries: dict[str, list[set[str]]] | None = None
    if args.skip_if_in_wandb:
        print("Fetching finished runs from wandb...")
        wandb_finished_summaries = _fetch_wandb_finished_summaries()
        print(f"Found {len(wandb_finished_summaries)} finished run names.")

    source_configs = select_source_configs(args.version, args.base_config)
    variants = VARIANTS if args.variant == "both" else (args.variant,)

    for source_path in source_configs:
        generate_fixed_var_eval_configs(
            source_path=source_path,
            source_map=source_map,
            variables=args.variable,
            variants=variants,
            inference_names=args.inference_name,
            checkpoint_path=args.checkpoint_path,
            existing_only=args.existing_only,
            wandb_run_names=wandb_run_names,
            wandb_finished_summaries=wandb_finished_summaries,
        )


if __name__ == "__main__":
    main()
