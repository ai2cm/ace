"""Generate the run-agnostic inference configs for the 2026-10-19 FM runs.

Every config mounts its checkpoint at /ckpt.tar (supplied per run by
submit_inference_jobs.py), so one config serves every training run. Configs
are written to inference_configs/. Families:

eval
    ``ace-eval-suite-config-4deg-2026-06-26-windows.yaml``: an evaluator suite
    (run_eval_suite.py) reproducing the 2026-06-26 eval suite's test windows.
    Its entries are the inline inference entries of the 2026-06-26 nc-sfno
    fm-a2-cond training config (inference_sources/), with names, windows, step
    counts and aggregators kept and the SHiELD label ``amip`` renamed ``c96``,
    the label the 2026-10-19 runs give every SHiELD C96 dataset. The loader
    form is unchanged: the ERA5 and AMIP stores and their variable set are
    those the 2026-10-19 runs train on. No data writer, as in the old suite.

cm4xshield / cm4-1pct
    ``ace-eval-suite-config-4deg-cm4-xshield.yaml`` and
    ``ace-eval-suite-config-4deg-cm4-1pct.yaml``: in-sample evaluator suites
    on the CM4, AM4 random-CO2 and X-SHiELD stores the 2026-10-19 runs train
    on. Each dataset entry is copied from the training config's train_loader
    (renamed_variables, the 1e-6 CO2 rescale, derived ocean_fraction, label)
    without its time subset; the aggregator is the training config's
    ``10year_oosample_varying_co2_c96`` entry's. The 140-year 1pctCO2 run is
    its own suite so it does not hold up the others.

sst
    ``ace-inference-sst-config-4deg-{grid}-{label}-{level}.yaml``: free runs
    with SST shifted by a constant 0/2/4 K on ERA5 (label era5) and on AMIP
    ic_0001 (label c96), ported from the 2026-06-26 generate_sst_configs.py
    best-inference sweep, but writing monthly netCDF only (no daily
    prediction files), as its epoch sweep did.

paper-som / paper-era5
    ``ace-paper-{kind}-config-4deg.yaml``: six kinds ported from the
    2026-06-26 generate_paper_configs.py (see PAPER_KINDS). The SHiELD-SOM
    kinds are labeled c96 (the 2026-06-06 ``som`` label is folded into c96 in
    the 2026-10-19 vocabulary), the ERA5 kinds era5. Every ``ens`` kind is one
    evaluator with 36 initial conditions, so one job per run.

Usage:
    python generate_inference_configs.py [--family FAMILY ...]
"""

import argparse
import copy
import pathlib
from collections.abc import Callable

import yaml
from _inference_common import (
    INFERENCE_CONFIGS_DIR,
    MOUNTED_CHECKPOINT_PATH,
    WANDB_ENTITY,
    WANDB_PROJECT,
    inference_run_names,
)
from submit_fm_jobs import CONFIG_PREFIX, RUN_CONFIGS_DIR, WANDB_PREFIX

HERE = pathlib.Path(__file__).parent

# The training config the CM4/X-SHiELD loader entries and aggregator are copied
# from. Every nc-sfno four-source config shares them (checked by
# check_shared_sources); they differ in seed, masking and label balancing only.
SOURCE_RUN_CONFIG = RUN_CONFIGS_DIR / (
    "ace-train-config-4deg-nc-sfno-fm-a2-cond-era5-c96-piC-1pct-cm4-xshield"
    "-s0-maskon.yaml"
)
OLD_INLINE_INFERENCE = (
    HERE / "inference_sources" / "2026-06-26-nc-sfno-fm-a2-cond-inline-inference.yaml"
)
TEN_YEAR_AGGREGATOR_ENTRY = "10year_oosample_varying_co2_c96"

# 2026-06-26 label -> 2026-10-19 label. Every 2026-06-26 SHiELD label (amip,
# ramped, som) is c96 now.
LABEL_RENAMES = {"amip": "c96", "ramped": "c96", "som": "c96", "era5": "era5"}

EVAL_SUITE_PREFIX = "ace-eval-suite-config-4deg-"
SST_CONFIG_PREFIX = "ace-inference-sst-config-4deg-"
PAPER_CONFIG_PREFIX = "ace-paper-"

EVAL_SUITE_CONFIGS = {
    "eval": f"{EVAL_SUITE_PREFIX}2026-06-26-windows.yaml",
    "cm4xshield": f"{EVAL_SUITE_PREFIX}cm4-xshield.yaml",
    "cm4-1pct": f"{EVAL_SUITE_PREFIX}cm4-1pct.yaml",
}


def _logging() -> dict:
    return {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "log_to_screen": True,
        "log_to_file": True,
        "log_to_wandb": True,
    }


def _no_files() -> dict:
    return {"save_prediction_files": False, "save_monthly_files": False}


def _load_yaml(path: pathlib.Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _dump(cfg: dict, path: pathlib.Path, header: str = "") -> None:
    class _NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        cfg, Dumper=_NoAliasDumper, default_flow_style=False, sort_keys=False
    )
    path.write_text(header + text)
    print(f"Wrote {path.relative_to(HERE)}")


# --- evaluator suites -----------------------------------------------------------


def _suite_entry(
    name: str,
    *,
    loader: dict,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    aggregator: dict,
    n_ensemble_per_ic: int = 1,
) -> dict:
    """One evaluator of a suite; ported from the 2026-06-26
    generate_eval_configs._build_eval_config.
    """
    return {
        "name": name,
        "config": {
            "experiment_dir": f"/results/{name}",
            "n_forward_steps": n_forward_steps,
            "forward_steps_in_memory": forward_steps_in_memory,
            "checkpoint_path": MOUNTED_CHECKPOINT_PATH,
            "logging": _logging(),
            "loader": loader,
            "aggregator": aggregator,
            "data_writer": _no_files(),
            "n_ensemble_per_ic": n_ensemble_per_ic,
            # Training mixes datasets on different vertical grids via
            # strict:false and pins the first concat member's coordinate into
            # the checkpoint; the standalone evaluator checks compatibility,
            # so allow the same mismatch here.
            "allow_incompatible_dataset": True,
        },
    }


def _suite(entries: list[dict]) -> dict:
    return {"experiment_dir": "/results", "logging": _logging(), "inferences": entries}


def _relabel(dataset: dict) -> dict:
    dataset = copy.deepcopy(dataset)
    if "labels" in dataset:
        dataset["labels"] = [LABEL_RENAMES[label] for label in dataset["labels"]]
    for key in ("concat", "merge"):
        if key in dataset:
            dataset[key] = [_relabel(d) for d in dataset[key]]
    return dataset


def build_eval_suite() -> dict:
    """The 2026-06-26 eval-suite windows on the 2026-10-19 checkpoints."""
    old_entries = _load_yaml(OLD_INLINE_INFERENCE)["inference"]
    entries = []
    for old in old_entries:
        loader = copy.deepcopy(old["loader"])
        loader["dataset"] = _relabel(loader["dataset"])
        entries.append(
            _suite_entry(
                old["name"],
                loader=loader,
                n_forward_steps=old["n_forward_steps"],
                forward_steps_in_memory=old["forward_steps_in_memory"],
                aggregator=copy.deepcopy(old.get("aggregator", {})),
                n_ensemble_per_ic=old.get("n_ensemble_per_ic", 1),
            )
        )
    return _suite(entries)


def _train_dataset(train_cfg: dict, file_pattern: str) -> dict:
    """The train_loader concat member reading `file_pattern`, without subset."""
    matches = [
        d
        for d in train_cfg["train_loader"]["dataset"]["concat"]
        if d["file_pattern"] == file_pattern
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one train dataset {file_pattern!r}, got {matches}")
    dataset = copy.deepcopy(matches[0])
    dataset.pop("subset", None)
    return dataset


def _ten_year_entry(train_cfg: dict) -> dict:
    for entry in train_cfg["inference"]:
        if entry["name"] == TEN_YEAR_AGGREGATOR_ENTRY:
            return entry
    raise ValueError(f"no {TEN_YEAR_AGGREGATOR_ENTRY!r} entry in {SOURCE_RUN_CONFIG}")


CM4_PICONTROL = "2026-06-19-CM4-piControl-atmosphere-land-4deg-8layer-200yr.zarr"
CM4_1PCT = "2026-06-19-CM4-1pctCO2-atmosphere-land-4deg-8layer-140yr.zarr"
AM4_RANDOM_CO2 = "random-CO2-{co2}-ic_0003.zarr"
XSHIELD = "2025-09-11-X-SHiELD-AMIP-4deg-8layer-11yr.zarr"

# Step counts from cm4-xshield-dataset-audit.md (daily copies): piControl
# 73,000 days, noleap from year 0151; 1pctCO2 51,100 days, noleap 0001-0141;
# AM4 random CO2 1,917 days per store; X-SHiELD 4,020 days, julian,
# 2013-01-02 .. 2024-01-04 (2013 is spin-up, left out of its stats).
CM4_PICONTROL_10YEAR_STEPS = 3650  # ten noleap years
CM4_PICONTROL_N_IC = 8
AM4_RANDOM_CO2_STEPS = 1916  # whole store from its first time
CM4_1PCT_STEPS = 51099  # whole store from its first time
XSHIELD_10YEAR_STEPS = 3652  # 2014-01-01 .. 2024-01-01
# Only four daily ICs fit: the store ends 2024-01-04.
XSHIELD_START_TIMES = [f"2014-01-0{day}T00:00:00" for day in range(1, 5)]


def _cm4_xshield_entries(train_cfg: dict, include_1pct: bool) -> list[dict]:
    ten_year = _ten_year_entry(train_cfg)
    aggregator = copy.deepcopy(ten_year["aggregator"])
    fsim = ten_year["forward_steps_in_memory"]
    workers = ten_year["loader"].get("num_data_workers", 4)

    def loader(file_pattern: str, start_indices: dict) -> dict:
        return {
            "dataset": _train_dataset(train_cfg, file_pattern),
            "start_indices": start_indices,
            "num_data_workers": workers,
        }

    if include_1pct:
        return [
            _suite_entry(
                "cm4_1pctco2_140year",
                loader=loader(CM4_1PCT, {"first": 0, "n_initial_conditions": 1}),
                n_forward_steps=CM4_1PCT_STEPS,
                forward_steps_in_memory=fsim,
                aggregator=aggregator,
            )
        ]
    entries = [
        _suite_entry(
            "cm4_picontrol_10year",
            loader=loader(
                CM4_PICONTROL,
                {"first": 0, "n_initial_conditions": CM4_PICONTROL_N_IC, "interval": 1},
            ),
            n_forward_steps=CM4_PICONTROL_10YEAR_STEPS,
            forward_steps_in_memory=fsim,
            aggregator=copy.deepcopy(aggregator),
        )
    ]
    for co2 in ("1x", "2x", "4x"):
        entries.append(
            _suite_entry(
                f"am4_random_co2_{co2}_ic0003",
                loader=loader(
                    AM4_RANDOM_CO2.format(co2=f"{co2}CO2"),
                    {"first": 0, "n_initial_conditions": 1},
                ),
                n_forward_steps=AM4_RANDOM_CO2_STEPS,
                forward_steps_in_memory=fsim,
                aggregator=copy.deepcopy(aggregator),
            )
        )
    entries.append(
        _suite_entry(
            "xshield_10year",
            loader=loader(XSHIELD, {"times": list(XSHIELD_START_TIMES)}),
            n_forward_steps=XSHIELD_10YEAR_STEPS,
            forward_steps_in_memory=fsim,
            aggregator=copy.deepcopy(aggregator),
        )
    )
    return entries


def build_cm4_xshield_suite() -> dict:
    return _suite(_cm4_xshield_entries(_load_yaml(SOURCE_RUN_CONFIG), False))


def build_cm4_1pct_suite() -> dict:
    return _suite(_cm4_xshield_entries(_load_yaml(SOURCE_RUN_CONFIG), True))


def check_shared_sources(run_config_paths: list[pathlib.Path]) -> None:
    """Fail unless every run config shares SOURCE_RUN_CONFIG's CM4/X-SHiELD
    loader entries and 10-year aggregator, so one suite fits all runs.
    """
    reference = _load_yaml(SOURCE_RUN_CONFIG)
    patterns = [CM4_PICONTROL, CM4_1PCT, XSHIELD] + [
        AM4_RANDOM_CO2.format(co2=f"{c}CO2") for c in ("1x", "2x", "4x")
    ]
    for path in run_config_paths:
        cfg = _load_yaml(path)
        for pattern in patterns:
            if _train_dataset(cfg, pattern) != _train_dataset(reference, pattern):
                raise ValueError(f"{path.name}: {pattern} entry differs")
        if (
            _ten_year_entry(cfg)["aggregator"]
            != _ten_year_entry(reference)["aggregator"]
        ):
            raise ValueError(f"{path.name}: 10-year aggregator differs")


# --- SST perturbations ------------------------------------------------------------

SST_PERTURBATIONS = {"p0k": 0.0, "p2k": 2.0, "p4k": 4.0}
SST_FORWARD_STEPS_IN_MEMORY = 73
SST_INITIAL_CONDITION_TIME = "1979-01-01T00:00:00"
# grid -> (data_path, file_pattern, n_forward_steps, label). Step counts are the
# 2026-06-26 sweep's (long_46year for ERA5, long_43year for AMIP).
SST_GRIDS = {
    "era5": (
        "/climate-default",
        "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr",
        16794,
        "era5",
    ),
    "c96": (
        "/climate-default/"
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset",
        "ic_0001.zarr",
        15683,
        "c96",
    ),
}


def sst_config_filename(grid: str, level: str) -> str:
    label = SST_GRIDS[grid][3]
    return f"{SST_CONFIG_PREFIX}{grid}-{label}-{level}.yaml"


def build_sst_config(grid: str, level: str) -> dict:
    data_path, file_pattern, n_steps, label = SST_GRIDS[grid]
    return {
        "checkpoint_path": MOUNTED_CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "experiment_dir": "/results",
        "n_forward_steps": n_steps,
        "forward_steps_in_memory": SST_FORWARD_STEPS_IN_MEMORY,
        # Sets the label on the initial condition and every forcing window.
        "labels": [label],
        # No daily prediction files (~15 GiB per job, unread); monthly means
        # only. The aggregator's diagnostics (annual_diagnostics.nc,
        # time_mean_diagnostics.nc, ...) are written regardless.
        "data_writer": {"save_monthly_files": True, "save_prediction_files": False},
        "initial_condition": {
            "path": f"{data_path}/{file_pattern}",
            "engine": "zarr",
            "start_indices": {"times": [SST_INITIAL_CONDITION_TIME]},
        },
        "forcing_loader": {
            "dataset": {
                "data_path": data_path,
                "file_pattern": file_pattern,
                "engine": "zarr",
            },
            "num_data_workers": 4,
            "perturbations": {
                "sst": [
                    {
                        "type": "constant",
                        "config": {"amplitude": SST_PERTURBATIONS[level]},
                    }
                ]
            },
        },
        "logging": {"project": WANDB_PROJECT, "entity": WANDB_ENTITY},
    }


# --- paper kinds --------------------------------------------------------------------

# Ported from the 2026-06-26 generate_paper_configs.py; see its docstring for
# the kind naming and the experiments.
SOM_DATASET = (
    "/climate-default/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset"
)
# Paper's 1xCO2 member (held out of every 2026-10-19 run, which train on
# ic_0001) and the 4xCO2 constant global-mean CO2 (volume mixing ratio).
SOM_CONTROL_FILE = "1xCO2-ic_0005.zarr"
SOM_4XCO2 = 0.0014537
SOM_ENSEMBLE_INITIAL_TIMES = [
    f"{year}-{month:02d}-02T00:00:00"
    for year in (2031, 2032, 2033)
    for month in range(1, 13)
]
ERA5_DATA_PATH = "/climate-default"
ERA5_FILE = "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"
ERA5_ABRUPT_FIRST_TIME = "2015-01-01T00:00:00"
ERA5_ABRUPT_N_STEPS = 3652
ERA5_ENSEMBLE_INITIAL_TIMES = [
    f"{year}-{month:02d}-01T00:00:00"
    for year in (2015, 2016, 2017)
    for month in range(1, 13)
]
# Four times ERA5's global_mean_co2 on 2015-01-01.
ERA5_CO2_4X = 4 * 0.00039861797
ENSEMBLE_N_STEPS = 90
ENSEMBLE_FORWARD_STEPS_IN_MEMORY = 1
EVALUATOR_FORWARD_STEPS_IN_MEMORY = 40
ABRUPT_MONTHLY_NAMES = [
    "surface_temperature",
    "LHTFLsfc",
    "SHTFLsfc",
    "DLWRFsfc",
    "DSWRFsfc",
    "ULWRFsfc",
    "USWRFsfc",
    "ULWRFtoa",
    "USWRFtoa",
    "PRATEsfc",
    "total_frozen_precipitation_rate",
]
SLAB_OCEAN_OVERRIDE = {
    "ocean": {
        "surface_temperature_name": "surface_temperature",
        "ocean_fraction_name": "ocean_fraction",
        "interpolate": False,
        "slab": {
            "mixed_layer_depth_name": "prescribed_mixed_layer_depth",
            "q_flux_name": "prescribed_qflux",
        },
    }
}
SOM_LABEL = LABEL_RENAMES["som"]
ERA5_LABEL = "era5"


def _zarr_dataset(data_path: str, file_pattern: str, co2: float | None = None) -> dict:
    dataset: dict = {
        "data_path": data_path,
        "file_pattern": file_pattern,
        "engine": "zarr",
    }
    if co2 is not None:
        dataset["overwrite"] = {"constant": {"global_mean_co2": co2}}
    return dataset


def _evaluator_config(
    *,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    loader_dataset: dict,
    start_times: list[str],
    data_writer: dict,
    label: str,
    aggregator: dict | None = None,
    slab: bool = False,
) -> dict:
    cfg: dict = {
        "experiment_dir": "/results",
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": forward_steps_in_memory,
        "checkpoint_path": MOUNTED_CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "labels": [label],
        "logging": _logging(),
        "loader": {
            "dataset": loader_dataset,
            "start_indices": {"times": list(start_times)},
            "num_data_workers": 8,
        },
        "data_writer": data_writer,
    }
    if aggregator is not None:
        cfg["aggregator"] = aggregator
    if slab:
        cfg["stepper_override"] = copy.deepcopy(SLAB_OCEAN_OVERRIDE)
    return cfg


def _ensemble_config(dataset: dict, times: list[str], label: str, slab: bool) -> dict:
    """Paper's 36-monthly-IC, 90-day ensemble evaluator: one config, one job."""
    return _evaluator_config(
        n_forward_steps=ENSEMBLE_N_STEPS,
        forward_steps_in_memory=ENSEMBLE_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=dataset,
        start_times=times,
        data_writer=_no_files(),
        label=label,
        slab=slab,
    )


def _era5_10yr_config(co2: float | None) -> dict:
    return _evaluator_config(
        n_forward_steps=ERA5_ABRUPT_N_STEPS,
        forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=_zarr_dataset(ERA5_DATA_PATH, ERA5_FILE, co2),
        start_times=[ERA5_ABRUPT_FIRST_TIME],
        data_writer={
            "save_prediction_files": False,
            "save_monthly_files": True,
            "names": list(ABRUPT_MONTHLY_NAMES),
        },
        label=ERA5_LABEL,
        aggregator={"log_zonal_mean_images": False},
    )


PAPER_KINDS: dict[str, tuple[str, Callable[[], dict]]] = {
    # kind -> (family, builder)
    "som-abrupt-4xCO2-ens-sstslab-eval": (
        "paper-som",
        lambda: _ensemble_config(
            _zarr_dataset(SOM_DATASET, SOM_CONTROL_FILE, SOM_4XCO2),
            SOM_ENSEMBLE_INITIAL_TIMES,
            SOM_LABEL,
            slab=True,
        ),
    ),
    "som-control-dataCO2-ens-sstslab-eval": (
        "paper-som",
        lambda: _ensemble_config(
            _zarr_dataset(SOM_DATASET, SOM_CONTROL_FILE),
            SOM_ENSEMBLE_INITIAL_TIMES,
            SOM_LABEL,
            slab=True,
        ),
    ),
    "era5-abrupt-4xCO2-10yr-sstprescribed-eval": (
        "paper-era5",
        lambda: _era5_10yr_config(ERA5_CO2_4X),
    ),
    "era5-control-dataCO2-10yr-sstprescribed-eval": (
        "paper-era5",
        lambda: _era5_10yr_config(None),
    ),
    "era5-abrupt-4xCO2-ens-sstprescribed-eval": (
        "paper-era5",
        lambda: _ensemble_config(
            _zarr_dataset(ERA5_DATA_PATH, ERA5_FILE, ERA5_CO2_4X),
            ERA5_ENSEMBLE_INITIAL_TIMES,
            ERA5_LABEL,
            slab=False,
        ),
    ),
    "era5-control-dataCO2-ens-sstprescribed-eval": (
        "paper-era5",
        lambda: _ensemble_config(
            _zarr_dataset(ERA5_DATA_PATH, ERA5_FILE),
            ERA5_ENSEMBLE_INITIAL_TIMES,
            ERA5_LABEL,
            slab=False,
        ),
    ),
}


def paper_config_filename(kind: str) -> str:
    return f"{PAPER_CONFIG_PREFIX}{kind}-config-4deg.yaml"


# --- families -------------------------------------------------------------------------

FAMILIES = ("eval", "cm4xshield", "cm4-1pct", "sst", "paper-som", "paper-era5")


def family_configs(family: str) -> dict[str, dict]:
    """Filename -> config of one family."""
    if family == "eval":
        return {EVAL_SUITE_CONFIGS[family]: build_eval_suite()}
    if family == "cm4xshield":
        return {EVAL_SUITE_CONFIGS[family]: build_cm4_xshield_suite()}
    if family == "cm4-1pct":
        return {EVAL_SUITE_CONFIGS[family]: build_cm4_1pct_suite()}
    if family == "sst":
        return {
            sst_config_filename(grid, level): build_sst_config(grid, level)
            for grid in SST_GRIDS
            for level in SST_PERTURBATIONS
        }
    if family in ("paper-som", "paper-era5"):
        return {
            paper_config_filename(kind): builder()
            for kind, (kind_family, builder) in PAPER_KINDS.items()
            if kind_family == family
        }
    raise ValueError(f"unknown family {family!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Families to (re)generate (default: all).",
    )
    args = parser.parse_args()
    check_shared_sources(
        [
            RUN_CONFIGS_DIR / f"{CONFIG_PREFIX}{run.removeprefix(WANDB_PREFIX)}.yaml"
            for run in inference_run_names()
        ]
    )
    header = "# Generated by generate_inference_configs.py; do not edit by hand.\n"
    for family in args.family:
        for filename, cfg in family_configs(family).items():
            _dump(cfg, INFERENCE_CONFIGS_DIR / filename, header)


if __name__ == "__main__":
    main()
