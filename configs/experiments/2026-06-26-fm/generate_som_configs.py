"""Generate slab-ocean (SOM) equilibrium and abrupt-CO2 configs for the FM runs.

Reproduces the ACE experiments of the ACE2S-SHiELD+ paper repository
(``ai2cm/ace2s-shield-plus-paper``, ``ACE-experiments/inference``) on the 4deg
daily SHiELD-SOM ensemble dataset the FM runs were trained on. Every config is
run-agnostic: the checkpoint is mounted at ``/ckpt.tar`` by submit_som_jobs.py
and the same config is reused across every training run.

The slab ocean is applied at inference time through ``stepper_override``, as in
the paper: the training configs prescribe SST, and the override swaps the
prescribed surface temperature for a mixed-layer tendency driven by the model's
own surface fluxes plus the dataset's ``prescribed_qflux`` and
``prescribed_mixed_layer_depth``. ``interpolate`` is left at the training
default (False) rather than the paper's True, so the only change from training
is the slab itself.

Every config carries ``labels: [som]``. Both the fm and c96 regimes train on the
SOM ensemble under that label, so it is the group a grouped-normalization (A2/A3)
checkpoint resolves and the conditioning one-hot a ``-cond`` checkpoint saw.

Kinds
-----
One config family per kind, named ``ace-som-{kind}-config-4deg-...yaml``:

``eq``
    Paper's equilibrium-climate inference, two stages per (climate, ic): a one
    year spin-up from the spin-up dataset's 2030 state writing
    ``/results/spin-up/restart.nc``, then the ten year main run from that
    restart under the climate's own member as forcing. Needs the daily 4deg
    spin-up dataset (MISSING_DATASETS["spin-up"]). Run by
    run-ace-som-two-stage.sh.
``eq-nospinup``
    Single-stage variant runnable today: the initial condition is the member's
    own 2031 state (already SHiELD-equilibrated), ten years minus the ic
    stagger. ACE's drift to its own equilibrium happens inside the scored
    window instead of the discarded spin-up year.
``eq-1000yr``
    Paper's 1000-year run: forcing is the 1xCO2 member tiled with
    ``n_repeats`` (all forcing but CO2 is climatological in the SOM runs) with
    CO2 overwritten to the climate's constant; initial condition from the
    climate's own member.
``data-only``
    Paper's data-only evaluator: ``loader`` and ``prediction_loader`` both
    point at a reference member, so the evaluator scores SHiELD against itself
    and logs the reference climate's diagnostics in the same format as a model
    run. The checkpoint is loaded only for variable names.
``abrupt-10yr``
    Ten-year abrupt-CO2 inference from the 1xCO2 member's 2031 state, CO2
    overwritten to the target climate's constant. Runnable today; no reference.
``abrupt-10yr-eval``
    Paper's abrupt-4xCO2 evaluator: same run scored against SHiELD's own
    abrupt-CO2 run, whose 2020 state is the initial condition. Needs the daily
    4deg abrupt-CO2 dataset (MISSING_DATASETS["abrupt"]).
``abrupt-data-only``
    SHiELD's abrupt-CO2 run evaluated against itself (same dataset dependency).
``abrupt-ens``
    Paper's abrupt-4xCO2 ensemble evaluator: 36 monthly 1xCO2 initial
    conditions, 90 days each, CO2 overwritten to 4x, scored against the 1xCO2
    member so the metrics are the response.
``abrupt-ens-data-only``
    SHiELD's 36-member abrupt-4xCO2 ensemble evaluated against itself. Needs
    the daily 4deg abrupt-4xCO2 ensemble dataset
    (MISSING_DATASETS["abrupt-ensemble"]).
``7day``
    Paper's seven-day ensemble inference from the same 36 initial conditions,
    once with the 1xCO2 forcing as is and once with CO2 overwritten to 4x.

Step counts are the paper's converted from 6-hourly to daily. See
MISSING_DATASETS.md for the datasets still to be produced.
"""

import argparse
import copy
import pathlib
from typing import NamedTuple

import yaml
from generate_eval_configs import WANDB_ENTITY, WANDB_PROJECT

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIR = HERE / "run_configs"
SOM_CONFIG_PREFIX = "ace-som-"
CHECKPOINT_PATH = "/ckpt.tar"
LABEL = "som"

# The 4deg daily SHiELD-SOM ensemble the FM and c96 regimes train on. Members are
# 1x/2x/4xCO2 ic_0001-ic_0005 and 3xCO2 ic_0001-ic_0002, each 3653 daily steps
# from 2031-01-01T06 to 2041-01-01T06.
SOM_DATASET = (
    "/climate-default/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset"
)
SOM_FIRST_TIME = "2031-01-01T06:00:00"
SOM_N_STEPS = 3652  # ten years at one step per day
SOM_MEMBERS = {
    "1xCO2": ("ic_0001", "ic_0002", "ic_0003", "ic_0004", "ic_0005"),
    "2xCO2": ("ic_0001", "ic_0002", "ic_0003", "ic_0004", "ic_0005"),
    "3xCO2": ("ic_0001", "ic_0002"),
    "4xCO2": ("ic_0001", "ic_0002", "ic_0003", "ic_0004", "ic_0005"),
}


class MissingDataset(NamedTuple):
    path: str
    purpose: str
    source: str
    how: str
    kinds: tuple[str, ...]


# Datasets the paper's experiments need which do not exist at 4deg daily yet.
# Configs referencing them are generated anyway so the machinery is ready; the
# paths carry a TBD date so a stale path fails loudly instead of running on the
# wrong data. Replace the TBD once the dataset is on weka, regenerate, and
# submit_som_jobs.py stops refusing the dependent kinds. See MISSING_DATASETS.md.
_MISSING_ROOT = "/climate-default/TBD-vertically-resolved-4deg-daily-c96-shield-som-"
MISSING_DATASETS = {
    "abrupt": MissingDataset(
        path=_MISSING_ROOT + "abrupt-co2-increase-fme-dataset",
        purpose=(
            "reference trajectory for the abrupt-CO2 evaluator and the abrupt "
            "data-only evaluator"
        ),
        source=(
            "gs://vcm-ml-intermediate/2024-08-14-vertically-resolved-4deg-c96-"
            "shield-som-abrupt-co2-increase-fme-dataset (6-hourly, abrupt-2x/3x/4x)"
        ),
        how=(
            "add a daily time_coarsen block to scripts/data_process/configs/"
            "shield-som-abrupt-co2-increase-c96-4deg-8layer.yaml, as in "
            "shield-som-ensemble-c96-4deg-8layer.yaml"
        ),
        kinds=("abrupt-10yr-eval", "abrupt-data-only"),
    ),
    "spin-up": MissingDataset(
        path=_MISSING_ROOT + "ensemble-spin-up-fme-dataset",
        purpose="initial condition and forcing of the equilibrium spin-up stage",
        source=(
            "gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/"
            "regridded-zarrs/gaussian_grid_45_by_90/{climate}-spin-up-ic_000N"
        ),
        how=(
            "clone scripts/data_process/configs/shield-som-spin-up-c96-1deg-8layer"
            ".yaml to 4deg (gaussian_grid_45_by_90 inputs, daily time_coarsen)"
        ),
        kinds=("eq",),
    ),
    "abrupt-ensemble": MissingDataset(
        path=_MISSING_ROOT + "abrupt-4xCO2-ensemble-fme-dataset",
        purpose="SHiELD's own 36-member abrupt-4xCO2 spread (abrupt-ens data-only)",
        source=(
            "gs://vcm-ml-raw-flexible-retention/2025-02-03-C96-SHiELD-SOM-abrupt-"
            "4xCO2-ensemble/regridded-zarrs/gaussian_grid_180_by_360/"
            "abrupt-4xCO2-ic_00NN (1deg only; no 4deg regrid exists)"
        ),
        how=(
            "regrid to gaussian_grid_45_by_90, then clone scripts/data_process/"
            "configs/shield-som-abrupt4xCO2-ensemble-c96-1deg-8layer.yaml to 4deg "
            "with a daily time_coarsen"
        ),
        kinds=("abrupt-ens-data-only",),
    ),
}
MISSING_PATH_MARKER = "/TBD-"


class Climate(NamedTuple):
    #: Member the paper uses for this climate's forcing, initial conditions
    #: and reference: the held-out ic_0005 where it exists, ic_0002 for 3xCO2.
    member: str
    #: Paper's constant global-mean CO2 (kg/kg) for overwriting the forcing.
    co2: float


CLIMATES = {
    "1xCO2": Climate(member="ic_0005", co2=0.00036343),
    "2xCO2": Climate(member="ic_0005", co2=0.00072686),
    "3xCO2": Climate(member="ic_0002", co2=0.00109029),
    "4xCO2": Climate(member="ic_0005", co2=0.0014537),
}
ABRUPT_CLIMATES = ("2xCO2", "3xCO2", "4xCO2")
CONTROL_CLIMATE = "1xCO2"

# Paper staggers five initial conditions one timestep apart and runs each as
# its own job; at a daily step the stagger is one day.
N_INITIAL_CONDITIONS = 5
# Paper writes daily precipitation only for the 1x and 3x climates.
DAILY_PRECIP_CLIMATES = ("1xCO2", "3xCO2")

# Spin-up stage: paper starts 2030-01-01 in the spin-up dataset and runs one
# year, so every stagger ends at 2031-01-01T06 where the main forcing begins.
SPIN_UP_FIRST_TIME = "2030-01-01T06:00:00"
SPIN_UP_N_STEPS = 365
SPIN_UP_EXPERIMENT_DIR = "/results/spin-up"

# 1000-year run: 365250 daily steps; the 10-year forcing member repeated 101
# times spans 1010 years.
THOUSAND_YEAR_N_STEPS = 365250
THOUSAND_YEAR_N_REPEATS = 101
THOUSAND_YEAR_INITIAL_TIME = "2032-01-01T06:00:00"

# Abrupt-CO2 reference run (paper: 2025-03-28 1deg dataset from 2020-01-01T06,
# 14604 six-hourly steps); the 4deg source starts at the same time.
ABRUPT_FIRST_TIME = "2020-01-01T06:00:00"
ABRUPT_N_STEPS = 3651
# Paper's monthly netCDF variable allowlist for the abrupt runs.
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

# Abrupt ensemble: 36 monthly initial conditions (paper: 2031-01 .. 2033-12)
# run 90 days; the seven-day ensembles reuse the same initial conditions.
ENSEMBLE_INITIAL_TIMES = [
    f"{year}-{month:02d}-01T06:00:00"
    for year in (2031, 2032, 2033)
    for month in range(1, 13)
]
ABRUPT_ENSEMBLE_N_STEPS = 90
ABRUPT_ENSEMBLE_N_MEMBERS = 36
SEVEN_DAY_N_STEPS = 7

# Steps kept in memory: the long-run value used by every FM inference config,
# and the paper's values for the short ensembles and the evaluators.
FORWARD_STEPS_IN_MEMORY = 73
ENSEMBLE_FORWARD_STEPS_IN_MEMORY = 1
EVALUATOR_FORWARD_STEPS_IN_MEMORY = 40

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

INFERENCE_KINDS = ("eq", "eq-nospinup", "eq-1000yr", "abrupt-10yr", "7day")
EVALUATOR_KINDS = (
    "data-only",
    "abrupt-10yr-eval",
    "abrupt-data-only",
    "abrupt-ens",
    "abrupt-ens-data-only",
)
KINDS = INFERENCE_KINDS + EVALUATOR_KINDS
# Kinds run once per reference member with a fixed checkpoint, not per run.
DATA_ONLY_KINDS = ("data-only", "abrupt-data-only", "abrupt-ens-data-only")


def som_config_filename(kind: str, *parts: str) -> str:
    suffix = "-".join(parts)
    return f"{SOM_CONFIG_PREFIX}{kind}-config-4deg-{suffix}.yaml"


def member_path(climate: str, member: str) -> str:
    return f"{SOM_DATASET}/{climate}-{member}.zarr"


def _logging(log_to_wandb: bool = True) -> dict:
    return {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "log_to_screen": True,
        "log_to_file": True,
        "log_to_wandb": log_to_wandb,
    }


def _zarr_dataset(data_path: str, file_pattern: str, co2: float | None = None) -> dict:
    dataset: dict = {
        "data_path": data_path,
        "file_pattern": file_pattern,
        "engine": "zarr",
    }
    if co2 is not None:
        dataset["overwrite"] = {"constant": {"global_mean_co2": co2}}
    return dataset


def _member_dataset(climate: str, member: str, co2: float | None = None) -> dict:
    return _zarr_dataset(SOM_DATASET, f"{climate}-{member}.zarr", co2)


def _daily_precip_files(climate: str) -> list[dict]:
    """Paper's daily-PRATEsfc zarr, for the 1x and 3x climates only.

    The paper coarsens its 6-hourly output by 4 to get daily means; at a daily
    step the raw output already is daily, so no time_coarsen.
    """
    if climate not in DAILY_PRECIP_CLIMATES:
        return []
    return [
        {
            "label": "daily",
            "names": ["PRATEsfc"],
            "save_reference": False,
            "format": {"name": "zarr"},
        }
    ]


def _inference_config(
    *,
    experiment_dir: str,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    initial_condition_path: str,
    initial_condition_engine: str,
    initial_times: list[str],
    forcing_dataset: dict,
    data_writer: dict,
    log_to_wandb: bool = True,
) -> dict:
    return {
        "experiment_dir": experiment_dir,
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": forward_steps_in_memory,
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "labels": [LABEL],
        "logging": _logging(log_to_wandb),
        "initial_condition": {
            "path": initial_condition_path,
            "engine": initial_condition_engine,
            "start_indices": {"times": initial_times},
        },
        "forcing_loader": {
            "dataset": forcing_dataset,
            "num_data_workers": 4,
        },
        "data_writer": data_writer,
        "stepper_override": copy.deepcopy(SLAB_OCEAN_OVERRIDE),
    }


def _evaluator_config(
    *,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    loader_dataset: dict,
    start_indices: dict,
    data_writer: dict,
    prediction_dataset: dict | None = None,
    aggregator: dict | None = None,
) -> dict:
    loader = {
        "dataset": loader_dataset,
        "start_indices": start_indices,
        "num_data_workers": 8,
    }
    cfg: dict = {
        "experiment_dir": "/results",
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": forward_steps_in_memory,
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "labels": [LABEL],
        "logging": _logging(),
        "loader": loader,
    }
    if prediction_dataset is not None:
        cfg["prediction_loader"] = {**loader, "dataset": prediction_dataset}
    cfg["data_writer"] = data_writer
    if aggregator is not None:
        cfg["aggregator"] = aggregator
    cfg["stepper_override"] = copy.deepcopy(SLAB_OCEAN_OVERRIDE)
    return cfg


def _no_files() -> dict:
    return {"save_prediction_files": False, "save_monthly_files": False}


def _files_only(files: list[dict]) -> dict:
    return {**_no_files(), "files": files}


def _stagger_time(first_time: str, offset_days: int) -> str:
    """The date `offset_days` after `first_time`, within the same month."""
    date, clock = first_time.split("T")
    year, month, day = date.split("-")
    return f"{year}-{month}-{int(day) + offset_days:02d}T{clock}"


def build_eq_configs() -> dict[str, dict]:
    configs = {}
    spin_up_root = MISSING_DATASETS["spin-up"].path
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            spin_up_pattern = f"{climate}-spin-up-{spec.member}.zarr"
            configs[som_config_filename("eq-spinup", climate, f"ic{ic}")] = (
                _inference_config(
                    experiment_dir=SPIN_UP_EXPERIMENT_DIR,
                    n_forward_steps=SPIN_UP_N_STEPS - offset,
                    forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
                    initial_condition_path=f"{spin_up_root}/{spin_up_pattern}",
                    initial_condition_engine="zarr",
                    initial_times=[_stagger_time(SPIN_UP_FIRST_TIME, offset)],
                    forcing_dataset=_zarr_dataset(spin_up_root, spin_up_pattern),
                    data_writer=_no_files(),
                    log_to_wandb=False,
                )
            )
            configs[som_config_filename("eq-main", climate, f"ic{ic}")] = (
                _inference_config(
                    experiment_dir="/results",
                    n_forward_steps=SOM_N_STEPS,
                    forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
                    initial_condition_path=f"{SPIN_UP_EXPERIMENT_DIR}/restart.nc",
                    initial_condition_engine="netcdf4",
                    initial_times=[SOM_FIRST_TIME],
                    forcing_dataset=_member_dataset(climate, spec.member),
                    data_writer=_files_only(_daily_precip_files(climate)),
                )
            )
    return configs


def build_eq_nospinup_configs() -> dict[str, dict]:
    configs = {}
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            configs[som_config_filename("eq-nospinup", climate, f"ic{ic}")] = (
                _inference_config(
                    experiment_dir="/results",
                    n_forward_steps=SOM_N_STEPS - offset,
                    forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
                    initial_condition_path=member_path(climate, spec.member),
                    initial_condition_engine="zarr",
                    initial_times=[_stagger_time(SOM_FIRST_TIME, offset)],
                    forcing_dataset=_member_dataset(climate, spec.member),
                    data_writer=_files_only(_daily_precip_files(climate)),
                )
            )
    return configs


def build_eq_1000yr_configs() -> dict[str, dict]:
    control = CLIMATES[CONTROL_CLIMATE]
    configs = {}
    for climate, spec in CLIMATES.items():
        forcing = _member_dataset(CONTROL_CLIMATE, control.member, co2=spec.co2)
        forcing["n_repeats"] = THOUSAND_YEAR_N_REPEATS
        configs[som_config_filename("eq-1000yr", climate)] = _inference_config(
            experiment_dir="/results",
            n_forward_steps=THOUSAND_YEAR_N_STEPS,
            forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
            initial_condition_path=member_path(climate, spec.member),
            initial_condition_engine="zarr",
            initial_times=[THOUSAND_YEAR_INITIAL_TIME],
            forcing_dataset=forcing,
            data_writer=_no_files(),
        )
    return configs


def build_data_only_configs() -> dict[str, dict]:
    configs = {}
    for climate, members in SOM_MEMBERS.items():
        for member in members:
            dataset = _member_dataset(climate, member)
            configs[som_config_filename("data-only", climate, member)] = (
                _evaluator_config(
                    n_forward_steps=SOM_N_STEPS,
                    forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
                    loader_dataset=dataset,
                    start_indices={"times": [SOM_FIRST_TIME]},
                    data_writer=_files_only(_daily_precip_files(climate)),
                    prediction_dataset=copy.deepcopy(dataset),
                )
            )
    return configs


def _abrupt_monthly_writer() -> dict:
    return {
        "save_prediction_files": False,
        "save_monthly_files": True,
        "names": list(ABRUPT_MONTHLY_NAMES),
    }


def _abrupt_aggregator() -> dict:
    return {"log_zonal_mean_images": False, "log_histograms": True}


def build_abrupt_10yr_configs() -> dict[str, dict]:
    control = CLIMATES[CONTROL_CLIMATE]
    configs = {}
    for climate in ABRUPT_CLIMATES:
        configs[som_config_filename("abrupt-10yr", climate)] = _inference_config(
            experiment_dir="/results",
            n_forward_steps=SOM_N_STEPS,
            forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
            initial_condition_path=member_path(CONTROL_CLIMATE, control.member),
            initial_condition_engine="zarr",
            initial_times=[SOM_FIRST_TIME],
            forcing_dataset=_member_dataset(
                CONTROL_CLIMATE, control.member, co2=CLIMATES[climate].co2
            ),
            data_writer=_abrupt_monthly_writer(),
        )
    return configs


def build_abrupt_10yr_eval_configs() -> dict[str, dict]:
    root = MISSING_DATASETS["abrupt"].path
    configs = {}
    for climate in ABRUPT_CLIMATES:
        configs[som_config_filename("abrupt-10yr-eval", climate)] = _evaluator_config(
            n_forward_steps=ABRUPT_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=_zarr_dataset(root, f"abrupt-{climate}.zarr"),
            start_indices={"times": [ABRUPT_FIRST_TIME]},
            data_writer=_abrupt_monthly_writer(),
            aggregator=_abrupt_aggregator(),
        )
    return configs


def build_abrupt_data_only_configs() -> dict[str, dict]:
    root = MISSING_DATASETS["abrupt"].path
    configs = {}
    for climate in ABRUPT_CLIMATES:
        dataset = _zarr_dataset(root, f"abrupt-{climate}.zarr")
        configs[som_config_filename("abrupt-data-only", climate)] = _evaluator_config(
            n_forward_steps=ABRUPT_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=dataset,
            start_indices={"times": [ABRUPT_FIRST_TIME]},
            data_writer=_abrupt_monthly_writer(),
            aggregator=_abrupt_aggregator(),
            prediction_dataset=copy.deepcopy(dataset),
        )
    return configs


def build_abrupt_ens_configs() -> dict[str, dict]:
    control = CLIMATES[CONTROL_CLIMATE]
    return {
        som_config_filename("abrupt-ens", "4xCO2"): _evaluator_config(
            n_forward_steps=ABRUPT_ENSEMBLE_N_STEPS,
            forward_steps_in_memory=ENSEMBLE_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=_member_dataset(
                CONTROL_CLIMATE, control.member, co2=CLIMATES["4xCO2"].co2
            ),
            start_indices={"times": list(ENSEMBLE_INITIAL_TIMES)},
            data_writer=_no_files(),
        )
    }


def build_abrupt_ens_data_only_configs() -> dict[str, dict]:
    root = MISSING_DATASETS["abrupt-ensemble"].path
    configs = {}
    for n in range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1):
        member = f"ic_{n:04d}"
        dataset = _zarr_dataset(root, f"abrupt4xCO2-{member}.zarr")
        configs[som_config_filename("abrupt-ens-data-only", "4xCO2", member)] = (
            _evaluator_config(
                n_forward_steps=ABRUPT_ENSEMBLE_N_STEPS - 1,
                forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
                loader_dataset=dataset,
                start_indices={"list": [0]},
                data_writer=_no_files(),
                prediction_dataset=copy.deepcopy(dataset),
            )
        )
    return configs


def build_7day_configs() -> dict[str, dict]:
    control = CLIMATES[CONTROL_CLIMATE]
    configs = {}
    for climate in (CONTROL_CLIMATE, "4xCO2"):
        co2 = None if climate == CONTROL_CLIMATE else CLIMATES[climate].co2
        configs[som_config_filename("7day", climate)] = _inference_config(
            experiment_dir="/results",
            n_forward_steps=SEVEN_DAY_N_STEPS,
            forward_steps_in_memory=ENSEMBLE_FORWARD_STEPS_IN_MEMORY,
            initial_condition_path=member_path(CONTROL_CLIMATE, control.member),
            initial_condition_engine="zarr",
            initial_times=list(ENSEMBLE_INITIAL_TIMES),
            forcing_dataset=_member_dataset(CONTROL_CLIMATE, control.member, co2=co2),
            data_writer=_no_files(),
        )
    return configs


BUILDERS = {
    "eq": build_eq_configs,
    "eq-nospinup": build_eq_nospinup_configs,
    "eq-1000yr": build_eq_1000yr_configs,
    "data-only": build_data_only_configs,
    "abrupt-10yr": build_abrupt_10yr_configs,
    "abrupt-10yr-eval": build_abrupt_10yr_eval_configs,
    "abrupt-data-only": build_abrupt_data_only_configs,
    "abrupt-ens": build_abrupt_ens_configs,
    "abrupt-ens-data-only": build_abrupt_ens_data_only_configs,
    "7day": build_7day_configs,
}
assert set(BUILDERS) == set(KINDS)


def references_missing_dataset(config_path: pathlib.Path) -> bool:
    """True if a generated config still points at a TBD dataset path."""
    return MISSING_PATH_MARKER in config_path.read_text()


def generate_configs(kinds: list[str] | None = None) -> None:
    RUN_CONFIGS_DIR.mkdir(exist_ok=True)
    for kind in kinds or KINDS:
        for filename, cfg in BUILDERS[kind]().items():
            out_path = RUN_CONFIGS_DIR / filename
            out_path.write_text(
                yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            )
            print(f"Wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        nargs="+",
        choices=KINDS,
        default=None,
        help="Only (re)generate these kinds (default: all).",
    )
    args = parser.parse_args()
    generate_configs(args.kind)


if __name__ == "__main__":
    main()
