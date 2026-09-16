"""Generate the ACE2S-SHiELD+ paper experiment configs for the FM runs.

Reproduces the ACE experiments of the ACE2S-SHiELD+ paper repository
(``ai2cm/ace2s-shield-plus-paper``, ``ACE-experiments/inference``) on the 4deg
daily SHiELD datasets the FM runs were trained on: the slab-ocean (SOM) climate
experiments on the SOM ensemble, and the prescribed-SST experiments on the AMIP
and ramped-SST random-CO2 stores. Every config is run-agnostic: the checkpoint
is mounted at ``/ckpt.tar`` by submit_paper_jobs.py and the same config is
reused across every training run.

For the slab-ocean kinds the slab is applied at inference time through
``stepper_override``, as in the paper: the training configs prescribe SST, and
the override swaps the prescribed surface temperature for a mixed-layer tendency
driven by the model's own surface fluxes plus the dataset's ``prescribed_qflux``
and ``prescribed_mixed_layer_depth``. ``interpolate`` is left at the training
default (False) rather than the paper's True, so the only change from training
is the slab itself.

Every config carries the label of its data (``som`` for the slab-ocean kinds;
see "Prescribed-SST kinds" for the rest). Both the fm and c96 regimes train on
these stores under these labels, so the label is the group a grouped-
normalization (A2/A3) checkpoint resolves and the conditioning one-hot a
``-cond`` checkpoint saw.

Slab-ocean kinds
----------------
One config family per kind, named ``ace-paper-{kind}-config-4deg-...yaml``:

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
``abrupt-10yr-eval-sst``
    The abrupt-CO2 evaluator with the surface prescribed instead of the slab:
    SST and sea ice are read from SHiELD's abrupt run at every step (the
    training-time ocean, no ``stepper_override``), so the score isolates the
    atmospheric response given SHiELD's own surface warming.
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

Prescribed-SST kinds
--------------------
These run the training-time ocean (SST and sea ice read from the reference at
every step, no ``stepper_override``, so ``interpolate`` stays the training
default where the paper sets it True) and score against the reference member,
which is a valid step-by-step target under prescribed SST. The paper ran the
AMIP experiments as free inference plus separate data-only evaluators and
compared offline; an evaluator gives that comparison directly.

``eq-eval-sst``
    Prescribed-SST control for ``eq``/``eq-nospinup``: the same climates,
    members and five staggered initial conditions, scored against the member
    itself. Separates the atmospheric error from the slab-feedback error.
``amip-eval``
    Paper's AMIP run on the held-out ensemble member ic_0002, 1979-01-01 to
    the end of the store, as one evaluator run instead of the paper's three
    chained inference stages (spin-up 1979 / train-validate 1980-2011 / test
    2012-2020): restart chaining gives the identical trajectory, so the
    windows are an analysis cut. Writes the paper's daily-PRATEsfc zarr.
``amip-p4k``, ``amip-p2k``
    Same on SHiELD's AMIP +4 K / +2 K SST runs, initialized from their own
    1979 state. Unlike the SST sweep (generate_sst_configs.py), which
    perturbs the forcing SST and has no reference, these score against
    SHiELD's own response to the perturbation.
``amip-data-only``
    AMIP ic_0002, +4 K and +2 K evaluated against themselves from 1980-01-01
    (the paper's data-only evaluators skip the spin-up year).
``random-co2-eval``
    Paper's random-CO2 evaluator on the held-out member ic_0003 of the
    ramped-climatological-SST random-CO2-perturbation runs, 1x/2x/4xCO2,
    2019-10-01 to the end of the store.

Labels follow the data: ``som`` for SOM members, ``amip`` for the AMIP family,
``ramped`` for the random-CO2 runs. The +2 K/+4 K runs never appeared in
training under any label; ``amip`` is the closest distribution and what the
SST sweep implies. "Held out" means held out of the norm-ablation cells: the
hand-written fm-random-v1/v3 and fm-0.x-v1 runs trained on AMIP ic_0002 and
ramped ic_0003 as well.

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
PAPER_CONFIG_PREFIX = "ace-paper-"
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
    #: False until the dataset is on weka; submit_paper_jobs.py refuses the
    #: dependent kinds while it is False.
    available: bool = False


# Datasets the paper's experiments need which do not exist at 4deg daily yet.
# Configs referencing them are generated anyway so the machinery is ready. A
# path is the real name once its processing config exists (so the two agree)
# and a TBD date before that, so a stale path fails loudly instead of running
# on the wrong data. Flip ``available`` once the dataset is on weka and
# submit_paper_jobs.py stops refusing the dependent kinds. See MISSING_DATASETS.md.
_MISSING_ROOT = "/climate-default/TBD-vertically-resolved-4deg-daily-c96-shield-som-"
MISSING_DATASETS = {
    "abrupt": MissingDataset(
        path=(
            "/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-"
            "som-abrupt-co2-increase-fme-dataset"
        ),
        purpose=(
            "reference trajectory for the abrupt-CO2 evaluator and the abrupt "
            "data-only evaluator"
        ),
        source=(
            "gs://vcm-ml-intermediate/2024-08-14-vertically-resolved-4deg-c96-"
            "shield-som-abrupt-co2-increase-fme-dataset (6-hourly, abrupt-2x/3x/4x)"
        ),
        how=(
            "make shield_som_abrupt_co2_increase_c96_dataset RESOLUTION=4deg in "
            "scripts/data_process (argo), then copy_zarrs_to_weka.py"
        ),
        kinds=("abrupt-10yr-eval", "abrupt-10yr-eval-sst", "abrupt-data-only"),
        available=True,  # copied to weka 2026-09-16
    ),
    "spin-up": MissingDataset(
        path=(
            "/climate-default/2026-09-16-vertically-resolved-4deg-daily-c96-shield-"
            "som-ensemble-spin-up-fme-dataset"
        ),
        purpose="initial condition and forcing of the equilibrium spin-up stage",
        source=(
            "gs://vcm-ml-raw-flexible-retention/2024-07-03-C96-SHiELD-SOM/"
            "regridded-zarrs/gaussian_grid_45_by_90/{climate}-spin-up-ic_000N"
        ),
        how=(
            "make shield_som_c96_spin_up_dataset RESOLUTION=4deg in "
            "scripts/data_process (argo), then copy_zarrs_to_weka.py"
        ),
        kinds=("eq",),
        available=True,  # copied to weka 2026-09-16
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


class PrescribedSstDataset(NamedTuple):
    data_path: str
    file_pattern: str


# AMIP family, prescribed observed SST: the SHiELD AMIP ensemble (the norm-
# ablation cells trained on ic_0001 and held out ic_0002) and SHiELD's AMIP
# runs with the SST uniformly +2 K / +4 K. All three stores run
# 1979-01-01T06 .. 2021-12-15T06 (15690 daily steps; the ensemble members start
# earlier, in 1939).
AMIP_LABEL = "amip"
AMIP_HELD_OUT_MEMBER = "ic_0002"
AMIP_VARIANTS = {
    AMIP_HELD_OUT_MEMBER: PrescribedSstDataset(
        "/climate-default/"
        "2026-01-28-vertically-resolved-c96-4deg-daily-shield-amip-ensemble-dataset",
        f"{AMIP_HELD_OUT_MEMBER}.zarr",
    ),
    "p4k": PrescribedSstDataset(
        "/climate-default/"
        "2026-07-09-vertically-resolved-c96-4deg-daily-shield-amip-p4k-dataset",
        "AMIP-p4K.zarr",
    ),
    "p2k": PrescribedSstDataset(
        "/climate-default/"
        "2026-07-09-vertically-resolved-c96-4deg-daily-shield-amip-p2k-dataset",
        "AMIP-p2K.zarr",
    ),
}
# Paper's AMIP inference starts 1979-01-01T06 and its data-only evaluator
# 1980-01-01, skipping the spin-up year; both run to the end of the store.
AMIP_FIRST_TIME = "1979-01-01T06:00:00"
AMIP_N_STEPS = 15689
AMIP_DATA_ONLY_FIRST_TIME = "1980-01-01T06:00:00"
AMIP_DATA_ONLY_N_STEPS = 15324

# Ramped-climatological-SST random-CO2 runs: the cells trained on the unnamed
# member and ic_0002 of each climate, ic_0003 is held out. 1919 daily steps from
# 2019-10-01T06; the paper's 7675 six-hourly steps cover the same span.
RAMPED_LABEL = "ramped"
RAMPED_DATASET = (
    "/climate-default/2026-06-08-vertically-resolved-c96-shield-ramped-climSST-"
    "random-CO2-ensemble-fme-dataset-4deg-daily"
)
RAMPED_HELD_OUT_MEMBER = "ic_0003"
RAMPED_CLIMATES = ("1xCO2", "2xCO2", "4xCO2")
RAMPED_FIRST_TIME = "2019-10-01T06:00:00"
RAMPED_N_STEPS = 1918

INFERENCE_KINDS = ("eq", "eq-nospinup", "eq-1000yr", "abrupt-10yr", "7day")
EVALUATOR_KINDS = (
    "data-only",
    "abrupt-10yr-eval",
    "abrupt-10yr-eval-sst",
    "abrupt-data-only",
    "abrupt-ens",
    "abrupt-ens-data-only",
    "eq-eval-sst",
    "amip-eval",
    "amip-p4k",
    "amip-p2k",
    "amip-data-only",
    "random-co2-eval",
)
KINDS = INFERENCE_KINDS + EVALUATOR_KINDS
# Kinds run once per reference member with a fixed checkpoint, not per run.
DATA_ONLY_KINDS = (
    "data-only",
    "abrupt-data-only",
    "abrupt-ens-data-only",
    "amip-data-only",
)


def paper_config_filename(kind: str, *parts: str) -> str:
    suffix = "-".join(parts)
    return f"{PAPER_CONFIG_PREFIX}{kind}-config-4deg-{suffix}.yaml"


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


def _daily_precip_file() -> dict:
    """Paper's daily-PRATEsfc zarr.

    The paper coarsens its 6-hourly output by 4 to get daily means; at a daily
    step the raw output already is daily, so no time_coarsen.
    """
    return {
        "label": "daily",
        "names": ["PRATEsfc"],
        "save_reference": False,
        "format": {"name": "zarr"},
    }


def _daily_precip_files(climate: str) -> list[dict]:
    """The daily-PRATEsfc zarr for the 1x and 3x climates only, as the paper."""
    if climate not in DAILY_PRECIP_CLIMATES:
        return []
    return [_daily_precip_file()]


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
    slab: bool = True,
    label: str = LABEL,
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
        "labels": [label],
        "logging": _logging(),
        "loader": loader,
    }
    if prediction_dataset is not None:
        cfg["prediction_loader"] = {**loader, "dataset": prediction_dataset}
    cfg["data_writer"] = data_writer
    if aggregator is not None:
        cfg["aggregator"] = aggregator
    if slab:
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
            configs[paper_config_filename("eq-spinup", climate, f"ic{ic}")] = (
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
            configs[paper_config_filename("eq-main", climate, f"ic{ic}")] = (
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
            configs[paper_config_filename("eq-nospinup", climate, f"ic{ic}")] = (
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
        configs[paper_config_filename("eq-1000yr", climate)] = _inference_config(
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
            configs[paper_config_filename("data-only", climate, member)] = (
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
        configs[paper_config_filename("abrupt-10yr", climate)] = _inference_config(
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


def _build_abrupt_10yr_eval_configs(kind: str, slab: bool) -> dict[str, dict]:
    root = MISSING_DATASETS["abrupt"].path
    configs = {}
    for climate in ABRUPT_CLIMATES:
        configs[paper_config_filename(kind, climate)] = _evaluator_config(
            n_forward_steps=ABRUPT_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=_zarr_dataset(root, f"abrupt-{climate}.zarr"),
            start_indices={"times": [ABRUPT_FIRST_TIME]},
            data_writer=_abrupt_monthly_writer(),
            aggregator=_abrupt_aggregator(),
            slab=slab,
        )
    return configs


def build_abrupt_10yr_eval_configs() -> dict[str, dict]:
    return _build_abrupt_10yr_eval_configs("abrupt-10yr-eval", slab=True)


def build_abrupt_10yr_eval_sst_configs() -> dict[str, dict]:
    return _build_abrupt_10yr_eval_configs("abrupt-10yr-eval-sst", slab=False)


def build_abrupt_data_only_configs() -> dict[str, dict]:
    root = MISSING_DATASETS["abrupt"].path
    configs = {}
    for climate in ABRUPT_CLIMATES:
        dataset = _zarr_dataset(root, f"abrupt-{climate}.zarr")
        configs[paper_config_filename("abrupt-data-only", climate)] = _evaluator_config(
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
        paper_config_filename("abrupt-ens", "4xCO2"): _evaluator_config(
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
        configs[paper_config_filename("abrupt-ens-data-only", "4xCO2", member)] = (
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
        configs[paper_config_filename("7day", climate)] = _inference_config(
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


def _prescribed_sst_aggregator() -> dict:
    """Paper's aggregator for the AMIP and random-CO2 evaluators."""
    return {"log_zonal_mean_images": False}


def build_eq_eval_sst_configs() -> dict[str, dict]:
    configs = {}
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            configs[paper_config_filename("eq-eval-sst", climate, f"ic{ic}")] = (
                _evaluator_config(
                    n_forward_steps=SOM_N_STEPS - offset,
                    forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
                    loader_dataset=_member_dataset(climate, spec.member),
                    start_indices={"times": [_stagger_time(SOM_FIRST_TIME, offset)]},
                    data_writer=_files_only(_daily_precip_files(climate)),
                    aggregator=_prescribed_sst_aggregator(),
                    slab=False,
                )
            )
    return configs


def _amip_evaluator_config(variant: str, data_only: bool) -> dict:
    """Evaluator on one AMIP-family store: the model against it, or (data_only)
    the store against itself from the paper's later start.
    """
    dataset = _zarr_dataset(*AMIP_VARIANTS[variant])
    if data_only:
        first_time, n_steps = AMIP_DATA_ONLY_FIRST_TIME, AMIP_DATA_ONLY_N_STEPS
    else:
        first_time, n_steps = AMIP_FIRST_TIME, AMIP_N_STEPS
    return _evaluator_config(
        n_forward_steps=n_steps,
        forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=dataset,
        start_indices={"times": [first_time]},
        data_writer=_files_only([_daily_precip_file()]),
        aggregator=_prescribed_sst_aggregator(),
        prediction_dataset=copy.deepcopy(dataset) if data_only else None,
        slab=False,
        label=AMIP_LABEL,
    )


def build_amip_eval_configs() -> dict[str, dict]:
    variant = AMIP_HELD_OUT_MEMBER
    return {
        paper_config_filename("amip-eval", variant): _amip_evaluator_config(
            variant, data_only=False
        )
    }


def build_amip_p4k_configs() -> dict[str, dict]:
    return {
        paper_config_filename("amip-p4k", "p4k"): _amip_evaluator_config(
            "p4k", data_only=False
        )
    }


def build_amip_p2k_configs() -> dict[str, dict]:
    return {
        paper_config_filename("amip-p2k", "p2k"): _amip_evaluator_config(
            "p2k", data_only=False
        )
    }


def build_amip_data_only_configs() -> dict[str, dict]:
    return {
        paper_config_filename("amip-data-only", variant): _amip_evaluator_config(
            variant, data_only=True
        )
        for variant in AMIP_VARIANTS
    }


def build_random_co2_eval_configs() -> dict[str, dict]:
    configs = {}
    for climate in RAMPED_CLIMATES:
        pattern = (
            f"ramped-sst-{climate}-random-perturbation-{RAMPED_HELD_OUT_MEMBER}.zarr"
        )
        configs[paper_config_filename("random-co2-eval", climate)] = _evaluator_config(
            n_forward_steps=RAMPED_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=_zarr_dataset(RAMPED_DATASET, pattern),
            start_indices={"times": [RAMPED_FIRST_TIME]},
            data_writer=_no_files(),
            aggregator=_prescribed_sst_aggregator(),
            slab=False,
            label=RAMPED_LABEL,
        )
    return configs


BUILDERS = {
    "eq": build_eq_configs,
    "eq-nospinup": build_eq_nospinup_configs,
    "eq-1000yr": build_eq_1000yr_configs,
    "data-only": build_data_only_configs,
    "abrupt-10yr": build_abrupt_10yr_configs,
    "abrupt-10yr-eval": build_abrupt_10yr_eval_configs,
    "abrupt-10yr-eval-sst": build_abrupt_10yr_eval_sst_configs,
    "abrupt-data-only": build_abrupt_data_only_configs,
    "abrupt-ens": build_abrupt_ens_configs,
    "abrupt-ens-data-only": build_abrupt_ens_data_only_configs,
    "7day": build_7day_configs,
    "eq-eval-sst": build_eq_eval_sst_configs,
    "amip-eval": build_amip_eval_configs,
    "amip-p4k": build_amip_p4k_configs,
    "amip-p2k": build_amip_p2k_configs,
    "amip-data-only": build_amip_data_only_configs,
    "random-co2-eval": build_random_co2_eval_configs,
}
assert set(BUILDERS) == set(KINDS)


def references_missing_dataset(config_path: pathlib.Path) -> bool:
    """True if a generated config points at a dataset not yet on weka."""
    text = config_path.read_text()
    return any(
        dataset.path in text
        for dataset in MISSING_DATASETS.values()
        if not dataset.available
    )


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
