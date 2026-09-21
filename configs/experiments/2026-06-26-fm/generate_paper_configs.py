"""Generate the ACE2S-SHiELD+ paper experiment configs for the FM runs.

Reproduces the ACE experiments of the ACE2S-SHiELD+ paper repository
(``ai2cm/ace2s-shield-plus-paper``, ``ACE-experiments/inference``) on the 4deg
daily datasets the FM runs were trained on: the slab-ocean (SOM) climate
experiments on the SHiELD-SOM ensemble, the prescribed-SST experiments on the
AMIP and ramped-SST random-CO2 stores, and their transfer to ERA5 with
prescribed observed SST. Every config is run-agnostic: the checkpoint is
mounted at ``/ckpt.tar`` by submit_paper_jobs.py and the same config is reused
across every training run.

Kind names
----------
A kind is one config family. Its name has six slots, always all present, no
token containing ``-``::

    {data}-{experiment}-{co2}-{shape}-{ocean}-{mode}

data
    The store that supplies SST, sea ice, initial state and reference, and the
    label every config of the kind carries (``som``, ``amip``, ``ramped``,
    ``era5``; a grouped-normalization checkpoint resolves the group it trained
    with and a ``-cond`` checkpoint sees the one-hot it saw). ``som``: the
    SHiELD-SOM equilibrium members; ``somabrupt``: SHiELD's own ten-year
    abrupt-4xCO2 run (MISSING_DATASETS["abrupt"]); ``somabruptens``: SHiELD's
    36-member abrupt-4xCO2 ensemble (MISSING_DATASETS["abrupt-ensemble"]);
    ``amip``, ``amipp4k``, ``amipp2k``: the SHiELD AMIP run on the held-out
    member and SHiELD's AMIP runs with SST +4 K / +2 K; ``ramped``: the
    ramped-climatological-SST random-CO2 runs; ``era5``: ERA5 reanalysis.
experiment
    ``eq`` (equilibrium climate, paper protocol with spin-up),
    ``eqnospinup``, ``abrupt`` (the CO2-step experiment), ``control``
    (forcing exactly as stored).
co2
    ``4xCO2``: the run is at four times the control's CO2, overwritten with
    the constant when the data slot is a control store, read from the store
    when it is an abrupt store. ``dataCO2``: CO2 as stored (the climate's
    constant for SOM members, observed for AMIP and ERA5, random for ramped).
shape
    ``10yr``, ``1000yr``, ``43yr`` (AMIP 1979-2021), ``42yr`` (AMIP from
    1980), ``5yr`` (ramped 2019-10 .. 2024-12), ``ens`` (36 monthly initial
    conditions x 90 days), ``7day`` (the same 36 x 7 days).
ocean
    Where SST and sea ice come from at inference. ``sstslab``: computed by a
    mixed-layer ocean applied through ``stepper_override``, as in the paper,
    from the model's own surface fluxes plus the SOM store's
    ``prescribed_qflux`` and ``prescribed_mixed_layer_depth``;
    ``interpolate`` stays the training default (False) where the paper sets
    True, so the only change from training is the slab itself. SOM data only.
    ``sstprescribed``: read from the data store at every step, as in training
    (no ``stepper_override``). ``sstdata``: no model ran; the reference
    data's own (``dataonly`` mode).
mode
    ``inference`` (free run, no target), ``eval`` (evaluator against the
    data store), ``dataonly`` (the store evaluated against itself, so its
    diagnostics land in the same format; the checkpoint is loaded only for
    variable names).

Training never runs a slab: all 96 training configs prescribe SST from the
data. ``sstprescribed`` kinds are the training-time setup, with CO2
overwritten where the co2 slot says ``4xCO2`` on a control store;
``sstslab`` is the only mode that adds a mechanism the model never saw. The
prescribed-SST 4xCO2 kinds come in pairs that differ only in the data slot:
``som`` (SST held at the control climate while CO2 steps: the direct
atmospheric response, no SHiELD counterpart) versus ``somabrupt`` /
``somabruptens`` (SST from SHiELD's own 4xCO2 run: the response given
SHiELD's surface warming, scorable against it).

Slab-ocean kinds (SHiELD-SOM)
-----------------------------
``som-eq-dataCO2-10yr-sstslab-inference``
    Paper's equilibrium-climate inference, two stages per (climate, ic): a one
    year spin-up from the spin-up dataset's 2030 state writing
    ``/results/spin-up/restart.nc``, then the ten year main run from that
    restart under the climate's own member as forcing (read through
    EQ_FORCING_DIR, see there). Configs ``-spinup-`` and ``-main-``; run by
    run-ace-som-two-stage.sh.
``som-eqnospinup-dataCO2-10yr-sstslab-inference``
    Single-stage variant (ours): the initial condition is the member's own 2031
    state, ten years minus the ic stagger.
``som-eq-dataCO2-1000yr-sstslab-inference``
    Paper's 1000-year run: the 1xCO2 member tiled with ``n_repeats`` (all
    forcing but CO2 is climatological in the SOM runs) with CO2 overwritten to
    the climate's constant; initial condition from the climate's own member.
``som-eq-dataCO2-10yr-sstdata-dataonly``
    Paper's data-only evaluator on every SOM member.
``som-abrupt-4xCO2-10yr-sstslab-inference``
    Ten-year 4xCO2 step from the 1xCO2 member's 2031 state, free (ours).
``somabrupt-abrupt-4xCO2-10yr-sstslab-eval``
    Paper's abrupt-4xCO2 evaluator: the slab run initialized from SHiELD's own
    abrupt-4xCO2 run at 2020-01-01 and scored against it.
``somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly``
    SHiELD's abrupt-4xCO2 run evaluated against itself.
``som-abrupt-4xCO2-ens-sstslab-eval``
    Paper's abrupt-4xCO2 ensemble evaluator: 36 monthly 1xCO2 initial
    conditions, 90 days each, CO2 overwritten to 4x, scored against the 1xCO2
    member so the metrics are the response (figures 8 and 10).
``somabruptens-abrupt-4xCO2-ens-sstdata-dataonly``
    SHiELD's 36-member abrupt-4xCO2 ensemble evaluated against itself. Needs
    the daily 4deg abrupt-4xCO2 ensemble dataset
    (MISSING_DATASETS["abrupt-ensemble"]).
``som-control-dataCO2-7day-sstslab-inference``,
``som-abrupt-4xCO2-7day-sstslab-inference``
    Paper's seven-day ensemble inference from the same 36 initial conditions,
    with the 1xCO2 forcing as is and with CO2 overwritten to 4x (figure 9).

Prescribed-SST kinds (SHiELD)
-----------------------------
``som-eq-dataCO2-10yr-sstprescribed-eval``
    Prescribed-SST control for the equilibrium runs: same climates, members
    and five staggered initial conditions, scored against the member.
    Separates atmospheric error from slab-feedback error.
``somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval``
    The abrupt evaluator with SST from SHiELD's abrupt-4xCO2 run instead of
    the slab: the atmospheric response given SHiELD's own surface warming.
``som-abrupt-4xCO2-10yr-sstprescribed-eval``
    Ten-year 4xCO2 step with SST held at the 1xCO2 member's, scored against
    the 1xCO2 member.
``somabruptens-abrupt-4xCO2-ens-sstprescribed-eval``
    Prescribed-SST figure 8: one evaluator per member of SHiELD's abrupt-4xCO2
    ensemble, SST, sea ice and CO2 from that member, scored against it. Needs
    MISSING_DATASETS["abrupt-ensemble"]; 36 jobs per run.
``som-abrupt-4xCO2-ens-sstprescribed-eval``
    The ensemble CO2 step with SST held at 1xCO2; minus
    ``somabruptens-abrupt-4xCO2-ens-sstprescribed-eval`` it is the SST-mediated
    response.
``som-control-dataCO2-ens-sstprescribed-eval``
    The 36 initial conditions run 90 days on the 1xCO2 member as is: the 1xCO2
    reference line of figure 8 and ACE's control drift.
``amip-control-dataCO2-43yr-sstprescribed-eval``, ``amipp4k-…``, ``amipp2k-…``
    Paper's AMIP runs on the held-out ensemble member ic_0002 and on SHiELD's
    AMIP +4 K / +2 K SST runs, 1979-01-01 to the end of the store, as one
    evaluator each instead of the paper's three chained inference stages
    (spin-up 1979 / train-validate 1980-2011 / test 2012-2020): restart
    chaining gives the identical trajectory, so the windows are an analysis
    cut. Write the paper's daily-PRATEsfc zarr.
``amip-control-dataCO2-42yr-sstdata-dataonly``, ``amipp4k-…``, ``amipp2k-…``
    AMIP ic_0002, +4 K and +2 K evaluated against themselves from 1980-01-01
    (the paper's data-only evaluators skip the spin-up year).
``ramped-control-dataCO2-5yr-sstprescribed-eval``
    Paper's random-CO2 evaluator on the held-out member ic_0003 of the
    ramped-SST random-CO2 runs, 1x/2x/4xCO2, 2019-10-01 to the store's end.

Prescribed-SST kinds (ERA5)
---------------------------
The paper's abrupt-4xCO2 experiments transferred to ERA5, which has no slab
fields, so ``sstprescribed`` only. "4x" is four times ERA5's own global-mean CO2
at the start date (2015-01-01), held constant, as the paper's 4x is four times
the SOM control's constant. The reference is ERA5 itself, whose CO2 keeps
rising through the window, so the response is measured against a slowly
drifting baseline; SHiELD's is flat. ERA5's ten-year control needs no kind:
the eval suites' ``10year`` entry runs the same window.

``era5-abrupt-4xCO2-10yr-sstprescribed-eval``
    2015-01-01, ten years, observed SST, CO2 overwritten to 4x 2015.
``era5-abrupt-4xCO2-ens-sstprescribed-eval``
    36 monthly initial conditions 2015-01 .. 2017-12, 90 days, same CO2.
``era5-control-dataCO2-ens-sstprescribed-eval``
    The same 36 initial conditions with CO2 as observed.

"Held out" means held out of the norm-ablation cells: the hand-written
fm-random-v1/v3 and fm-0.x-v1 runs trained on AMIP ic_0002 and ramped ic_0003
as well. The +2 K/+4 K AMIP runs never appeared in training; ``amip`` is the
closest label and what the SST sweep implies.

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

# The 4deg daily SHiELD-SOM ensemble the FM and c96 regimes train on. Members are
# 1x/2x/4xCO2 ic_0001-ic_0005 and 3xCO2 ic_0001-ic_0002, each 3653 daily steps.
# Every 4deg daily store labels a day's mean at the following 00Z, so the first
# label is 2031-01-02T00 (the mean of 2031-01-01) and the last 2041-01-01T00;
# fme matches initial-condition timestamps exactly, so every start time below
# is such a label.
SOM_LABEL = "som"
SOM_DATASET = (
    "/climate-default/"
    "2026-06-08-vertically-resolved-4deg-daily-c96-shield-som-ensemble-fme-dataset"
)
SOM_FIRST_TIME = "2031-01-02T00:00:00"
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
        kinds=(
            "somabrupt-abrupt-4xCO2-10yr-sstslab-eval",
            "somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval",
            "somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly",
        ),
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
        kinds=("som-eq-dataCO2-10yr-sstslab-inference",),
        available=True,  # copied to weka 2026-09-16
    ),
    "abrupt-ensemble": MissingDataset(
        path=(
            "/climate-default/2026-09-21-vertically-resolved-4deg-daily-c96-shield-"
            "som-abrupt-4xCO2-ensemble-fme-dataset"
        ),
        purpose=(
            "SHiELD's own 36-member abrupt-4xCO2 spread: the data-only rows of "
            "figure 8 and the per-member prescribed-SST evaluators"
        ),
        source=(
            "gs://vcm-ml-raw-flexible-retention/2025-02-03-C96-SHiELD-SOM-abrupt-"
            "4xCO2-ensemble/regridded-zarrs/gaussian_grid_45_by_90/"
            "abrupt-4xCO2-ic_00NN (45x90 regrid of 2026-09-21)"
        ),
        how=(
            "make shield_som_abrupt_4xco2_ensemble_c96_dataset RESOLUTION=4deg in "
            "scripts/data_process (argo), then copy_zarrs_to_weka.py"
        ),
        kinds=(
            "somabruptens-abrupt-4xCO2-ens-sstdata-dataonly",
            "somabruptens-abrupt-4xCO2-ens-sstprescribed-eval",
        ),
        available=True,  # copied to weka 2026-09-21
    ),
}


class Climate(NamedTuple):
    #: Member the paper uses for this climate's forcing, initial conditions
    #: and reference: the held-out ic_0005 where it exists, ic_0002 for 3xCO2.
    member: str
    #: Paper's constant global-mean CO2 (volume mixing ratio) for overwriting
    #: the forcing.
    co2: float


CLIMATES = {
    "1xCO2": Climate(member="ic_0005", co2=0.00036343),
    "2xCO2": Climate(member="ic_0005", co2=0.00072686),
    "3xCO2": Climate(member="ic_0002", co2=0.00109029),
    "4xCO2": Climate(member="ic_0005", co2=0.0014537),
}
CONTROL_CLIMATE = "1xCO2"
# The paper's abrupt experiments step CO2 to 4x only.
ABRUPT_CLIMATE = "4xCO2"

# Paper staggers five initial conditions one timestep apart and runs each as
# its own job; at a daily step the stagger is one day.
N_INITIAL_CONDITIONS = 5
# Paper writes daily precipitation only for the 1x and 3x climates.
DAILY_PRECIP_CLIMATES = ("1xCO2", "3xCO2")

# Spin-up stage: paper starts 2030-01-01 in the spin-up dataset and runs one
# year. The daily spin-up store holds 2030-01-02T00 .. 2031-01-01T00 (365
# labels), so every stagger runs to its last label, 2031-01-01T00, and the
# main stage restarts there. The main member store begins one day later, so
# the main stage's forcing is read from EQ_FORCING_DIR, a weka directory of
# symlinks that lists each climate's spin-up store before its member store
# (XarrayDataConfig concatenates a glob's files in sorted path order).
SPIN_UP_FIRST_TIME = "2030-01-02T00:00:00"
SPIN_UP_LAST_TIME = "2031-01-01T00:00:00"
SPIN_UP_N_STEPS = 364
SPIN_UP_EXPERIMENT_DIR = "/results/spin-up"
EQ_FORCING_DIR = (
    "/climate-default/2026-09-17-vertically-resolved-4deg-daily-c96-shield-som-"
    "eq-spin-up-plus-member-symlinks"
)

# 1000-year run: 365250 daily steps; the 10-year forcing member repeated 101
# times spans 1010 years.
THOUSAND_YEAR_N_STEPS = 365250
THOUSAND_YEAR_N_REPEATS = 101
THOUSAND_YEAR_INITIAL_TIME = "2032-01-01T00:00:00"

# Abrupt-CO2 reference run (paper: 2025-03-28 1deg dataset from 2020-01-01T06,
# 14604 six-hourly steps); the 4deg daily store's first label is 2020-01-02T00.
ABRUPT_FIRST_TIME = "2020-01-02T00:00:00"
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

# Abrupt ensemble: 36 monthly initial conditions (paper: the 1st of each month
# 2031-01 .. 2033-12) run 90 days; at daily labels the 1st's mean is labelled
# the 2nd at 00Z. The seven-day ensembles reuse the same initial conditions.
ENSEMBLE_INITIAL_TIMES = [
    f"{year}-{month:02d}-02T00:00:00"
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
# runs with the SST uniformly +2 K / +4 K. The +2 K / +4 K stores run
# 1979-01-02T00 .. 2021-12-16T00 (15690 daily labels); the ensemble members start
# in 1939 and end the same day.
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
# Paper's AMIP inference starts 1979-01-01 and its data-only evaluator
# 1980-01-01, skipping the spin-up year; both run to the end of the store.
AMIP_FIRST_TIME = "1979-01-02T00:00:00"
AMIP_N_STEPS = 15689
AMIP_DATA_ONLY_FIRST_TIME = "1980-01-01T00:00:00"
AMIP_DATA_ONLY_N_STEPS = 15325

# Ramped-climatological-SST random-CO2 runs: the cells trained on the unnamed
# member and ic_0002 of each climate, ic_0003 is held out. 1919 daily labels
# from 2019-10-02T00; the paper's 7675 six-hourly steps cover the same span.
RAMPED_LABEL = "ramped"
RAMPED_DATASET = (
    "/climate-default/2026-06-08-vertically-resolved-c96-shield-ramped-climSST-"
    "random-CO2-ensemble-fme-dataset-4deg-daily"
)
RAMPED_HELD_OUT_MEMBER = "ic_0003"
RAMPED_CLIMATES = ("1xCO2", "2xCO2", "4xCO2")
RAMPED_FIRST_TIME = "2019-10-02T00:00:00"
RAMPED_N_STEPS = 1918

# ERA5, 4deg daily 1940-2025, labels at 00Z (the training configs' own inline
# inference starts there). The abrupt window is the eval suites' ``10year``
# control window, so the ten-year control run already exists for every run;
# the 36 ensemble initial conditions are its first three years.
ERA5_LABEL = "era5"
ERA5_DATASET = PrescribedSstDataset(
    "/climate-default", "2026-04-17-era5-4deg-8layer-daily-1940-2025.zarr"
)
ERA5_ABRUPT_FIRST_TIME = "2015-01-01T00:00:00"
ERA5_ABRUPT_N_STEPS = 3652
ERA5_ENSEMBLE_INITIAL_TIMES = [
    f"{year}-{month:02d}-01T00:00:00"
    for year in (2015, 2016, 2017)
    for month in range(1, 13)
]
# Four times the store's global_mean_co2 on 2015-01-01 (3.9862e-4, volume
# mixing ratio; Meinshausen et al. 2017 / NOAA GML), as the paper's 4x is four
# times the SOM control's constant. One constant for the whole run and for all
# 36 ensemble members, as in the paper.
ERA5_CO2_2015 = 0.00039861797
ERA5_CO2_4X = 4 * ERA5_CO2_2015

KINDS = (
    # slab-ocean, SHiELD-SOM
    "som-eq-dataCO2-10yr-sstslab-inference",
    "som-eqnospinup-dataCO2-10yr-sstslab-inference",
    "som-eq-dataCO2-1000yr-sstslab-inference",
    "som-eq-dataCO2-10yr-sstdata-dataonly",
    "som-abrupt-4xCO2-10yr-sstslab-inference",
    "somabrupt-abrupt-4xCO2-10yr-sstslab-eval",
    "somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly",
    "som-abrupt-4xCO2-ens-sstslab-eval",
    "somabruptens-abrupt-4xCO2-ens-sstdata-dataonly",
    "som-control-dataCO2-7day-sstslab-inference",
    "som-abrupt-4xCO2-7day-sstslab-inference",
    # prescribed SST, SHiELD
    "som-eq-dataCO2-10yr-sstprescribed-eval",
    "somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval",
    "som-abrupt-4xCO2-10yr-sstprescribed-eval",
    "somabruptens-abrupt-4xCO2-ens-sstprescribed-eval",
    "som-abrupt-4xCO2-ens-sstprescribed-eval",
    "som-control-dataCO2-ens-sstprescribed-eval",
    "amip-control-dataCO2-43yr-sstprescribed-eval",
    "amipp4k-control-dataCO2-43yr-sstprescribed-eval",
    "amipp2k-control-dataCO2-43yr-sstprescribed-eval",
    "amip-control-dataCO2-42yr-sstdata-dataonly",
    "amipp4k-control-dataCO2-42yr-sstdata-dataonly",
    "amipp2k-control-dataCO2-42yr-sstdata-dataonly",
    "ramped-control-dataCO2-5yr-sstprescribed-eval",
    # prescribed SST, ERA5
    "era5-abrupt-4xCO2-10yr-sstprescribed-eval",
    "era5-abrupt-4xCO2-ens-sstprescribed-eval",
    "era5-control-dataCO2-ens-sstprescribed-eval",
)


def kind_mode(kind: str) -> str:
    """``inference``, ``evaluator`` or ``data-only``, from the kind's last token."""
    if kind.endswith("-dataonly"):
        return "data-only"
    if kind.endswith("-eval"):
        return "evaluator"
    if kind.endswith("-inference"):
        return "inference"
    raise ValueError(f"{kind!r} has no mode token")


def kind_grid(kind: str) -> str:
    """Forcing grid of a kind's data: ``era5`` or ``shield`` (SOM, AMIP and
    ramped stores are all SHiELD C96 output).
    """
    return "era5" if kind.startswith("era5-") else "shield"


INFERENCE_KINDS = tuple(k for k in KINDS if kind_mode(k) == "inference")
EVALUATOR_KINDS = tuple(k for k in KINDS if kind_mode(k) != "inference")
# Kinds run once per reference member with a fixed checkpoint, not per run.
DATA_ONLY_KINDS = tuple(k for k in KINDS if kind_mode(k) == "data-only")


def paper_config_filename(kind: str, *parts: str) -> str:
    suffix = "".join(f"-{part}" for part in parts)
    return f"{PAPER_CONFIG_PREFIX}{kind}-config-4deg{suffix}.yaml"


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


def _control_dataset(co2: float | None = None) -> dict:
    """The 1xCO2 paper member, optionally with CO2 overwritten."""
    return _member_dataset(CONTROL_CLIMATE, CLIMATES[CONTROL_CLIMATE].member, co2)


def _abrupt_dataset() -> dict:
    """SHiELD's own abrupt-4xCO2 run (MISSING_DATASETS["abrupt"])."""
    root = MISSING_DATASETS["abrupt"].path
    return _zarr_dataset(root, f"abrupt-{ABRUPT_CLIMATE}.zarr")


def _abrupt_ensemble_member_dataset(member: str) -> dict:
    root = MISSING_DATASETS["abrupt-ensemble"].path
    return _zarr_dataset(root, f"abrupt4xCO2-{member}.zarr")


def _era5_dataset(co2: float | None = None) -> dict:
    return _zarr_dataset(*ERA5_DATASET, co2)


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
    """A free slab-ocean run on SOM data (every inference kind is one)."""
    return {
        "experiment_dir": experiment_dir,
        "n_forward_steps": n_forward_steps,
        "forward_steps_in_memory": forward_steps_in_memory,
        "checkpoint_path": CHECKPOINT_PATH,
        "allow_incompatible_dataset": True,
        "labels": [SOM_LABEL],
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
    slab: bool = False,
    label: str = SOM_LABEL,
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


def _abrupt_monthly_writer() -> dict:
    return {
        "save_prediction_files": False,
        "save_monthly_files": True,
        "names": list(ABRUPT_MONTHLY_NAMES),
    }


def _abrupt_aggregator() -> dict:
    """Paper's aggregator for the abrupt-4xCO2 evaluator."""
    return {"log_zonal_mean_images": False, "log_histograms": True}


def _prescribed_sst_aggregator() -> dict:
    """Paper's aggregator for the AMIP and random-CO2 evaluators."""
    return {"log_zonal_mean_images": False}


def _stagger_time(first_time: str, offset_days: int) -> str:
    """The date `offset_days` after `first_time`, within the same month."""
    date, clock = first_time.split("T")
    year, month, day = date.split("-")
    return f"{year}-{month}-{int(day) + offset_days:02d}T{clock}"


def eq_forcing_symlinks() -> list[tuple[str, str]]:
    """(link name in EQ_FORCING_DIR, target store) for every eq climate: the
    spin-up store sorts before the member store so a ``{climate}-{member}-*``
    glob concatenates them in time order.
    """
    links = []
    spin_up_root = MISSING_DATASETS["spin-up"].path
    for climate, spec in CLIMATES.items():
        links.append(
            (
                f"{climate}-{spec.member}-0-spin-up.zarr",
                f"{spin_up_root}/{climate}-spin-up-{spec.member}.zarr",
            )
        )
        links.append(
            (f"{climate}-{spec.member}-1-main.zarr", member_path(climate, spec.member))
        )
    return links


# --- slab-ocean kinds ---------------------------------------------------------


def build_som_eq_10yr_slab_inference_configs() -> dict[str, dict]:
    kind = "som-eq-dataCO2-10yr-sstslab-inference"
    configs = {}
    spin_up_root = MISSING_DATASETS["spin-up"].path
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            spin_up_pattern = f"{climate}-spin-up-{spec.member}.zarr"
            configs[paper_config_filename(kind, "spinup", climate, f"ic{ic}")] = (
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
            configs[paper_config_filename(kind, "main", climate, f"ic{ic}")] = (
                _inference_config(
                    experiment_dir="/results",
                    n_forward_steps=SOM_N_STEPS,
                    forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
                    initial_condition_path=f"{SPIN_UP_EXPERIMENT_DIR}/restart.nc",
                    initial_condition_engine="netcdf4",
                    initial_times=[SPIN_UP_LAST_TIME],
                    forcing_dataset=_zarr_dataset(
                        EQ_FORCING_DIR, f"{climate}-{spec.member}-*.zarr"
                    ),
                    data_writer=_files_only(_daily_precip_files(climate)),
                )
            )
    return configs


def build_som_eq_nospinup_10yr_slab_inference_configs() -> dict[str, dict]:
    kind = "som-eqnospinup-dataCO2-10yr-sstslab-inference"
    configs = {}
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            configs[paper_config_filename(kind, climate, f"ic{ic}")] = (
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


def build_som_eq_1000yr_slab_inference_configs() -> dict[str, dict]:
    kind = "som-eq-dataCO2-1000yr-sstslab-inference"
    configs = {}
    for climate, spec in CLIMATES.items():
        forcing = _control_dataset(co2=spec.co2)
        forcing["n_repeats"] = THOUSAND_YEAR_N_REPEATS
        configs[paper_config_filename(kind, climate)] = _inference_config(
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


def build_som_eq_10yr_data_only_configs() -> dict[str, dict]:
    kind = "som-eq-dataCO2-10yr-sstdata-dataonly"
    configs = {}
    for climate, members in SOM_MEMBERS.items():
        for member in members:
            dataset = _member_dataset(climate, member)
            configs[paper_config_filename(kind, climate, member)] = _evaluator_config(
                n_forward_steps=SOM_N_STEPS,
                forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
                loader_dataset=dataset,
                start_indices={"times": [SOM_FIRST_TIME]},
                data_writer=_files_only(_daily_precip_files(climate)),
                prediction_dataset=copy.deepcopy(dataset),
            )
    return configs


def build_som_abrupt_4xco2_10yr_slab_inference_configs() -> dict[str, dict]:
    kind = "som-abrupt-4xCO2-10yr-sstslab-inference"
    control = CLIMATES[CONTROL_CLIMATE]
    return {
        paper_config_filename(kind): _inference_config(
            experiment_dir="/results",
            n_forward_steps=SOM_N_STEPS,
            forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
            initial_condition_path=member_path(CONTROL_CLIMATE, control.member),
            initial_condition_engine="zarr",
            initial_times=[SOM_FIRST_TIME],
            forcing_dataset=_control_dataset(co2=CLIMATES[ABRUPT_CLIMATE].co2),
            data_writer=_abrupt_monthly_writer(),
        )
    }


def _abrupt_10yr_eval_config(slab: bool) -> dict:
    """The abrupt-4xCO2 evaluator against SHiELD's own abrupt run."""
    return _evaluator_config(
        n_forward_steps=ABRUPT_N_STEPS,
        forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=_abrupt_dataset(),
        start_indices={"times": [ABRUPT_FIRST_TIME]},
        data_writer=_abrupt_monthly_writer(),
        aggregator=_abrupt_aggregator(),
        slab=slab,
    )


def build_som_abrupt_4xco2_10yr_slab_eval_configs() -> dict[str, dict]:
    kind = "somabrupt-abrupt-4xCO2-10yr-sstslab-eval"
    return {paper_config_filename(kind): _abrupt_10yr_eval_config(slab=True)}


def build_som_abrupt_4xco2_10yr_data_only_configs() -> dict[str, dict]:
    kind = "somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly"
    dataset = _abrupt_dataset()
    return {
        paper_config_filename(kind): _evaluator_config(
            n_forward_steps=ABRUPT_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=dataset,
            start_indices={"times": [ABRUPT_FIRST_TIME]},
            data_writer=_abrupt_monthly_writer(),
            aggregator=_abrupt_aggregator(),
            prediction_dataset=copy.deepcopy(dataset),
        )
    }


def _ensemble_evaluator_config(
    loader_dataset: dict,
    initial_times: list[str],
    slab: bool,
    label: str = SOM_LABEL,
) -> dict:
    """The paper's 36-monthly-IC, 90-day ensemble evaluator (one job)."""
    return _evaluator_config(
        n_forward_steps=ABRUPT_ENSEMBLE_N_STEPS,
        forward_steps_in_memory=ENSEMBLE_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=loader_dataset,
        start_indices={"times": list(initial_times)},
        data_writer=_no_files(),
        slab=slab,
        label=label,
    )


def build_som_abrupt_4xco2_ens_slab_eval_configs() -> dict[str, dict]:
    kind = "som-abrupt-4xCO2-ens-sstslab-eval"
    return {
        paper_config_filename(kind): _ensemble_evaluator_config(
            _control_dataset(co2=CLIMATES[ABRUPT_CLIMATE].co2),
            ENSEMBLE_INITIAL_TIMES,
            slab=True,
        )
    }


def _abrupt_ensemble_member_configs(kind: str, data_only: bool) -> dict[str, dict]:
    """One evaluator per member of SHiELD's abrupt-4xCO2 ensemble."""
    configs = {}
    for n in range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1):
        member = f"ic_{n:04d}"
        dataset = _abrupt_ensemble_member_dataset(member)
        configs[paper_config_filename(kind, member)] = _evaluator_config(
            n_forward_steps=ABRUPT_ENSEMBLE_N_STEPS - 1,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=dataset,
            start_indices={"list": [0]},
            data_writer=_no_files(),
            prediction_dataset=copy.deepcopy(dataset) if data_only else None,
        )
    return configs


def build_som_abrupt_4xco2_ens_data_only_configs() -> dict[str, dict]:
    return _abrupt_ensemble_member_configs(
        "somabruptens-abrupt-4xCO2-ens-sstdata-dataonly", data_only=True
    )


def _seven_day_config(co2: float | None) -> dict:
    control = CLIMATES[CONTROL_CLIMATE]
    return _inference_config(
        experiment_dir="/results",
        n_forward_steps=SEVEN_DAY_N_STEPS,
        forward_steps_in_memory=ENSEMBLE_FORWARD_STEPS_IN_MEMORY,
        initial_condition_path=member_path(CONTROL_CLIMATE, control.member),
        initial_condition_engine="zarr",
        initial_times=list(ENSEMBLE_INITIAL_TIMES),
        forcing_dataset=_control_dataset(co2=co2),
        data_writer=_no_files(),
    )


def build_som_control_7day_slab_inference_configs() -> dict[str, dict]:
    kind = "som-control-dataCO2-7day-sstslab-inference"
    return {paper_config_filename(kind): _seven_day_config(co2=None)}


def build_som_abrupt_4xco2_7day_slab_inference_configs() -> dict[str, dict]:
    kind = "som-abrupt-4xCO2-7day-sstslab-inference"
    return {
        paper_config_filename(kind): _seven_day_config(co2=CLIMATES[ABRUPT_CLIMATE].co2)
    }


# --- prescribed-SST kinds, SHiELD ---------------------------------------------


def build_som_eq_10yr_sst_eval_configs() -> dict[str, dict]:
    kind = "som-eq-dataCO2-10yr-sstprescribed-eval"
    configs = {}
    for climate, spec in CLIMATES.items():
        for ic in range(1, N_INITIAL_CONDITIONS + 1):
            offset = ic - 1
            configs[paper_config_filename(kind, climate, f"ic{ic}")] = (
                _evaluator_config(
                    n_forward_steps=SOM_N_STEPS - offset,
                    forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
                    loader_dataset=_member_dataset(climate, spec.member),
                    start_indices={"times": [_stagger_time(SOM_FIRST_TIME, offset)]},
                    data_writer=_files_only(_daily_precip_files(climate)),
                    aggregator=_prescribed_sst_aggregator(),
                )
            )
    return configs


def build_som_abrupt_4xco2_10yr_sst_eval_configs() -> dict[str, dict]:
    kind = "somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval"
    return {paper_config_filename(kind): _abrupt_10yr_eval_config(slab=False)}


def _abrupt_10yr_sst_fixed_config(
    loader_dataset: dict, first_time: str, n_steps: int, label: str
) -> dict:
    """Ten-year CO2 step with SST held at the control store's, scored against
    the control store.
    """
    return _evaluator_config(
        n_forward_steps=n_steps,
        forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
        loader_dataset=loader_dataset,
        start_indices={"times": [first_time]},
        data_writer=_abrupt_monthly_writer(),
        aggregator=_prescribed_sst_aggregator(),
        label=label,
    )


def build_som_abrupt_4xco2_10yr_sst_fixed_eval_configs() -> dict[str, dict]:
    kind = "som-abrupt-4xCO2-10yr-sstprescribed-eval"
    return {
        paper_config_filename(kind): _abrupt_10yr_sst_fixed_config(
            _control_dataset(co2=CLIMATES[ABRUPT_CLIMATE].co2),
            SOM_FIRST_TIME,
            SOM_N_STEPS,
            SOM_LABEL,
        )
    }


def build_som_abrupt_4xco2_ens_sst_eval_configs() -> dict[str, dict]:
    return _abrupt_ensemble_member_configs(
        "somabruptens-abrupt-4xCO2-ens-sstprescribed-eval", data_only=False
    )


def build_som_abrupt_4xco2_ens_sst_fixed_eval_configs() -> dict[str, dict]:
    kind = "som-abrupt-4xCO2-ens-sstprescribed-eval"
    return {
        paper_config_filename(kind): _ensemble_evaluator_config(
            _control_dataset(co2=CLIMATES[ABRUPT_CLIMATE].co2),
            ENSEMBLE_INITIAL_TIMES,
            slab=False,
        )
    }


def build_som_control_ens_sst_eval_configs() -> dict[str, dict]:
    kind = "som-control-dataCO2-ens-sstprescribed-eval"
    return {
        paper_config_filename(kind): _ensemble_evaluator_config(
            _control_dataset(), ENSEMBLE_INITIAL_TIMES, slab=False
        )
    }


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
        label=AMIP_LABEL,
    )


def build_amip_sst_eval_configs() -> dict[str, dict]:
    kind = "amip-control-dataCO2-43yr-sstprescribed-eval"
    return {
        paper_config_filename(kind, AMIP_HELD_OUT_MEMBER): _amip_evaluator_config(
            AMIP_HELD_OUT_MEMBER, data_only=False
        )
    }


def build_amip_p4k_sst_eval_configs() -> dict[str, dict]:
    kind = "amipp4k-control-dataCO2-43yr-sstprescribed-eval"
    return {paper_config_filename(kind): _amip_evaluator_config("p4k", data_only=False)}


def build_amip_p2k_sst_eval_configs() -> dict[str, dict]:
    kind = "amipp2k-control-dataCO2-43yr-sstprescribed-eval"
    return {paper_config_filename(kind): _amip_evaluator_config("p2k", data_only=False)}


def build_amip_data_only_configs() -> dict[str, dict]:
    kind = "amip-control-dataCO2-42yr-sstdata-dataonly"
    return {
        paper_config_filename(kind, AMIP_HELD_OUT_MEMBER): _amip_evaluator_config(
            AMIP_HELD_OUT_MEMBER, data_only=True
        )
    }


def build_amipp4k_data_only_configs() -> dict[str, dict]:
    kind = "amipp4k-control-dataCO2-42yr-sstdata-dataonly"
    return {paper_config_filename(kind): _amip_evaluator_config("p4k", data_only=True)}


def build_amipp2k_data_only_configs() -> dict[str, dict]:
    kind = "amipp2k-control-dataCO2-42yr-sstdata-dataonly"
    return {paper_config_filename(kind): _amip_evaluator_config("p2k", data_only=True)}


def build_ramped_random_co2_sst_eval_configs() -> dict[str, dict]:
    kind = "ramped-control-dataCO2-5yr-sstprescribed-eval"
    configs = {}
    for climate in RAMPED_CLIMATES:
        pattern = (
            f"ramped-sst-{climate}-random-perturbation-{RAMPED_HELD_OUT_MEMBER}.zarr"
        )
        configs[paper_config_filename(kind, climate)] = _evaluator_config(
            n_forward_steps=RAMPED_N_STEPS,
            forward_steps_in_memory=EVALUATOR_FORWARD_STEPS_IN_MEMORY,
            loader_dataset=_zarr_dataset(RAMPED_DATASET, pattern),
            start_indices={"times": [RAMPED_FIRST_TIME]},
            data_writer=_no_files(),
            aggregator=_prescribed_sst_aggregator(),
            label=RAMPED_LABEL,
        )
    return configs


# --- prescribed-SST kinds, ERA5 -----------------------------------------------


def build_era5_abrupt_4xco2_10yr_sst_fixed_eval_configs() -> dict[str, dict]:
    kind = "era5-abrupt-4xCO2-10yr-sstprescribed-eval"
    return {
        paper_config_filename(kind): _abrupt_10yr_sst_fixed_config(
            _era5_dataset(co2=ERA5_CO2_4X),
            ERA5_ABRUPT_FIRST_TIME,
            ERA5_ABRUPT_N_STEPS,
            ERA5_LABEL,
        )
    }


def build_era5_abrupt_4xco2_ens_sst_fixed_eval_configs() -> dict[str, dict]:
    kind = "era5-abrupt-4xCO2-ens-sstprescribed-eval"
    return {
        paper_config_filename(kind): _ensemble_evaluator_config(
            _era5_dataset(co2=ERA5_CO2_4X),
            ERA5_ENSEMBLE_INITIAL_TIMES,
            slab=False,
            label=ERA5_LABEL,
        )
    }


def build_era5_control_ens_sst_eval_configs() -> dict[str, dict]:
    kind = "era5-control-dataCO2-ens-sstprescribed-eval"
    return {
        paper_config_filename(kind): _ensemble_evaluator_config(
            _era5_dataset(), ERA5_ENSEMBLE_INITIAL_TIMES, slab=False, label=ERA5_LABEL
        )
    }


BUILDERS = {
    "som-eq-dataCO2-10yr-sstslab-inference": (build_som_eq_10yr_slab_inference_configs),
    "som-eqnospinup-dataCO2-10yr-sstslab-inference": (
        build_som_eq_nospinup_10yr_slab_inference_configs
    ),
    "som-eq-dataCO2-1000yr-sstslab-inference": (
        build_som_eq_1000yr_slab_inference_configs
    ),
    "som-eq-dataCO2-10yr-sstdata-dataonly": (build_som_eq_10yr_data_only_configs),
    "som-abrupt-4xCO2-10yr-sstslab-inference": (
        build_som_abrupt_4xco2_10yr_slab_inference_configs
    ),
    "somabrupt-abrupt-4xCO2-10yr-sstslab-eval": (
        build_som_abrupt_4xco2_10yr_slab_eval_configs
    ),
    "somabrupt-abrupt-4xCO2-10yr-sstdata-dataonly": (
        build_som_abrupt_4xco2_10yr_data_only_configs
    ),
    "som-abrupt-4xCO2-ens-sstslab-eval": (build_som_abrupt_4xco2_ens_slab_eval_configs),
    "somabruptens-abrupt-4xCO2-ens-sstdata-dataonly": (
        build_som_abrupt_4xco2_ens_data_only_configs
    ),
    "som-control-dataCO2-7day-sstslab-inference": (
        build_som_control_7day_slab_inference_configs
    ),
    "som-abrupt-4xCO2-7day-sstslab-inference": (
        build_som_abrupt_4xco2_7day_slab_inference_configs
    ),
    "som-eq-dataCO2-10yr-sstprescribed-eval": (build_som_eq_10yr_sst_eval_configs),
    "somabrupt-abrupt-4xCO2-10yr-sstprescribed-eval": (
        build_som_abrupt_4xco2_10yr_sst_eval_configs
    ),
    "som-abrupt-4xCO2-10yr-sstprescribed-eval": (
        build_som_abrupt_4xco2_10yr_sst_fixed_eval_configs
    ),
    "somabruptens-abrupt-4xCO2-ens-sstprescribed-eval": (
        build_som_abrupt_4xco2_ens_sst_eval_configs
    ),
    "som-abrupt-4xCO2-ens-sstprescribed-eval": (
        build_som_abrupt_4xco2_ens_sst_fixed_eval_configs
    ),
    "som-control-dataCO2-ens-sstprescribed-eval": (
        build_som_control_ens_sst_eval_configs
    ),
    "amip-control-dataCO2-43yr-sstprescribed-eval": (build_amip_sst_eval_configs),
    "amipp4k-control-dataCO2-43yr-sstprescribed-eval": (
        build_amip_p4k_sst_eval_configs
    ),
    "amipp2k-control-dataCO2-43yr-sstprescribed-eval": (
        build_amip_p2k_sst_eval_configs
    ),
    "amip-control-dataCO2-42yr-sstdata-dataonly": (build_amip_data_only_configs),
    "amipp4k-control-dataCO2-42yr-sstdata-dataonly": (build_amipp4k_data_only_configs),
    "amipp2k-control-dataCO2-42yr-sstdata-dataonly": (build_amipp2k_data_only_configs),
    "ramped-control-dataCO2-5yr-sstprescribed-eval": (
        build_ramped_random_co2_sst_eval_configs
    ),
    "era5-abrupt-4xCO2-10yr-sstprescribed-eval": (
        build_era5_abrupt_4xco2_10yr_sst_fixed_eval_configs
    ),
    "era5-abrupt-4xCO2-ens-sstprescribed-eval": (
        build_era5_abrupt_4xco2_ens_sst_fixed_eval_configs
    ),
    "era5-control-dataCO2-ens-sstprescribed-eval": (
        build_era5_control_ens_sst_eval_configs
    ),
}
assert set(BUILDERS) == set(KINDS)
assert all(
    kind in KINDS for dataset in MISSING_DATASETS.values() for kind in dataset.kinds
)


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
