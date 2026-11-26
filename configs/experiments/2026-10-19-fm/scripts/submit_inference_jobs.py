"""Submit inference jobs on checkpoints of finished FM training runs.

Runs: the training runs of one architecture (--arch, default sfno: nc-sfno
and swin-v2.2: nc-swin-v2.2, their four-source runs, no e3sm; sfno-v2:
nc-sfno-v2 and swin-v2.2-v2: nc-swin-v2.2-v2, the 2026-11-09 runs of all
four data arms; sfno-v2-ext80, swin-v2.2-v2-ext80 and, once X is chosen,
swin-v2.2-v2-lr{X}-e120: the 2026-11-11 follow-ups of the v2 runs,
create_fm_v2_followup_jobs.py; sfno-v2-ft and swin-v2.2-v2-ft: the 2026-11-12
fine-tuning sweep of the ext80 fm-spencer-balanced runs, 48 runs each, read
from run_configs/ft/, create_fm_v2_finetune_jobs.py; sfno-v2-ft-e32,
sfno-v2-ft-e128, swin-v2.2-v2-ft-e32 and swin-v2.2-v2-ft-e128: its 32- and
128-epoch long fine-tunes, 9 runs each, read from run_configs/ft-e32/ and
run_configs/ft-e128/; sfno-v3: the 18 November 18 per-cell normalization runs;
sfno-v3-grid and swin-v2.2-v3-grid: the 2026-11-25 5-source v3 grid runs,
and sfno-v3-c96amip, swin-v2.2-v3-c96amip, sfno-v3-era5-1979 and
swin-v2.2-v3-era5-1979 its single-source runs (_inference_common
.V3_GRID_ARCHS, V3_SINGLE_SOURCE_ARCHS; checkpoints recorded on Weka in
fm_v3_grid_jobs.json, and fm_v3_jobs.json for the grid's sfno-v3 cells);
_inference_common.ARCHS and inference_run_names; no diagnostics) whose Beaker
experiment succeeded. The cm4xshield and cm4-1pct families skip runs not
trained on CM4 and X-SHiELD (the v2 fm-era5 arms), and use the arch's
suite variant (sfno-v2, swin-v2.2-v2: ...-cm4-xshield-v2.yaml /
...-cm4-1pct-v2.yaml,
generate_inference_configs.CM4_XSHIELD_VARIANTS).
Runs still training are listed as skipped; re-run with --skip-if-in-beaker
once they finish to submit only what is missing. Each job mounts the run's
training_checkpoints/best_inference_ckpt.tar at /ckpt.tar and runs a
run-agnostic config from inference_configs/ (generate_inference_configs.py)
through run-ace-inference-job.sh.

The sfno-v3 runs use the checkpoint paths recorded in fm_v3_jobs.json on
Weka. Evaluate both EMA checkpoints with:

    python submit_inference_jobs.py --arch sfno-v3 --family eval \
        --ckpt bestinf lastema --skip-if-in-beaker

The remaining families use the same checkpoints and flags, with
--family cm4xshield cm4-1pct sst paper-som paper-era5.

The sfno-v2-c96amip and swin-v2.2-v2-c96amip cohort (C96 AMIP-only) and the
separate sfno-v2-era5-1979 and swin-v2.2-v2-era5-1979 cohort (ERA5 over the
same years) also use recorded Weka checkpoints. Their full suite gives every
dataset the run's only trained label (c96 or era5, respectively), including
the out-of-sample CM4/X-SHiELD evaluations.

Families (--family, default all):

  eval        {run}-bestinf                     evaluator suite, 2026-06-26 windows
  cm4xshield  {run}-cm4xshield-bestinf          evaluator suite, CM4/AM4/X-SHiELD
  cm4-1pct    {run}-cm4-1pct-bestinf            evaluator suite, CM4 1pctCO2 140 yr
  sst         {run}-sst-{grid}-{level}          inference, 2 grids x 3 levels
  paper-som   {run}-{kind}                      evaluator, 2 SOM slab kinds
  paper-era5  {run}-{kind}                      evaluator, 4 ERA5 kinds

Checkpoint (--ckpt, one or more, default bestinf;
_inference_common.CHECKPOINTS and EPOCH_CHECKPOINTS):

  bestinf     training_checkpoints/best_inference_ckpt.tar  (EMA weights)
  lastema     training_checkpoints/ema_ckpt_0150.tar        (EMA, last epoch;
              ema_ckpt_0050.tar for sfno-v2 and swin-v2.2-v2,
              ema_ckpt_0080.tar for the ext80 forks, ema_ckpt_0120.tar
              for sfno-v3 and the e120 runs, ema_ckpt_0008.tar for the ft runs,
              ema_ckpt_0032.tar and ema_ckpt_0128.tar for the ft-e32 and
              ft-e128 runs; LAST_EMA_EPOCHS)
  emaN        training_checkpoints/ema_ckpt_{N:04d}.tar     (EMA, epoch N;
              the ft archs only, and the eval family only: {run}-emaN.
              ft: ema1..ema7; ft-e32: ema2, ema4, ema8, ema16; ft-e128:
              ema8, ema16, ema32, ema64
              (_inference_common.EPOCH_CHECKPOINT_ARCHS). An arch's last
              epoch is lastema, so ft has no ema8; an emaN the arch does
              not have is an error)
  all         every checkpoint the arch has for each selected family
              (_inference_common.arch_checkpoints): bestinf and lastema,
              and for the ft archs' eval family also their emaN

The ft runs are evaluated as they finish, with one idempotent command per
arch: the eval family at all 9 checkpoints of every ft run whose training
experiment has succeeded, skipping runs still training and jobs already in
Beaker (succeeded or running; failed and canceled ones are resubmitted):

    python submit_inference_jobs.py --arch sfno-v2-ft --family eval \
        --ckpt all --skip-if-in-beaker
    python submit_inference_jobs.py --arch swin-v2.2-v2-ft --family eval \
        --ckpt all --skip-if-in-beaker

The long fine-tunes likewise, at their 6 eval checkpoints (54 jobs per arch):

    python submit_inference_jobs.py --arch sfno-v2-ft-e32 --family eval \
        --ckpt all --skip-if-in-beaker

and the same for sfno-v2-ft-e128, swin-v2.2-v2-ft-e32 and
swin-v2.2-v2-ft-e128. Later the other families at bestinf and lastema (the
same for the other ft archs):

    python submit_inference_jobs.py --arch sfno-v2-ft \
        --family cm4xshield cm4-1pct sst paper-som paper-era5 \
        --ckpt bestinf lastema --skip-if-in-beaker

--ckpt lastema: eval-suite job names swap "-bestinf" for "-lastema"; sst and
paper job names (which carry no checkpoint suffix) get "-lastema" appended.
Every family logs to a "-lastema" wandb group (LASTEMA_WANDB_GROUPS), and the
--ckpt emaN jobs to one "-emaN" group per epoch (e.g.
ace2-fm-eval-ema3-ft-2026-11-12, ace2-fm-eval-ema16-ft-e32-2026-11-12). The
wandb groups of an arch with its own training date (GROUP_DATES; sfno-v2,
swin-v2.2-v2: 2026-11-09) carry that date instead of 2026-10-19
(wandb_groups), and the follow-up archs their phase tag and date (e.g.
ace2-fm-eval-ext80-2026-11-11, ace2-fm-eval-lastema-ext80-2026-11-11);
the two v2 archs share their groups, as sfno and swin-v2.2
do (their job names differ by arch token). Where
the lastema name would pass Beaker's 128-character limit (the balanced runs'
ERA5 paper kinds), only the Beaker experiment name is shortened
(beaker_job_name: "-era5-c96-piC-1pct-cm4-xshield-" becomes "-4src-" and,
for the ft runs' longest ERA5 paper names (swin-v2.2-v2-ft at lastema, the
long fine-tunes at lastema and, for swin, bestinf), "-spencer-balanced-"
becomes "-spb-"); the wandb run name stays the full one.
--old-runs (lastema, eval family only) adds the arch's 2026-06-26 baselines
(_inference_common.OLD_RUNS[arch]: era5-a1, fm-a2-cond, c96-a1; nc-sfno for
sfno, nc-swin-v2.1 for swin-v2.2), each on the eval suite of its old -bestinf
job, so their last-epoch EMA scores cover the same entries as their
best-inference ones.

On an agent VM (_agent_vm: /etc/agent-sandbox/proxy.toml exists; `beaker`
is a read-only shim and `gantry` submits through the Beaker proxy) only the
archs with Weka checkpoints (WEKA_RECORDS: sfno-v3, the v2 and v3
single-source archs, sfno-v3-grid, swin-v2.2-v3-grid) work, and:
- A training run counts as finished when `gantry status` of its recorded
  experiment id (else `beaker experiment get`, for experiments gantry does
  not know) says succeeded with exit code 0: the experiment-level status,
  i.e. that of its latest job, so a preempted and resumed run counts only
  once its final job succeeded.
- --skip-if-in-beaker looks each job up under its VM name and its real-gantry
  name (job_name_aliases) in `gantry list --all`, the ids recorded in
  records/fm_v3_grid_eval_jobs.json (`gantry status`) and one
  `beaker workspace experiments --text <run>` listing per training run (the
  shim caps a listing at 100; a capped listing is redone job name by job
  name), so jobs submitted elsewhere (e.g. the sfno-v3 cohort's) count too.
- Each job is submitted with /usr/local/bin/gantry run (agent_gantry_command:
  supported flags only, the pushed BRANCH, --weka climate-default, the wandb
  key from --env-secret WANDB_API_KEY=wandb-api-key-ai2cm-sa), runs the
  checkpoint from its /climate-default/agents/alexeyy/... path (passed as
  --checkpoint-path / checkpoint_path override, no dataset mount), writes to
  /results, and is recorded in records/fm_v3_grid_eval_jobs.json (commit
  it). The shim mounts the /climate-default folders the config names (the
  ERA5 store through its "# WEKA_MOUNT:" header). The proxy names the
  experiment agent-alexeyy-<name>-<suffix>, so names are kept to
  AGENT_NAME_LIMIT (109) characters: the longest paper kinds become "-sstp"
  and "-slab" (agent_job_name; the wandb run name stays the full one).
- --dry-run runs `gantry run --dry-run` for every job (checking flags, branch
  and mounts without sending anything) and prints the request of the first.
As each training run finishes, submit exactly its 30 jobs (eval, cm4xshield,
cm4-1pct, sst x 6, paper-som x 2, paper-era5 x 4, at bestinf and lastema):

    cd configs/experiments/2026-10-19-fm && PYTHONPATH=<repo root> \
        python scripts/submit_inference_jobs.py --arch sfno-v3-grid \
        --run <run> --family eval cm4xshield cm4-1pct sst paper-som \
        paper-era5 --ckpt bestinf lastema --skip-if-in-beaker \
        --beaker-priority normal

(a run still training is listed as skipped and nothing is submitted; the same
for swin-v2.2-v3-grid and the four single-source archs). Status:
`python scripts/eval_table.py --arch <arch>`.

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before a real submission; this is checked unless --dry-run is given.
Every config a submission uses is validated against this checkout's fme
first (--dry-run validates too, so a dry run proves the configs load).

Usage:
    python submit_inference_jobs.py [--arch {sfno,swin-v2.2,sfno-v2,swin-v2.2-v2,
                                            sfno-v3,sfno-v2-ext80,swin-v2.2-v2-ext80,
                                            sfno-v2-ft,swin-v2.2-v2-ft,
                                            sfno-v2-ft-e32,sfno-v2-ft-e128,
                                            swin-v2.2-v2-ft-e32,
                                            swin-v2.2-v2-ft-e128,
                                            sfno-v2-c96amip,swin-v2.2-v2-c96amip,
                                            sfno-v2-era5-1979,
                                            swin-v2.2-v2-era5-1979}]
                                    [--family FAMILY ...] [--run RUN ...]
                                    [--kind KIND ...] [--sst-grid GRID ...]
                                    [--sst-level LEVEL ...] [--job-name NAME ...]
                                    [--ckpt {bestinf,lastema,emaN,all} ...]
                                    [--old-runs]
                                    [--use-map] [--dry-run]
                                    [--skip-if-in-beaker] [--exclude-job NAME ...]
                                    [--beaker-workspace WORKSPACE]
                                    [--beaker-cluster CLUSTER [CLUSTER ...]]
                                    [--beaker-priority PRIORITY]
"""

import argparse
import json
import pathlib
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import NamedTuple

from _agent_vm import (
    AGENT_NAME_LIMIT,
    AGENT_PREFIX,
    GANTRY,
    LOOKUP_THREADS,
    exists,
    experiment_job,
    find_experiments,
    on_agent_vm,
    training_status,
)
from _beaker_listing import (
    DEFAULT_TEXT,
    OK,
    experiment_status,
    fetch_experiments_by_name,
    list_experiments,
)
from _inference_common import (
    ARCHS,
    BEAKER_MAP_PATH,
    CHECKPOINTS,
    DEFAULT_ARCH,
    DEFAULT_GROUP_DATE,
    EPOCH_CHECKPOINT_FAMILIES,
    EPOCH_CHECKPOINTS,
    GROUP_DATES,
    INFERENCE_CONFIGS_DIR,
    LAST_EMA_EPOCHS,
    OLD_RUNS,
    SINGLE_SOURCE_ARCH_ARMS,
    V3_ARCH,
    V3_GRID_ARCHS,
    V3_GRID_RECORD_PATH,
    V3_RECORD_PATH,
    V3_SINGLE_SOURCE_ARCHS,
    WANDB_PROJECT,
    arch_checkpoints,
    checkpoint_path,
    checkpoint_suffix,
    finished_runs,
    inference_run_names,
)
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_fm_c96amip_jobs import LABELS as C96AMIP_LABELS
from generate_inference_configs import (
    CM4_XSHIELD_FAMILIES,
    EVAL_SUITE_CONFIGS,
    FAMILIES,
    PAPER_KINDS,
    SST_GRIDS,
    SST_PERTURBATIONS,
    arch_eval_suite_config,
    cm4_xshield_runs,
    is_v3_c3,
    paper_config_filename,
    sst_config_filename,
)
from submit_fm_c96amip_jobs import RECORD_PATH as C96AMIP_RECORD_PATH
from submit_fm_jobs import WANDB_PREFIX, add_default_beaker_args
from submit_fm_v3_jobs import (
    BRANCH,
    REPO_ROOT,
    SUBMITTED_RE,
    WANDB_SECRET,
    WANDB_USERNAME,
    WEKA,
    head_commit,
    image,
    read_record,
    write_record,
)

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
RUN_SCRIPT = HERE / "run-ace-inference-job.sh"
CONFIGS_DIRNAME = INFERENCE_CONFIGS_DIR.name
# Arch -> the submit scripts' records of its runs' Weka checkpoints, the first
# record naming a run winning. The grid archs' sfno-v3 fm-e3sm-balanced
# maskoff cells are in fm_v3_jobs.json.
WEKA_RECORDS = {
    V3_ARCH: (V3_RECORD_PATH,),
    **dict.fromkeys(SINGLE_SOURCE_ARCH_ARMS, (C96AMIP_RECORD_PATH,)),
    **dict.fromkeys(V3_GRID_ARCHS, (V3_GRID_RECORD_PATH, V3_RECORD_PATH)),
    **dict.fromkeys(V3_SINGLE_SOURCE_ARCHS, (V3_GRID_RECORD_PATH,)),
}
# Agent VM (_agent_vm): every inference job submitted through the gantry shim,
# Beaker job name -> experiment, id, url, commit, run, family, checkpoint, ...
EVAL_RECORD_PATH = EXP_DIR / "records" / "fm_v3_grid_eval_jobs.json"
VM_PRIORITIES = ("low", "normal", "high")
VM_MIN_RUNTIME = "4h"
VM_SHARED_MEMORY = "50GiB"

WANDB_GROUPS = {
    "eval": "ace2-fm-eval-2026-10-19",
    "cm4xshield": "ace2-fm-cm4xshield-eval-2026-10-19",
    "cm4-1pct": "ace2-fm-cm4xshield-eval-2026-10-19",
    "sst": "ace2-fm-sst-perts-2026-10-19",
    "paper-som": "ace2-fm-paper-2026-10-19",
    "paper-era5": "ace2-fm-paper-2026-10-19",
}
# wandb groups of the --ckpt lastema jobs, kept apart from the bestinf ones.
LASTEMA_WANDB_GROUPS = {
    "eval": "ace2-fm-eval-lastema-2026-10-19",
    "cm4xshield": "ace2-fm-cm4xshield-eval-lastema-2026-10-19",
    "cm4-1pct": "ace2-fm-cm4xshield-eval-lastema-2026-10-19",
    "sst": "ace2-fm-sst-perts-lastema-2026-10-19",
    "paper-som": "ace2-fm-paper-lastema-2026-10-19",
    "paper-era5": "ace2-fm-paper-lastema-2026-10-19",
}


def wandb_groups(ckpt: str, arch: str) -> dict[str, str]:
    """Family -> wandb group of `arch`'s jobs at `ckpt`: WANDB_GROUPS or
    LASTEMA_WANDB_GROUPS, with the arch's GROUP_DATES date; a per-epoch
    checkpoint (EPOCH_CHECKPOINTS) has the lastema groups with its own name
    for "lastema".
    """
    groups = WANDB_GROUPS if ckpt == "bestinf" else LASTEMA_WANDB_GROUPS
    date = GROUP_DATES.get(arch, DEFAULT_GROUP_DATE)
    tag = ckpt if ckpt in EPOCH_CHECKPOINTS else "lastema"
    return {
        family: group.replace(DEFAULT_GROUP_DATE, date).replace("lastema", tag)
        for family, group in groups.items()
    }


def listing_text(arch: str) -> str:
    """Beaker `--text` filter covering `arch`'s training runs and their
    inference jobs. The v2 archs (GROUP_DATES) list only their own names
    (the default "ace2-fm-" matches ~9k experiments and takes many minutes);
    the other archs keep the default, which also covers their --old-runs
    baselines.
    """
    if arch in SINGLE_SOURCE_ARCH_ARMS:
        return f"{WANDB_PREFIX}{ARCHS[arch]}-{SINGLE_SOURCE_ARCH_ARMS[arch]}-"
    if arch in GROUP_DATES:
        return f"{WANDB_PREFIX}{ARCHS[arch]}-fm-"
    return DEFAULT_TEXT


# run-ace-inference-job.sh MODE per family.
MODES = {
    "eval": "suite",
    "cm4xshield": "suite",
    "cm4-1pct": "suite",
    "sst": "inference",
    "paper-som": "evaluator",
    "paper-era5": "evaluator",
}
# Beaker rejects experiment names over 128 characters.
BEAKER_NAME_LIMIT = 128
LONG_SOURCES_TOKEN = "-era5-c96-piC-1pct-cm4-xshield-"
SHORT_SOURCES_TOKEN = "-4src-"
# Tokens beaker_job_name shortens, in order, until the name fits. The second
# is for the ft runs (their names have no LONG_SOURCES_TOKEN).
BEAKER_NAME_SHORTENINGS = (
    (LONG_SOURCES_TOKEN, SHORT_SOURCES_TOKEN),
    ("-spencer-balanced-", "-spb-"),
)
# On an agent VM the proxy adds 19 characters to the name (AGENT_NAME_LIMIT),
# so the v3 grid's longest paper names (up to 119 characters) are shortened
# further, by their kind only (not the run name, which the per-run listings
# of agent_listing_groups search for).
AGENT_NAME_SHORTENINGS = (
    *BEAKER_NAME_SHORTENINGS,
    ("-sstprescribed-eval", "-sstp"),
    ("-sstslab-eval", "-slab"),
)
# --ckpt value standing for every checkpoint of the arch (arch_checkpoints).
ALL_CHECKPOINTS = "all"
EVAL_SUITE_NAME_SUFFIXES = {
    "eval": "-bestinf",
    "cm4xshield": "-cm4xshield-bestinf",
    "cm4-1pct": "-cm4-1pct-bestinf",
}


class Job(NamedTuple):
    name: str  # Beaker experiment name (the --name given to gantry)
    family: str
    config: str  # filename in inference_configs/
    dataset_id: str
    wandb_group: str
    wandb_name: str
    checkpoint: str  # path on Weka or inside the training result dataset
    run: str = ""  # training run
    ckpt: str = ""  # --ckpt choice


class TrainingOutput(NamedTuple):
    dataset_id: str
    weka_checkpoints: dict[str, str] | None = None

    def checkpoint(self, ckpt: str, arch: str) -> str:
        if self.weka_checkpoints is not None:
            return self.weka_checkpoints[ckpt]
        return checkpoint_path(ckpt, arch)


def beaker_job_name(
    name: str,
    limit: int = BEAKER_NAME_LIMIT,
    shortenings: tuple[tuple[str, str], ...] = BEAKER_NAME_SHORTENINGS,
) -> str:
    """`name`, or, if over `limit`, `name` with the tokens of `shortenings`
    replaced one after the other until it fits (the wandb run name stays
    `name`).
    """
    short = name
    for long_token, short_token in shortenings:
        if len(short) <= limit:
            break
        short = short.replace(long_token, short_token, 1)
    if len(short) > limit:
        raise ValueError(f"job name over {limit} characters: {short}")
    return short


def agent_job_name(name: str) -> str:
    """The --name of `name`'s job on an agent VM (AGENT_NAME_LIMIT)."""
    return beaker_job_name(name, AGENT_NAME_LIMIT, AGENT_NAME_SHORTENINGS)


def job_name(name: str) -> str:
    """The Beaker job name of wandb run name `name` on this machine."""
    return agent_job_name(name) if on_agent_vm() else beaker_job_name(name)


def job_name_aliases(name: str) -> set[str]:
    """Every Beaker job name `name`'s job may have been submitted under: on an
    agent VM or with the real gantry.
    """
    return {agent_job_name(name), beaker_job_name(name)}


def agent_listing_groups(jobs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Beaker listing text -> the job_name_aliases it finds, for (training
    run, wandb run name) `jobs`: one listing per run, and one per alias that
    does not contain its run (a run name shortened by BEAKER_NAME_SHORTENINGS).
    """
    groups: dict[str, set[str]] = {}
    for run, name in jobs:
        for alias in job_name_aliases(name):
            groups.setdefault(run if run in alias else alias, set()).add(alias)
    return groups


def family_jobs(
    family: str,
    run: str,
    kinds: list[str] | None = None,
    sst_grids: list[str] | None = None,
    sst_levels: list[str] | None = None,
    ckpt: str = "bestinf",
    arch: str = DEFAULT_ARCH,
) -> list[tuple[str, str]]:
    """(job name, config filename) of one run's jobs in one family, narrowed
    by `kinds` (paper families) and `sst_grids`/`sst_levels` (sst). `ckpt`
    sets the eval-suite job-name suffix and, unless bestinf, the suffix
    appended to sst and paper job names (checkpoint_suffix). `arch` picks the
    CM4/X-SHiELD suite variant (arch_eval_suite_config).
    """
    if family in EVAL_SUITE_CONFIGS:
        suffix = EVAL_SUITE_NAME_SUFFIXES[family].replace(
            "-bestinf", checkpoint_suffix(ckpt)
        )
        return [(f"{run}{suffix}", arch_eval_suite_config(family, arch, run))]
    suffix = "" if ckpt == "bestinf" else checkpoint_suffix(ckpt)
    c3 = is_v3_c3(arch, run)
    if family == "sst":
        return [
            (
                f"{run}-sst-{grid}-{level}{suffix}",
                sst_config_filename(grid, level, c3=c3),
            )
            for grid in SST_GRIDS
            if sst_grids is None or grid in sst_grids
            for level in SST_PERTURBATIONS
            if sst_levels is None or level in sst_levels
        ]
    if family in ("paper-som", "paper-era5"):
        return [
            (f"{run}-{kind}{suffix}", paper_config_filename(kind, c3=c3))
            for kind, (kind_family, _) in PAPER_KINDS.items()
            if kind_family == family and (kinds is None or kind in kinds)
        ]
    raise ValueError(f"unknown family {family!r}")


def select_jobs(
    families: Iterable[str],
    runs: Iterable[str],
    ckpt: str = "bestinf",
    arch: str = DEFAULT_ARCH,
    kinds: list[str] | None = None,
    sst_grids: list[str] | None = None,
    sst_levels: list[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """(family, run, job name, config filename) of every job of `families`
    on `runs` (sorted), family by family: family_jobs, without the
    CM4/X-SHiELD families on runs not in cm4_xshield_runs(arch), except for the
    explicitly out-of-sample evaluations of the single-source runs (C96
    AMIP-only and ERA5-1979). The job name
    is the wandb run name; beaker_job_name gives the Beaker experiment name.
    """
    cm4_xshield_ok = set(cm4_xshield_runs(arch))
    return [
        (family, run, name, config)
        for family in families
        for run in sorted(runs)
        if family not in CM4_XSHIELD_FAMILIES
        or arch in SINGLE_SOURCE_ARCH_ARMS
        or run in cm4_xshield_ok
        for name, config in family_jobs(
            family, run, kinds, sst_grids, sst_levels, ckpt, arch
        )
    ]


def checkpoint_families(
    arch: str, ckpts: Iterable[str], families: Iterable[str]
) -> dict[str, list[str]]:
    """Checkpoint -> the families of `families` that `arch` has jobs in at it
    (arch_checkpoints), for each of `ckpts` in order; ALL_CHECKPOINTS stands
    for every checkpoint of the arch. A checkpoint named outright that has
    none of `families` (a per-epoch one on another arch, or without the eval
    family) is an error; under ALL_CHECKPOINTS it is left out.
    """
    families = list(families)
    selected: dict[str, list[str]] = {}
    for ckpt in ckpts:
        expanded = arch_checkpoints(arch) if ckpt == ALL_CHECKPOINTS else [ckpt]
        for name in expanded:
            at = [f for f in families if name in arch_checkpoints(arch, f)]
            if at:
                selected[name] = at
            elif ckpt != ALL_CHECKPOINTS:
                raise SystemExit(
                    f"--ckpt {name}: {arch} has no {'/'.join(families)} jobs at it "
                    f"(per-epoch checkpoints: {arch_checkpoints(arch)[2:] or 'none'}"
                    f", families {'/'.join(EPOCH_CHECKPOINT_FAMILIES)} only)"
                )
    return selected


def resolve_runs(
    args: argparse.Namespace,
) -> tuple[dict[str, TrainingOutput], dict[str, str]]:
    runs = inference_run_names(args.arch)
    if args.run is not None:
        unknown = sorted(set(args.run) - set(runs))
        if unknown:
            raise SystemExit(f"unknown run(s) {unknown}; available: {runs}")
        runs = [r for r in runs if r in args.run]
    if args.arch in WEKA_RECORDS:
        return _resolve_weka_runs(runs, args.beaker_workspace, args.arch)
    if args.use_map:
        mapping = json.loads(BEAKER_MAP_PATH.read_text())
        ready = {r: TrainingOutput(mapping[r]) for r in runs if r in mapping}
        skipped = {
            r: f"not in {BEAKER_MAP_PATH.name}" for r in runs if r not in mapping
        }
        return ready, skipped
    text = listing_text(args.arch)
    print(f"Listing experiments in {args.beaker_workspace} matching {text}...")
    experiments = fetch_experiments_by_name(args.beaker_workspace, text)
    datasets, skipped = finished_runs(runs, experiments)
    return {run: TrainingOutput(dataset) for run, dataset in datasets.items()}, skipped


def _resolve_weka_runs(
    runs: list[str], workspace: str, arch: str
) -> tuple[dict[str, TrainingOutput], dict[str, str]]:
    record_paths = WEKA_RECORDS[arch]
    records: dict[str, dict] = {}
    for record_path in reversed(record_paths):
        if record_path.exists():
            records.update(json.loads(record_path.read_text()))
    record_names = " / ".join(path.name for path in record_paths)
    ids = [records[run]["id"] for run in runs if run in records]
    statuses = _recorded_statuses(ids, workspace, arch)
    ready = {}
    skipped = {}
    for run in runs:
        record = records.get(run)
        if record is None:
            skipped[run] = f"not in {record_names}"
            continue
        exp_id = record["id"]
        if exp_id not in statuses:
            skipped[run] = f"recorded Beaker experiment {exp_id} not found"
            continue
        status = statuses[exp_id]
        if status != OK:
            skipped[run] = f"Beaker status {status} ({exp_id})"
            continue
        ready[run] = TrainingOutput(
            "-",
            {
                "bestinf": record["best_inference_ckpt"],
                "lastema": record["ema_ckpt_pattern"].format(
                    epoch=LAST_EMA_EPOCHS[arch]
                ),
            },
        )
    return ready, skipped


def _recorded_statuses(ids: list[str], workspace: str, arch: str) -> dict[str, str]:
    """Experiment id -> status (_beaker_listing) of each of `ids` found. On an
    agent VM, from `gantry status` per id (else `beaker experiment get`):
    the experiment-level status, ok only if its latest job exited 0
    (_agent_vm.training_status). Elsewhere, from one listing of the arch.
    """
    if on_agent_vm():
        print(f"Looking up {len(ids)} recorded experiment(s) with gantry status...")
        with ThreadPoolExecutor(LOOKUP_THREADS) as ex:
            jobs = dict(zip(ids, ex.map(experiment_job, ids)))
        return {i: training_status(job) for i, job in jobs.items() if job is not None}
    text = listing_text(arch)
    print(f"Listing experiments in {workspace} matching {text}...")
    return {
        exp["id"]: experiment_status(exp) for exp in list_experiments(workspace, text)
    }


def agent_existing_jobs(jobs: list[Job], workspace: str) -> set[str]:
    """Names of `jobs` already succeeded, running or queued in Beaker, seen
    from an agent VM: under either of their job_name_aliases, in
    `gantry list --all`, EVAL_RECORD_PATH's ids (`gantry status`) or one
    beaker shim listing per training run (_agent_vm.find_experiments).
    """
    if not jobs:
        return set()
    aliases = {job.name: job_name_aliases(job.wandb_name) for job in jobs}
    groups = agent_listing_groups((job.run, job.wandb_name) for job in jobs)
    recorded = {n: e["id"] for n, e in read_record(EVAL_RECORD_PATH).items()}
    print(
        f"Looking up {len(jobs)} job(s) in gantry list, {EVAL_RECORD_PATH.name} "
        f"and {len(groups)} beaker listing(s) in {workspace}..."
    )
    found = find_experiments(groups, recorded, workspace)
    return {
        name
        for name, names in aliases.items()
        if exists(job for alias in names for job in found.get(alias, ()))
    }


def agent_gantry_command(
    job: Job,
    dataset_label: str | None,
    priority: str,
    clusters: list[str],
    dry_run: bool,
) -> list[str]:
    """The gantry shim command of one inference job on an agent VM: as
    run-ace-inference-job.sh for a Weka checkpoint, with only flags the proxy
    supports (no --workspace, --budget or --dataset-secret: the proxy sets the
    first two, and the configs read no GCS data), on the pushed BRANCH. The
    checkpoint is passed as an override of its /climate-default path, so the
    shim mounts the agent's own folder and the data folders the config names.
    Outputs go to /results (the configs' experiment_dir).
    """
    config = str((INFERENCE_CONFIGS_DIR / job.config).relative_to(REPO_ROOT))
    mode = MODES[job.family]
    if mode == "suite":
        script = str((HERE / "run_eval_suite.py").relative_to(REPO_ROOT))
        command = ["python", "-I", script, config, "--checkpoint-path", job.checkpoint]
        if dataset_label is not None:
            command += ["--dataset-label", dataset_label]
    else:
        command = ["python", "-I", "-m", f"fme.ace.{mode}", config]
        command += ["--override", f"checkpoint_path={job.checkpoint}"]
        if dataset_label is not None:
            command += [f"labels=[{dataset_label}]"]
    cmd = [GANTRY, "run", "--branch", BRANCH]
    cmd += ["--name", job.name, "--task-name", job.name]
    cmd += ["--description", f"Run ACE FM {mode} on a Weka checkpoint"]
    cmd += ["--beaker-image", image(), "--priority", priority]
    cmd += ["--min-runtime", VM_MIN_RUNTIME]
    for cluster in clusters:
        cmd += ["--cluster", cluster]
    env = {
        "WANDB_USERNAME": WANDB_USERNAME,
        "WANDB_NAME": job.wandb_name,
        "WANDB_JOB_TYPE": "inference",
        "WANDB_RUN_GROUP": job.wandb_group,
        "WANDB_PROJECT": WANDB_PROJECT,
    }
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--env-secret", WANDB_SECRET]
    cmd += ["--gpus", "1", "--shared-memory", VM_SHARED_MEMORY]
    cmd += ["--weka", WEKA]
    cmd += ["--system-python", "--install", "pip install --no-deps ."]
    # Jobs run the pushed commit; uncommitted edits in this worktree do not
    # reach them (check_configs_at_head covers the configs).
    cmd += ["--allow-dirty"]
    if dry_run:
        cmd += ["--dry-run"]
    return [*cmd, "--", *command]


def submit_agent_job(
    job: Job,
    arch: str,
    dataset_label: str | None,
    args: argparse.Namespace,
    show_request: bool = True,
) -> None:
    """Submit (or, with --dry-run, have the shim print) one job through the
    gantry shim, and record a submission in EVAL_RECORD_PATH.
    """
    cmd = agent_gantry_command(
        job, dataset_label, args.beaker_priority, args.beaker_cluster, args.dry_run
    )
    print("Submitting:", subprocess.list2cmdline(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    # A dry run's request (JSON) is printed for the first job only.
    if not args.dry_run or proc.returncode != 0 or show_request:
        sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"gantry exited {proc.returncode} for {job.name}")
    if args.dry_run:
        return
    match = SUBMITTED_RE.search(proc.stdout)
    if match is None:
        raise SystemExit(f"could not parse the gantry output for {job.name}")
    assert match["name"].startswith(AGENT_PREFIX + job.name), match
    record = read_record(EVAL_RECORD_PATH)
    record[job.name] = {
        "experiment": match["name"],
        "id": match["id"],
        "url": match["url"],
        "commit": head_commit(),
        "priority": args.beaker_priority,
        "submitted": datetime.now(UTC).isoformat(timespec="seconds"),
        "arch": arch,
        "run": job.run,
        "family": job.family,
        "ckpt": job.ckpt,
        "config": job.config,
        "checkpoint": job.checkpoint,
        "wandb_name": job.wandb_name,
        "wandb_group": job.wandb_group,
        **({"dataset_label": dataset_label} if dataset_label is not None else {}),
    }
    write_record(record, EVAL_RECORD_PATH)


def validate_configs(
    families_configs: list[tuple[str, str]], dataset_label: str | None = None
) -> None:
    """Validate each (family, config filename) against this checkout's fme."""
    for family, config in families_configs:
        path = str(INFERENCE_CONFIGS_DIR / config)
        mode = MODES[family]
        if mode == "suite":
            cmd = [
                sys.executable,
                str(HERE / "run_eval_suite.py"),
                "--validate-only",
                path,
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "fme.ace.validate_config",
                "--config_type",
                mode,
                path,
            ]
        if dataset_label is not None:
            cmd += (
                ["--dataset-label", dataset_label]
                if mode == "suite"
                else ["--override", f"labels=[{dataset_label}]"]
            )
        subprocess.run(cmd, check=True)
        print(f"Validated ({mode}): {config}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arch",
        choices=list(ARCHS),
        default=DEFAULT_ARCH,
        help=f"Architecture of the training runs (default: {DEFAULT_ARCH}).",
    )
    parser.add_argument(
        "--family",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        help="Job families to submit (default: all).",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these training run names (default: all finished).",
    )
    parser.add_argument(
        "--use-map",
        action="store_true",
        help=(
            f"Take result datasets from {BEAKER_MAP_PATH.name} "
            "(update_beaker_map.py) instead of a live Beaker listing."
        ),
    )
    parser.add_argument(
        "--kind",
        nargs="+",
        choices=list(PAPER_KINDS),
        default=None,
        help="paper-som/paper-era5: only these kinds (default: all of the family).",
    )
    parser.add_argument(
        "--sst-grid",
        nargs="+",
        choices=list(SST_GRIDS),
        default=None,
        help="sst: only these forcing grids (default: all).",
    )
    parser.add_argument(
        "--sst-level",
        nargs="+",
        choices=list(SST_PERTURBATIONS),
        default=None,
        help="sst: only these perturbation levels (default: all).",
    )
    parser.add_argument(
        "--job-name",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only these exact job names (applied after the other filters).",
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        choices=[*CHECKPOINTS, *EPOCH_CHECKPOINTS, ALL_CHECKPOINTS],
        default=["bestinf"],
        help=(
            "Checkpoint(s) to evaluate (default: bestinf). The per-epoch ones "
            f"({', '.join(EPOCH_CHECKPOINTS)}) are for the ft archs' eval family "
            f"only; {ALL_CHECKPOINTS}: every checkpoint the arch has for each "
            "selected family."
        ),
    )
    parser.add_argument(
        "--old-runs",
        action="store_true",
        help=(
            "Also submit the arch's 2026-06-26 baselines (OLD_RUNS[arch]) on "
            "their old eval suites; needs --ckpt lastema and --family eval."
        ),
    )
    # Same Beaker defaults as the training submitter (submit_fm_jobs.py).
    add_default_beaker_args(parser)
    args = parser.parse_args()
    agent_vm = on_agent_vm()
    if agent_vm and args.arch not in WEKA_RECORDS:
        parser.error(
            f"{args.arch} reads its checkpoints from Beaker result datasets, which "
            "the agent VM's beaker shim cannot list; on an agent VM only the archs "
            f"with Weka checkpoints work: {', '.join(WEKA_RECORDS)}"
        )
    if agent_vm and args.beaker_priority not in VM_PRIORITIES:
        parser.error(f"--beaker-priority must be one of {VM_PRIORITIES} on an agent VM")
    if args.arch in WEKA_RECORDS and args.use_map:
        records = " / ".join(path.name for path in WEKA_RECORDS[args.arch])
        parser.error(f"{args.arch} uses {records} for checkpoints")
    if args.old_runs and (args.ckpt != ["lastema"] or args.family != ["eval"]):
        raise SystemExit("--old-runs needs --ckpt lastema and --family eval")
    if args.old_runs and args.arch not in OLD_RUNS:
        parser.error(f"--old-runs is unavailable for {args.arch}")
    ckpt_families = checkpoint_families(args.arch, args.ckpt, args.family)
    cm4_xshield_ok = set(cm4_xshield_runs(args.arch))

    ready, skipped = resolve_runs(args)
    print(f"\n{len(ready)} run(s) with a succeeded training job:")
    for run, output in sorted(ready.items()):
        source = "Weka" if output.weka_checkpoints is not None else output.dataset_id
        print(f"  {run}  {source}")
    if skipped:
        print(f"{len(skipped)} run(s) skipped:")
        for run, reason in sorted(skipped.items()):
            print(f"  {run}  ({reason})")
    no_cm4_xshield = sorted(set(ready) - cm4_xshield_ok)
    if (
        args.arch not in SINGLE_SOURCE_ARCH_ARMS
        and no_cm4_xshield
        and set(args.family) & set(CM4_XSHIELD_FAMILIES)
    ):
        families = "/".join(CM4_XSHIELD_FAMILIES)
        print(f"No {families} jobs (not trained on CM4/X-SHiELD):")
        for run in no_cm4_xshield:
            print(f"  {run}")
    print()

    jobs = [
        Job(
            job_name(name),
            family,
            config,
            ready[run].dataset_id,
            wandb_groups(ckpt, args.arch)[family],
            name,
            ready[run].checkpoint(ckpt, args.arch),
            run,
            ckpt,
        )
        for ckpt, families in ckpt_families.items()
        for family, run, name, config in select_jobs(
            families,
            ready,
            ckpt,
            args.arch,
            args.kind,
            args.sst_grid,
            args.sst_level,
        )
    ]
    if args.old_runs:
        (ckpt,) = ckpt_families
        suffix = checkpoint_suffix(ckpt)
        jobs += [
            Job(
                f"{run}{suffix}",
                "eval",
                config,
                dataset,
                wandb_groups(ckpt, args.arch)["eval"],
                f"{run}{suffix}",
                checkpoint_path(ckpt, args.arch),
                run,
                ckpt,
            )
            for run, (dataset, config) in sorted(OLD_RUNS[args.arch].items())
        ]
    if args.job_name is not None:
        unknown = sorted(set(args.job_name) - {job.name for job in jobs})
        if unknown:
            raise SystemExit(f"--job-name not among the selected jobs: {unknown}")
        jobs = [job for job in jobs if job.name in args.job_name]
    jobs = drop_jobs_in_beaker(
        jobs,
        lambda job: job.name,
        args,
        [listing_text(args.arch)],
        existing=(
            (lambda: agent_existing_jobs(jobs, args.beaker_workspace))
            if agent_vm
            else None
        ),
    )
    if not jobs:
        print("Nothing to submit.")
        return

    needed = sorted({(job.family, job.config) for job in jobs})
    for _, config in needed:
        if not (INFERENCE_CONFIGS_DIR / config).exists():
            raise FileNotFoundError(
                f"{config} not found - run generate_inference_configs.py first"
            )
    if not args.dry_run:
        check_configs_at_head(
            [INFERENCE_CONFIGS_DIR / config for config in {c for _, c in needed}]
        )
    dataset_label = (
        C96AMIP_LABELS[SINGLE_SOURCE_ARCH_ARMS[args.arch]]
        if args.arch in SINGLE_SOURCE_ARCH_ARMS
        else None
    )
    validate_configs(needed, dataset_label)

    for i, job in enumerate(jobs):
        if agent_vm:
            submit_agent_job(job, args.arch, dataset_label, args, show_request=i == 0)
            continue
        submit_job(
            RUN_SCRIPT,
            [
                MODES[job.family],
                f"{CONFIGS_DIRNAME}/{job.config}",
                job.name,
                job.wandb_group,
                job.dataset_id,
                job.checkpoint,
                job.wandb_name,
                *([dataset_label] if dataset_label is not None else []),
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=EXP_DIR,
            extra_env={"SKIP_VALIDATE": "1"},
        )

    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.family] = counts.get(job.family, 0) + 1
    verb = "would be submitted" if args.dry_run else "submitted"
    summary = ", ".join(f"{family}: {n}" for family, n in counts.items())
    print(f"\n{len(jobs)} job(s) {verb} ({summary})")
    for checkpoint in dict.fromkeys(job.checkpoint for job in jobs):
        print(f"Checkpoint: {checkpoint}")
    for family, group in dict.fromkeys((job.family, job.wandb_group) for job in jobs):
        print(f"wandb group ({family}): {group}")
    longest = max(jobs, key=lambda job: len(job.name))
    print(f"Longest job name ({len(longest.name)} chars): {longest.name}")
    if agent_vm and not args.dry_run:
        print(f"Recorded in {EVAL_RECORD_PATH.relative_to(REPO_ROOT)}; commit it.")


if __name__ == "__main__":
    main()
