"""Submit inference jobs on a checkpoint of the 2026-10-19 and 2026-11-09 runs.

Runs: the training runs of one architecture (--arch, default sfno: nc-sfno
and swin-v2.2: nc-swin-v2.2, their four-source runs, no e3sm; sfno-v2:
nc-sfno-v2 and swin-v2.2-v2: nc-swin-v2.2-v2, the 2026-11-09 runs of all
four data arms; sfno-v2-ext80, swin-v2.2-v2-ext80 and, once X is chosen,
swin-v2.2-v2-lr{X}-e120: the 2026-11-11 follow-ups of the v2 runs,
create_fm_v2_followup_jobs.py; sfno-v2-ft and swin-v2.2-v2-ft: the 2026-11-12
fine-tuning sweep of the ext80 fm-spencer-balanced runs, 48 runs each, read
from run_configs/ft/, create_fm_v2_finetune_jobs.py;
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
              for the e120 runs, ema_ckpt_0008.tar for the ft runs;
              LAST_EMA_EPOCHS)
  ema1..ema7  training_checkpoints/ema_ckpt_000N.tar        (EMA, epoch N;
              the ft archs only, and the eval family only: {run}-emaN.
              Their epoch 8 is lastema, so there is no ema8)
  all         every checkpoint the arch has for each selected family
              (_inference_common.arch_checkpoints): bestinf and lastema,
              and for the ft archs' eval family also ema1..ema7

The ft runs are evaluated as they finish, with one idempotent command per
arch: the eval family at all 9 checkpoints of every ft run whose training
experiment has succeeded, skipping runs still training and jobs already in
Beaker (succeeded or running; failed and canceled ones are resubmitted):

    python submit_inference_jobs.py --arch sfno-v2-ft --family eval \
        --ckpt all --skip-if-in-beaker
    python submit_inference_jobs.py --arch swin-v2.2-v2-ft --family eval \
        --ckpt all --skip-if-in-beaker

and later the other families at bestinf and lastema (the same for
swin-v2.2-v2-ft):

    python submit_inference_jobs.py --arch sfno-v2-ft \
        --family cm4xshield cm4-1pct sst paper-som paper-era5 \
        --ckpt bestinf lastema --skip-if-in-beaker

--ckpt lastema: eval-suite job names swap "-bestinf" for "-lastema"; sst and
paper job names (which carry no checkpoint suffix) get "-lastema" appended.
Every family logs to a "-lastema" wandb group (LASTEMA_WANDB_GROUPS), and the
--ckpt emaN jobs to one "-emaN" group per epoch (e.g.
ace2-fm-eval-ema3-ft-2026-11-12). The
wandb groups of an arch with its own training date (GROUP_DATES; sfno-v2,
swin-v2.2-v2: 2026-11-09) carry that date instead of 2026-10-19
(wandb_groups), and the follow-up archs their phase tag and date (e.g.
ace2-fm-eval-ext80-2026-11-11, ace2-fm-eval-lastema-ext80-2026-11-11);
the two v2 archs share their groups, as sfno and swin-v2.2
do (their job names differ by arch token). Where
the lastema name would pass Beaker's 128-character limit (the balanced runs'
ERA5 paper kinds), only the Beaker experiment name is shortened
(beaker_job_name: "-era5-c96-piC-1pct-cm4-xshield-" becomes "-4src-" and,
for the swin-v2.2-v2-ft runs' longest lastema ERA5 paper names,
"-spencer-balanced-" becomes "-spb-"); the wandb run name stays the full one.
--old-runs (lastema, eval family only) adds the arch's 2026-06-26 baselines
(_inference_common.OLD_RUNS[arch]: era5-a1, fm-a2-cond, c96-a1; nc-sfno for
sfno, nc-swin-v2.1 for swin-v2.2), each on the eval suite of its old -bestinf
job, so their last-epoch EMA scores cover the same entries as their
best-inference ones.

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before a real submission; this is checked unless --dry-run is given.
Every config a submission uses is validated against this checkout's fme
first (--dry-run validates too, so a dry run proves the configs load).

Usage:
    python submit_inference_jobs.py [--arch {sfno,swin-v2.2,sfno-v2,swin-v2.2-v2,
                                            sfno-v2-ext80,swin-v2.2-v2-ext80,
                                            sfno-v2-ft,swin-v2.2-v2-ft}]
                                    [--family FAMILY ...] [--run RUN ...]
                                    [--kind KIND ...] [--sst-grid GRID ...]
                                    [--sst-level LEVEL ...] [--job-name NAME ...]
                                    [--ckpt {bestinf,lastema,ema1..ema7,all} ...]
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
from typing import NamedTuple

from _beaker_listing import DEFAULT_TEXT, fetch_experiments_by_name
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
    OLD_RUNS,
    WANDB_PROJECT,
    arch_checkpoints,
    checkpoint_path,
    checkpoint_suffix,
    finished_runs,
    inference_run_names,
)
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from generate_inference_configs import (
    CM4_XSHIELD_FAMILIES,
    EVAL_SUITE_CONFIGS,
    FAMILIES,
    PAPER_KINDS,
    SST_GRIDS,
    SST_PERTURBATIONS,
    arch_eval_suite_config,
    cm4_xshield_runs,
    paper_config_filename,
    sst_config_filename,
)
from submit_fm_jobs import WANDB_PREFIX, add_default_beaker_args

HERE = pathlib.Path(__file__).parent
EXP_DIR = HERE.parent  # experiment root (configs, records); scripts live in scripts/
RUN_SCRIPT = HERE / "run-ace-inference-job.sh"
CONFIGS_DIRNAME = INFERENCE_CONFIGS_DIR.name

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
# --ckpt value standing for every checkpoint of the arch (arch_checkpoints).
ALL_CHECKPOINTS = "all"
EVAL_SUITE_NAME_SUFFIXES = {
    "eval": "-bestinf",
    "cm4xshield": "-cm4xshield-bestinf",
    "cm4-1pct": "-cm4-1pct-bestinf",
}


class Job(NamedTuple):
    name: str  # Beaker experiment name
    family: str
    config: str  # filename in inference_configs/
    dataset_id: str
    wandb_group: str
    wandb_name: str
    checkpoint: str  # file in the training result dataset


def beaker_job_name(name: str) -> str:
    """`name`, or, if over BEAKER_NAME_LIMIT, `name` with the tokens of
    BEAKER_NAME_SHORTENINGS replaced one after the other until it fits (the
    wandb run name stays `name`).
    """
    short = name
    for long_token, short_token in BEAKER_NAME_SHORTENINGS:
        if len(short) <= BEAKER_NAME_LIMIT:
            break
        short = short.replace(long_token, short_token, 1)
    if len(short) > BEAKER_NAME_LIMIT:
        raise ValueError(f"job name over {BEAKER_NAME_LIMIT} characters: {short}")
    return short


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
        return [(f"{run}{suffix}", arch_eval_suite_config(family, arch))]
    suffix = "" if ckpt == "bestinf" else checkpoint_suffix(ckpt)
    if family == "sst":
        return [
            (f"{run}-sst-{grid}-{level}{suffix}", sst_config_filename(grid, level))
            for grid in SST_GRIDS
            if sst_grids is None or grid in sst_grids
            for level in SST_PERTURBATIONS
            if sst_levels is None or level in sst_levels
        ]
    if family in ("paper-som", "paper-era5"):
        return [
            (f"{run}-{kind}{suffix}", paper_config_filename(kind))
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
    CM4/X-SHiELD families on runs not in cm4_xshield_runs(arch). The job name
    is the wandb run name; beaker_job_name gives the Beaker experiment name.
    """
    cm4_xshield_ok = set(cm4_xshield_runs(arch))
    return [
        (family, run, name, config)
        for family in families
        for run in sorted(runs)
        if family not in CM4_XSHIELD_FAMILIES or run in cm4_xshield_ok
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
) -> tuple[dict[str, str], dict[str, str]]:
    runs = inference_run_names(args.arch)
    if args.run is not None:
        unknown = sorted(set(args.run) - set(runs))
        if unknown:
            raise SystemExit(f"unknown run(s) {unknown}; available: {runs}")
        runs = [r for r in runs if r in args.run]
    if args.use_map:
        mapping = json.loads(BEAKER_MAP_PATH.read_text())
        ready = {r: mapping[r] for r in runs if r in mapping}
        skipped = {
            r: f"not in {BEAKER_MAP_PATH.name}" for r in runs if r not in mapping
        }
        return ready, skipped
    text = listing_text(args.arch)
    print(f"Listing experiments in {args.beaker_workspace} matching {text}...")
    experiments = fetch_experiments_by_name(args.beaker_workspace, text)
    return finished_runs(runs, experiments)


def validate_configs(families_configs: list[tuple[str, str]]) -> None:
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
    if args.old_runs and (args.ckpt != ["lastema"] or args.family != ["eval"]):
        raise SystemExit("--old-runs needs --ckpt lastema and --family eval")
    ckpt_families = checkpoint_families(args.arch, args.ckpt, args.family)
    cm4_xshield_ok = set(cm4_xshield_runs(args.arch))

    ready, skipped = resolve_runs(args)
    print(f"\n{len(ready)} run(s) with a succeeded training job:")
    for run, dataset in sorted(ready.items()):
        print(f"  {run}  {dataset}")
    if skipped:
        print(f"{len(skipped)} run(s) skipped:")
        for run, reason in sorted(skipped.items()):
            print(f"  {run}  ({reason})")
    no_cm4_xshield = sorted(set(ready) - cm4_xshield_ok)
    if no_cm4_xshield and set(args.family) & set(CM4_XSHIELD_FAMILIES):
        families = "/".join(CM4_XSHIELD_FAMILIES)
        print(f"No {families} jobs (not trained on CM4/X-SHiELD):")
        for run in no_cm4_xshield:
            print(f"  {run}")
    print()

    jobs = [
        Job(
            beaker_job_name(name),
            family,
            config,
            ready[run],
            wandb_groups(ckpt, args.arch)[family],
            name,
            checkpoint_path(ckpt, args.arch),
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
            )
            for run, (dataset, config) in sorted(OLD_RUNS[args.arch].items())
        ]
    if args.job_name is not None:
        unknown = sorted(set(args.job_name) - {job.name for job in jobs})
        if unknown:
            raise SystemExit(f"--job-name not among the selected jobs: {unknown}")
        jobs = [job for job in jobs if job.name in args.job_name]
    jobs = drop_jobs_in_beaker(
        jobs, lambda job: job.name, args, [listing_text(args.arch)]
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
    validate_configs(needed)

    for job in jobs:
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


if __name__ == "__main__":
    main()
