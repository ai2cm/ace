"""Submit the ACE2S-SHiELD+ paper experiment jobs for the FM checkpoints.

Runs the configs written by generate_paper_configs.py (see its docstring for the
kinds and their naming scheme) against every training run with a result
dataset in wandb_to_beaker_map.json whose regime saw the kind's forcing data,
mounting that run's best_inference_ckpt.tar at /ckpt.tar:

- SHiELD-data kinds (``som-``, ``amip-``, ``ramped-``): the fm and c96 regimes.
- ERA5-data kinds (``era5-``): the fm and era5 regimes plus the hand-written
  ERA5 runs (``nc-sfno-vN``), which have no regime segment.

Data-only kinds evaluate reference data against itself and run once per
reference member with a single checkpoint (--data-only-run) rather than once
per training run.

--kind is required; there is no default, since the full set is several
thousand jobs. --arm restricts to the norm-ablation cells (dropping the
hand-written runs), and --run/--arch/--regime narrow further. --climate applies
to the kinds that span climates (``som-eq-*`` and ``ramped-*``), --ic to the
staggered-IC kinds (``som-eq-dataCO2-10yr-*``, ``som-eqnospinup-*``), and
--ens-member to the per-member abrupt-4xCO2 ensemble kinds
(``somabruptens-*``), which otherwise expand to all 36 members.

Job names are the wandb run names: ``{run}-{kind}[-{climate}][-ic{n}]`` for
the per-run kinds and ``{kind}-{member}`` for the data-only kinds.

--skip-if-in-beaker drops every job whose name already has a succeeded or
running experiment in the workspace (failed and canceled ones are resubmitted),
so re-running a submission only fills in what is missing; the listing is
filtered on the ``ace2-fm-`` prefix, so data-only jobs are never skipped. A
gantry call that fails no longer aborts the batch: the job is reported and the
rest continue, and the script exits non-zero at the end.

Kinds whose configs point at a dataset that is not on weka yet (the entries of
generate_paper_configs.MISSING_DATASETS with available=False) are refused with a
pointer to MISSING_DATASETS.md unless --allow-missing-datasets is given.

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before submitting; this is checked unless --dry-run is given.

Usage:
    python submit_paper_jobs.py --kind KIND [KIND ...]
                              [--run RUN ...] [--arch ARCH ...]
                              [--regime {fm,c96,era5} ...] [--arm {a1,a2,a3} ...]
                              [--climate CLIMATE ...] [--ic IC ...]
                              [--ens-member N ...]
                              [--version {v1,v2,v3}]
                              [--data-only-run RUN]
                              [--skip-if-in-wandb] [--skip-if-in-beaker]
                              [--exclude-job NAME ...] [--allow-missing-datasets]
                              [--dry-run]
                              [--beaker-workspace WORKSPACE]
                              [--beaker-cluster CLUSTER [CLUSTER ...]]
                              [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

from _submit_common import (
    add_beaker_args,
    check_configs_at_head,
    drop_jobs_in_beaker,
    submit_job,
)
from _version_select import add_version_arg
from generate_eval_configs import (
    ARCHITECTURES,
    BASE_CONFIGS_DIR,
    RUN_CONFIGS_DIR,
    TRAINING_RESULT_DATASETS,
    WANDB_ENTITY,
    WANDB_PREFIX,
    WANDB_PROJECT,
    discover_source_configs,
    fetch_wandb_finished_summaries,
    source_config_to_run_name,
)
from generate_paper_configs import (
    ABRUPT_ENSEMBLE_N_MEMBERS,
    AMIP_HELD_OUT_MEMBER,
    CLIMATES,
    DATA_ONLY_KINDS,
    KINDS,
    MISSING_DATASETS,
    N_INITIAL_CONDITIONS,
    RAMPED_CLIMATES,
    SOM_MEMBERS,
    kind_grid,
    kind_mode,
    paper_config_filename,
    references_missing_dataset,
)

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parents[2]
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
INFERENCE_RUN_SCRIPT = HERE / "run-ace-inference.sh"
EVALUATOR_RUN_SCRIPT = HERE / "run-ace-evaluator.sh"
TWO_STAGE_RUN_SCRIPT = HERE / "run-ace-som-two-stage.sh"
WANDB_GROUP = "ace2-fm-paper-2026-06-26"
# best_inference_ckpt.tar is always written by training; mounted at /ckpt.tar.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"

# Training regime (segment after the architecture tag in the run name; "" for
# the hand-written ERA5 runs) -> forcing grids whose data the regime trained
# on. c96 cells never saw ERA5; era5 cells never saw SHiELD; fm cells saw both.
REGIME_GRIDS = {
    "fm": ("shield", "era5"),
    "c96": ("shield",),
    "era5": ("era5",),
    "": ("era5",),
}
REGIMES = ("fm", "c96", "era5")
ARMS = ("a1", "a2", "a3")
DEFAULT_DATA_ONLY_RUN = "ace2-fm-nc-swin-v2-fm-a1"

TWO_STAGE_KIND = "som-eq-dataCO2-10yr-sstslab-inference"
CLIMATE_IC_KINDS = (
    "som-eq-dataCO2-10yr-sstslab-inference",
    "som-eqnospinup-dataCO2-10yr-sstslab-inference",
    "som-eq-dataCO2-10yr-sstprescribed-eval",
)
ENSEMBLE_MEMBER_KINDS = (
    "somabruptens-abrupt-4xCO2-ens-sstprescribed-eval",
    "somabruptens-abrupt-4xCO2-ens-sstdata-dataonly",
)


class Job(NamedTuple):
    name: str
    run_script: pathlib.Path
    #: Config filenames (relative to run_configs/) in the order the run script
    #: takes them: one for single-stage kinds, spin-up then main for the
    #: two-stage equilibrium kind.
    configs: tuple[str, ...]
    dataset_id: str


def run_regime(run_name: str) -> str:
    """The training regime segment after the architecture tag
    (``nc-sfno-fm-a1`` -> ``fm``, ``nc-swin-v2-c96-a3-cond`` -> ``c96``); the
    hand-written ERA5 runs (``nc-sfno-v2``) have none and yield ``""``.
    """
    suffix = run_name.removeprefix(WANDB_PREFIX)
    for arch in ARCHITECTURES:
        if suffix.startswith(f"{arch}-"):
            return suffix.removeprefix(f"{arch}-").split("-", 1)[0]
    return ""


def run_arm(run_name: str) -> str | None:
    """The norm-ablation arm of a cell (``a1``/``a2``/``a3``), None for the
    hand-written runs.
    """
    match = re.search(r"-(a[123])(?:-|$)", run_name)
    return match.group(1) if match else None


def paper_runs(version: str | None = None) -> list[str]:
    """Every base or norm-ablation training run with a recorded result dataset,
    in any regime; runs_for_kind narrows per kind.
    """
    runs = []
    for source_path in discover_source_configs(
        version,
        architectures=ARCHITECTURES,
        source_dirs=(BASE_CONFIGS_DIR, RUN_CONFIGS_DIR),
    ):
        run_name = source_config_to_run_name(source_path.name)
        if run_name not in TRAINING_RESULT_DATASETS:
            print(f"Skipped {source_path.name} (no dataset ID for {run_name!r})")
            continue
        runs.append(run_name)
    return runs


def runs_for_kind(kind: str, runs: list[str]) -> list[str]:
    """The runs whose regime trained on the kind's forcing grid."""
    grid = kind_grid(kind)
    return [name for name in runs if grid in REGIME_GRIDS.get(run_regime(name), ())]


def _ic_member_tag(member: str) -> str:
    """``ic_0005`` -> ``ic5``, for job names."""
    return f"ic{int(member.removeprefix('ic_'))}"


def _variant_tag(variant: str) -> str:
    """``ic_0002`` -> ``ic2``, ``p4k`` -> ``p4k``, for job names."""
    return _ic_member_tag(variant) if variant.startswith("ic_") else variant


def _run_script(kind: str) -> pathlib.Path:
    if kind == TWO_STAGE_KIND:
        return TWO_STAGE_RUN_SCRIPT
    if kind_mode(kind) == "inference":
        return INFERENCE_RUN_SCRIPT
    return EVALUATOR_RUN_SCRIPT


def _expand(
    kind: str, climates: list[str], ics: list[int], ens_members: list[int]
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """(job-name parts, config filenames) for every job of a kind."""
    if kind in CLIMATE_IC_KINDS:
        if kind == TWO_STAGE_KIND:
            return [
                (
                    (climate, f"ic{ic}"),
                    (
                        paper_config_filename(kind, "spinup", climate, f"ic{ic}"),
                        paper_config_filename(kind, "main", climate, f"ic{ic}"),
                    ),
                )
                for climate in climates
                for ic in ics
            ]
        return [
            ((climate, f"ic{ic}"), (paper_config_filename(kind, climate, f"ic{ic}"),))
            for climate in climates
            for ic in ics
        ]
    if kind == "som-eq-dataCO2-1000yr-sstslab-inference":
        return [
            ((climate,), (paper_config_filename(kind, climate),))
            for climate in climates
        ]
    if kind == "som-eq-dataCO2-10yr-sstdata-dataonly":
        return [
            (
                (climate, _ic_member_tag(member)),
                (paper_config_filename(kind, climate, member),),
            )
            for climate in climates
            for member in SOM_MEMBERS[climate]
        ]
    if kind == "ramped-control-dataCO2-5yr-sstprescribed-eval":
        return [
            ((climate,), (paper_config_filename(kind, climate),))
            for climate in climates
            if climate in RAMPED_CLIMATES
        ]
    if kind in ENSEMBLE_MEMBER_KINDS:
        return [
            (
                (_ic_member_tag(f"ic_{n:04d}"),),
                (paper_config_filename(kind, f"ic_{n:04d}"),),
            )
            for n in ens_members
        ]
    if kind in (
        "amip-control-dataCO2-43yr-sstprescribed-eval",
        "amip-control-dataCO2-42yr-sstdata-dataonly",
    ):
        return [
            (
                (_variant_tag(AMIP_HELD_OUT_MEMBER),),
                (paper_config_filename(kind, AMIP_HELD_OUT_MEMBER),),
            )
        ]
    # Single-config kinds.
    return [((), (paper_config_filename(kind),))]


def model_jobs(
    kind: str,
    run_name: str,
    climates: list[str],
    ics: list[int],
    ens_members: list[int],
) -> list[Job]:
    """Jobs of a per-training-run kind for one run."""
    dataset_id = TRAINING_RESULT_DATASETS[run_name]
    return [
        Job("-".join((run_name, kind, *parts)), _run_script(kind), configs, dataset_id)
        for parts, configs in _expand(kind, climates, ics, ens_members)
    ]


def data_only_jobs(
    kind: str,
    data_only_run: str,
    climates: list[str],
    ens_members: list[int],
) -> list[Job]:
    """Jobs of a data-only kind: one per reference member, fixed checkpoint."""
    dataset_id = TRAINING_RESULT_DATASETS[data_only_run]
    return [
        Job("-".join((kind, *parts)), EVALUATOR_RUN_SCRIPT, configs, dataset_id)
        for parts, configs in _expand(kind, climates, [], ens_members)
    ]


def refuse_missing_datasets(kinds: list[str], config_filenames: list[str]) -> None:
    """Exit if any needed config points at a dataset not yet on weka."""
    stale = [
        name
        for name in config_filenames
        if references_missing_dataset(RUN_CONFIGS_DIR / name)
    ]
    if not stale:
        return
    lines = [
        "Refusing to submit: these configs reference datasets that are not on "
        "weka yet. Produce the dataset, mark it available in "
        "generate_paper_configs.MISSING_DATASETS, regenerate and commit; see "
        "MISSING_DATASETS.md. Pass --allow-missing-datasets to submit anyway.",
    ]
    for key, dataset in MISSING_DATASETS.items():
        needed = sorted(set(kinds) & set(dataset.kinds))
        if needed:
            lines.append(f"  {key}: {dataset.path}")
            lines.append(f"    needed by {', '.join(needed)}: {dataset.purpose}")
    lines.append("Configs:")
    lines.extend(f"  {name}" for name in stale)
    raise SystemExit("\n".join(lines))


def config_kind(config_filename: str) -> str:
    return config_filename.removeprefix("ace-paper-").split("-config-")[0]


def validate_configs(config_filenames: list[str]) -> None:
    for config_filename in config_filenames:
        mode = kind_mode(config_kind(config_filename))
        config_type = "inference" if mode == "inference" else "evaluator"
        # Validate against this checkout's fme, not the installed package:
        # gantry runs the configs from this repository at HEAD.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fme.ace.validate_config",
                "--config_type",
                config_type,
                str(RUN_CONFIGS_DIR / config_filename),
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        nargs="+",
        required=True,
        choices=KINDS,
        help="Experiment kinds to submit (see generate_paper_configs.py).",
    )
    add_version_arg(parser)
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Restrict to these training run names (default: all eligible).",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        default=None,
        choices=ARCHITECTURES,
        help="Restrict to these architecture tags (default: all).",
    )
    parser.add_argument(
        "--regime",
        nargs="+",
        default=None,
        choices=REGIMES,
        help=(
            "Restrict to these training regimes (default: every regime that "
            "trained on the kind's forcing data)."
        ),
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        default=None,
        choices=ARMS,
        help=(
            "Restrict to norm-ablation cells of these arms; drops the "
            "hand-written runs (default: no arm filter)."
        ),
    )
    parser.add_argument(
        "--climate",
        nargs="+",
        default=None,
        choices=list(CLIMATES),
        help=(
            "Restrict the climate-spanning kinds to these climates (default: "
            "all a kind applies to)."
        ),
    )
    parser.add_argument(
        "--ic",
        nargs="+",
        type=int,
        default=None,
        choices=range(1, N_INITIAL_CONDITIONS + 1),
        help="Restrict the staggered-IC kinds to these initial conditions.",
    )
    parser.add_argument(
        "--ens-member",
        nargs="+",
        type=int,
        default=None,
        choices=range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1),
        metavar="N",
        help=(
            "Restrict the per-member abrupt-4xCO2 ensemble kinds to these "
            "members (default: all 36)."
        ),
    )
    parser.add_argument(
        "--data-only-run",
        default=DEFAULT_DATA_ONLY_RUN,
        help=(
            "Training run whose checkpoint the data-only kinds load "
            f"(default: {DEFAULT_DATA_ONLY_RUN})."
        ),
    )
    parser.add_argument(
        "--skip-if-in-wandb",
        action="store_true",
        help=(
            "Skip each job whose name already has a finished run in wandb, so "
            "a resubmission only fills in what is missing."
        ),
    )
    parser.add_argument(
        "--allow-missing-datasets",
        action="store_true",
        help="Submit kinds whose configs point at datasets not yet on weka.",
    )
    add_beaker_args(
        parser,
        default_workspace="ai2/ace",
        default_cluster=["ai2/titan", "ai2/jupiter"],
        default_priority="normal",
    )
    args = parser.parse_args()

    climates = args.climate or list(CLIMATES)
    ics = args.ic or list(range(1, N_INITIAL_CONDITIONS + 1))
    ens_members = args.ens_member or list(range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1))

    runs = paper_runs(args.version)
    if args.run is not None:
        unknown_runs = sorted(set(args.run) - set(runs))
        if unknown_runs:
            raise KeyError(
                f"unknown training run(s) {unknown_runs} — available: {sorted(runs)}"
            )
        runs = [name for name in runs if name in args.run]
    if args.arch is not None:
        runs = [
            name
            for name in runs
            if any(
                name.removeprefix(WANDB_PREFIX).startswith(f"{arch}-")
                for arch in args.arch
            )
        ]
    if args.regime is not None:
        runs = [name for name in runs if run_regime(name) in args.regime]
    if args.arm is not None:
        runs = [name for name in runs if run_arm(name) in args.arm]

    if args.data_only_run not in TRAINING_RESULT_DATASETS:
        raise KeyError(f"no dataset ID for --data-only-run {args.data_only_run!r}")

    jobs: list[Job] = []
    used_runs: set[str] = set()
    for kind in args.kind:
        if kind in DATA_ONLY_KINDS:
            jobs.extend(data_only_jobs(kind, args.data_only_run, climates, ens_members))
        else:
            for run_name in runs_for_kind(kind, runs):
                used_runs.add(run_name)
                jobs.extend(model_jobs(kind, run_name, climates, ics, ens_members))

    if args.skip_if_in_wandb:
        print(f"Fetching finished runs from {WANDB_ENTITY}/{WANDB_PROJECT}...")
        finished = set(fetch_wandb_finished_summaries())
        pending = []
        for job in jobs:
            if job.name in finished:
                print(f"Skipping (already finished in wandb): {job.name}")
            else:
                pending.append(job)
        print(f"{len(jobs) - len(pending)} skipped, {len(pending)} to submit.")
        jobs = pending

    jobs = drop_jobs_in_beaker(jobs, lambda job: job.name, args)

    needed_configs = sorted({name for job in jobs for name in job.configs})
    for config_filename in needed_configs:
        if not (RUN_CONFIGS_DIR / config_filename).exists():
            raise FileNotFoundError(
                f"{config_filename} not found — run generate_paper_configs.py first"
            )
    if not args.allow_missing_datasets:
        refuse_missing_datasets(args.kind, needed_configs)

    if not args.dry_run:
        check_configs_at_head([RUN_CONFIGS_DIR / name for name in needed_configs])
        validate_configs(needed_configs)

    print(f"{len(jobs)} job(s) across {len(used_runs)} training run(s).")
    failed_submissions = []
    for job in jobs:
        try:
            submit_job(
                job.run_script,
                [
                    *(f"{RUN_CONFIGS_DIRNAME}/{name}" for name in job.configs),
                    job.name,
                    WANDB_GROUP,
                    job.dataset_id,
                    CHECKPOINT_PATH,
                ],
                wandb_project=WANDB_PROJECT,
                args=args,
                cwd=HERE,
                extra_env={"SKIP_VALIDATE": "1"},
            )
        except (subprocess.CalledProcessError, OSError) as err:
            print(f"SUBMISSION FAILED ({err}): {job.name}")
            failed_submissions.append(job.name)
    if failed_submissions:
        print(f"{len(failed_submissions)} submission(s) failed:")
        print("\n".join(f"  {name}" for name in failed_submissions))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
