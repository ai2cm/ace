"""Submit the ACE2S-SHiELD+ paper experiment jobs for the FM checkpoints.

Runs the configs written by generate_paper_configs.py (see its docstring for the
kinds) against every fm- and c96-regime training run with a result dataset in
wandb_to_beaker_map.json, mounting that run's best_inference_ckpt.tar at
/ckpt.tar. The era5 regime and the hand-written ERA5 runs are skipped: they
never saw the SOM, AMIP or ramped data or their labels. Data-only kinds
evaluate reference data against itself and run once per reference member with
a single checkpoint (--data-only-run) rather than once per training run.

--kind is required; there is no default, since the full set is well over a
thousand jobs. --arm restricts to the norm-ablation cells (dropping the
hand-written runs), and --run/--arch/--regime/--climate/--ic narrow further.
--climate applies to the SOM kinds and random-co2-eval (1x/2x/4x); --ic to
eq, eq-nospinup and eq-eval-sst; --ens-member to the per-member abrupt-4xCO2
ensemble kinds (abrupt-ens-eval-sst, abrupt-ens-data-only), which otherwise
expand to all 36 members.

Job names are the wandb run names: ``{run}-som-{kind}-{climate}[-ic{n}]`` for
the SOM kinds, ``{run}-amip-{variant}-eval`` and ``{run}-ramped-{climate}-eval``
for the prescribed-SST kinds, and ``som-``/``amip-``-prefixed names without a
run for the data-only kinds.

Kinds whose configs point at a dataset that is not on weka yet (the entries of
generate_paper_configs.MISSING_DATASETS with available=False) are refused with a
pointer to MISSING_DATASETS.md unless --allow-missing-datasets is given.

Gantry clones the repository at HEAD, so the configs must be committed and
pushed before submitting; this is checked unless --dry-run is given.

Usage:
    python submit_paper_jobs.py --kind KIND [KIND ...]
                              [--run RUN ...] [--arch ARCH ...]
                              [--regime {fm,c96} ...] [--arm {a1,a2,a3} ...]
                              [--climate CLIMATE ...] [--ic IC ...]
                              [--ens-member N ...]
                              [--version {v1,v2,v3}]
                              [--data-only-run RUN]
                              [--skip-if-in-wandb] [--allow-missing-datasets]
                              [--dry-run]
                              [--beaker-workspace WORKSPACE]
                              [--beaker-cluster CLUSTER [CLUSTER ...]]
                              [--beaker-priority PRIORITY]
"""

import argparse
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

from _submit_common import add_beaker_args, check_configs_at_head, submit_job
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
    ABRUPT_CLIMATES,
    ABRUPT_ENSEMBLE_N_MEMBERS,
    AMIP_HELD_OUT_MEMBER,
    AMIP_VARIANTS,
    CLIMATES,
    CONTROL_CLIMATE,
    DATA_ONLY_KINDS,
    EVALUATOR_KINDS,
    KINDS,
    MISSING_DATASETS,
    N_INITIAL_CONDITIONS,
    RAMPED_CLIMATES,
    SOM_MEMBERS,
    paper_config_filename,
    references_missing_dataset,
)

HERE = pathlib.Path(__file__).parent
RUN_CONFIGS_DIRNAME = RUN_CONFIGS_DIR.name
INFERENCE_RUN_SCRIPT = HERE / "run-ace-inference.sh"
EVALUATOR_RUN_SCRIPT = HERE / "run-ace-evaluator.sh"
TWO_STAGE_RUN_SCRIPT = HERE / "run-ace-som-two-stage.sh"
WANDB_GROUP = "ace2-fm-paper-2026-06-26"
# best_inference_ckpt.tar is always written by training; mounted at /ckpt.tar.
CHECKPOINT_PATH = "training_checkpoints/best_inference_ckpt.tar"

REGIMES = ("fm", "c96")
ARMS = ("a1", "a2", "a3")
DEFAULT_DATA_ONLY_RUN = "ace2-fm-nc-swin-v2-fm-a1"


class Job(NamedTuple):
    name: str
    run_script: pathlib.Path
    #: Config filenames (relative to run_configs/) in the order the run script
    #: takes them: one for single-stage kinds, spin-up then main for ``eq``.
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


def som_runs(version: str | None = None) -> list[str]:
    """Training run names the SOM experiments apply to: every base or
    norm-ablation config with a recorded result dataset in the fm or c96 regime.
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
        if run_regime(run_name) in REGIMES:
            runs.append(run_name)
    return runs


def _ic_member_tag(member: str) -> str:
    """``ic_0005`` -> ``ic5``, for job names."""
    return f"ic{int(member.removeprefix('ic_'))}"


def model_jobs(
    kind: str,
    run_name: str,
    climates: list[str],
    ics: list[int],
    ens_members: list[int],
) -> list[Job]:
    """Jobs of a per-training-run kind for one run."""
    dataset_id = TRAINING_RESULT_DATASETS[run_name]
    jobs = []
    if kind == "eq":
        for climate in climates:
            for ic in ics:
                jobs.append(
                    Job(
                        f"{run_name}-som-eq-{climate}-ic{ic}",
                        TWO_STAGE_RUN_SCRIPT,
                        (
                            paper_config_filename("eq-spinup", climate, f"ic{ic}"),
                            paper_config_filename("eq-main", climate, f"ic{ic}"),
                        ),
                        dataset_id,
                    )
                )
    elif kind == "eq-nospinup":
        for climate in climates:
            for ic in ics:
                jobs.append(
                    Job(
                        f"{run_name}-som-eq-nospinup-{climate}-ic{ic}",
                        INFERENCE_RUN_SCRIPT,
                        (paper_config_filename(kind, climate, f"ic{ic}"),),
                        dataset_id,
                    )
                )
    elif kind == "eq-1000yr":
        for climate in climates:
            jobs.append(
                Job(
                    f"{run_name}-som-eq1000-{climate}",
                    INFERENCE_RUN_SCRIPT,
                    (paper_config_filename(kind, climate),),
                    dataset_id,
                )
            )
    elif kind == "abrupt-10yr":
        for climate in climates:
            if climate in ABRUPT_CLIMATES:
                jobs.append(
                    Job(
                        f"{run_name}-som-abrupt-{climate}-10yr",
                        INFERENCE_RUN_SCRIPT,
                        (paper_config_filename(kind, climate),),
                        dataset_id,
                    )
                )
    elif kind in ("abrupt-10yr-eval", "abrupt-10yr-eval-sst"):
        suffix = kind.removeprefix("abrupt-")
        for climate in climates:
            if climate in ABRUPT_CLIMATES:
                jobs.append(
                    Job(
                        f"{run_name}-som-abrupt-{climate}-{suffix}",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, climate),),
                        dataset_id,
                    )
                )
    elif kind == "abrupt-ens":
        if "4xCO2" in climates:
            jobs.append(
                Job(
                    f"{run_name}-som-abrupt-4xCO2-ens",
                    EVALUATOR_RUN_SCRIPT,
                    (paper_config_filename(kind, "4xCO2"),),
                    dataset_id,
                )
            )
    elif kind == "7day":
        for climate in climates:
            if climate in (CONTROL_CLIMATE, "4xCO2"):
                jobs.append(
                    Job(
                        f"{run_name}-som-7day-{climate}",
                        INFERENCE_RUN_SCRIPT,
                        (paper_config_filename(kind, climate),),
                        dataset_id,
                    )
                )
    elif kind == "eq-eval-sst":
        for climate in climates:
            for ic in ics:
                jobs.append(
                    Job(
                        f"{run_name}-som-eq-eval-sst-{climate}-ic{ic}",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, climate, f"ic{ic}"),),
                        dataset_id,
                    )
                )
    elif kind in ("amip-eval", "amip-p4k", "amip-p2k"):
        variant = AMIP_HELD_OUT_MEMBER if kind == "amip-eval" else kind[5:]
        jobs.append(
            Job(
                f"{run_name}-amip-{_amip_variant_tag(variant)}-eval",
                EVALUATOR_RUN_SCRIPT,
                (paper_config_filename(kind, variant),),
                dataset_id,
            )
        )
    elif kind == "random-co2-eval":
        for climate in climates:
            if climate in RAMPED_CLIMATES:
                jobs.append(
                    Job(
                        f"{run_name}-ramped-{climate}-eval",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, climate),),
                        dataset_id,
                    )
                )
    elif kind == "abrupt-ens-eval-sst":
        if "4xCO2" in climates:
            for n in ens_members:
                member = f"ic_{n:04d}"
                jobs.append(
                    Job(
                        f"{run_name}-som-abrupt-4xCO2-ens-eval-sst-{_ic_member_tag(member)}",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, "4xCO2", member),),
                        dataset_id,
                    )
                )
    elif kind == "control-ens-eval-sst":
        if CONTROL_CLIMATE in climates:
            jobs.append(
                Job(
                    f"{run_name}-som-control-ens-eval-sst",
                    EVALUATOR_RUN_SCRIPT,
                    (paper_config_filename(kind, CONTROL_CLIMATE),),
                    dataset_id,
                )
            )
    elif kind == "abrupt-ens-fixed-sst":
        if "4xCO2" in climates:
            jobs.append(
                Job(
                    f"{run_name}-som-abrupt-4xCO2-ens-fixed-sst",
                    EVALUATOR_RUN_SCRIPT,
                    (paper_config_filename(kind, "4xCO2"),),
                    dataset_id,
                )
            )
    else:
        raise ValueError(f"{kind!r} is not a per-run kind")
    return jobs


def _amip_variant_tag(variant: str) -> str:
    """``ic_0002`` -> ``ic2``, ``p4k`` -> ``p4k``, for job names."""
    return _ic_member_tag(variant) if variant.startswith("ic_") else variant


def data_only_jobs(
    kind: str, data_only_run: str, climates: list[str], ens_members: list[int]
) -> list[Job]:
    """Jobs of a data-only kind: one per reference member, fixed checkpoint."""
    dataset_id = TRAINING_RESULT_DATASETS[data_only_run]
    jobs = []
    if kind == "data-only":
        for climate in climates:
            for member in SOM_MEMBERS[climate]:
                jobs.append(
                    Job(
                        f"som-data-only-{climate}-{_ic_member_tag(member)}",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, climate, member),),
                        dataset_id,
                    )
                )
    elif kind == "abrupt-data-only":
        for climate in climates:
            if climate in ABRUPT_CLIMATES:
                jobs.append(
                    Job(
                        f"som-abrupt-{climate}-data-only",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, climate),),
                        dataset_id,
                    )
                )
    elif kind == "abrupt-ens-data-only":
        if "4xCO2" in climates:
            for n in ens_members:
                member = f"ic_{n:04d}"
                jobs.append(
                    Job(
                        f"som-abrupt-4xCO2-ens-data-only-{_ic_member_tag(member)}",
                        EVALUATOR_RUN_SCRIPT,
                        (paper_config_filename(kind, "4xCO2", member),),
                        dataset_id,
                    )
                )
    elif kind == "amip-data-only":
        for variant in AMIP_VARIANTS:
            jobs.append(
                Job(
                    f"amip-{_amip_variant_tag(variant)}-data-only",
                    EVALUATOR_RUN_SCRIPT,
                    (paper_config_filename(kind, variant),),
                    dataset_id,
                )
            )
    else:
        raise ValueError(f"{kind!r} is not a data-only kind")
    return jobs


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


def validate_configs(config_filenames: list[str]) -> None:
    for config_filename in config_filenames:
        kind = config_filename.removeprefix("ace-paper-").split("-config-")[0]
        config_type = "evaluator" if kind in EVALUATOR_KINDS else "inference"
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
        help="Restrict to these training run names (default: all fm/c96 runs).",
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
        help="Restrict to these training regimes (default: fm and c96).",
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
        help="Restrict to these climates (default: all a kind applies to).",
    )
    parser.add_argument(
        "--ic",
        nargs="+",
        type=int,
        default=None,
        choices=range(1, N_INITIAL_CONDITIONS + 1),
        help=(
            "Restrict eq/eq-nospinup/eq-eval-sst to these staggered initial "
            "conditions."
        ),
    )
    parser.add_argument(
        "--ens-member",
        nargs="+",
        type=int,
        default=None,
        choices=range(1, ABRUPT_ENSEMBLE_N_MEMBERS + 1),
        metavar="N",
        help=(
            "Restrict abrupt-ens-eval-sst/abrupt-ens-data-only to these members "
            "of the 36-member abrupt-4xCO2 ensemble (default: all)."
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

    runs = som_runs(args.version)
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
    for kind in args.kind:
        if kind in DATA_ONLY_KINDS:
            jobs.extend(data_only_jobs(kind, args.data_only_run, climates, ens_members))
        else:
            for run_name in runs:
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

    print(f"{len(jobs)} job(s) across {len(runs)} training run(s).")
    for job in jobs:
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


if __name__ == "__main__":
    main()
