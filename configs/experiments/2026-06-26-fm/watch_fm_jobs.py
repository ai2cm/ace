"""One tick of the FM job watcher: fill in what is missing, retry what failed.

Beaker is the source of truth. Each tick lists the `ai2/ace` workspace once
and, against the expected set of jobs for the norm-ablation architectures,
does three things:

1. **Dependency stage.** For every training run that has succeeded, the jobs
   that read its checkpoints are generated (eval, fixed-variable suites,
   fine-tune configs), committed, pushed and submitted: evaluator runs on
   three checkpoints, the SST-perturbation sweep, the specific_total_water_0
   own and swapped suites (fm cells), the ERA5 fine-tune, and epoch 0 of the
   fine-tune SST sweep. For every fine-tune that has succeeded, epochs 1-10 of
   its SST sweep. Training runs not yet started (the mask10 A1 baselines) are
   submitted too.
2. **Failure scan.** A job whose experiment ended without an exit-0 job and
   without a user cancel is resubmitted, up to RETRY_LIMIT times. A swin
   training or fine-tune job whose log shows a non-finite loss is resubmitted
   at seed + 1 through seed_overrides.json, so it does not retrace the same
   trajectory. A job at the limit is marked exhausted, never touched again,
   and reported on a `NOTIFY:` line for the caller to forward.
3. **Report.** A summary of what was submitted, skipped and exhausted.

Submission goes through the six submit_*.py scripts with --skip-if-in-beaker
and --exclude-job, so a name already succeeded or running is never queued
twice and an exhausted one is held back even when its suite's siblings go.

State (attempt counts, exhausted names, last seen statuses) lives in
job_watch_state.json next to this file and is not committed.

Usage:
    python watch_fm_jobs.py [--dry-run] [--no-git] [--stage STAGE ...]

--dry-run prints the plan and runs the submit scripts with --dry-run.
--no-git skips pull, commit and push (for a dry run on a dirty tree).
"""

import argparse
import dataclasses
import datetime
import json
import pathlib
import subprocess
import sys
from collections.abc import Iterable, Sequence

from _beaker_listing import (
    CANCELED,
    FAILED,
    NO_JOB,
    OK,
    RUNNING,
    NamedExperiment,
    fetch_experiments_by_name,
)
from generate_eval_configs import EVAL_CHECKPOINT_NAME_SUFFIXES, WANDB_PREFIX
from generate_norm_ablation_configs import (
    ARCH_SOURCES,
    CONFIG_PREFIX,
    SEED_OVERRIDES_FILE,
    config_name,
    degenerate_reason,
)
from generate_norm_ablation_finetune_configs import (
    DEFAULT_EPOCHS,
    FINETUNE_SUFFIX,
    MASK10,
    MASK10_ARMS,
    UNMASKED,
)
from generate_norm_ablation_finetune_configs import REGIMES as FINETUNE_REGIMES
from generate_sst_configs import SST_PERTURBATIONS
from update_beaker_map import DEFAULT_MAP, refresh_map

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parents[2]
RUN_CONFIGS_DIR = HERE / "run_configs"
STATE_FILE = HERE / "job_watch_state.json"

REMOTE = "origin"
REMOTE_BRANCH = "exp/alexeyfm"

#: nc-swin-v2.1 is out of the watcher's scope: its cells are held back while
#: the instability the H1/H2 diagnostic configs probe is understood, and the
#: watcher must neither submit their dependent stages nor retry their stopped
#: jobs. Drop the name here to bring it back.
EXCLUDED_ARCHS = ("nc-swin-v2.1",)
ARCHS = tuple(arch for arch in ARCH_SOURCES if arch not in EXCLUDED_ARCHS)
REGIMES = ("era5", "c96", "fm")
ARMS = ("a1", "a2", "a3")
GRIDS_BY_REGIME = {"era5": ("era5",), "c96": ("c96",), "fm": ("era5", "c96")}
FINETUNE_GRIDS = ("era5", "c96")
LEVELS = tuple(SST_PERTURBATIONS)
EPOCHS = tuple(range(0, DEFAULT_EPOCHS + 1))
FIXED_VARIABLE = "specific_total_water_0"
FIXED_VARIANT_PARTS = ("", "swapped-")

RETRY_LIMIT = 5
NON_FINITE_MARKERS = ("non-finite", "nan")
LOG_TAIL_LINES = 400

BEAKER_ARGS = [
    "--beaker-workspace",
    "ai2/ace",
    "--beaker-cluster",
    "ai2/jupiter",
    "ai2/titan",
    "--beaker-priority",
    "normal",
]

# Stage -> the submit script that queues its jobs.
STAGE_SCRIPTS = {
    "train": "submit_norm_ablation_jobs.py",
    "eval": "submit_eval_jobs.py",
    "sst": "submit_sst_jobs.py",
    "q0": "submit_fixed_var_jobs.py",
    "finetune": "submit_norm_ablation_finetune_jobs.py",
    "ft-sst": "submit_sst_epoch_jobs.py",
}


@dataclasses.dataclass(frozen=True)
class Cell:
    arch: str
    regime: str
    arm: str
    conditional: bool
    masking: str

    @property
    def stem(self) -> str:
        return config_name(
            self.arch, self.regime, self.arm, self.conditional, self.masking
        ).removesuffix(".yaml")

    @property
    def suffix(self) -> str:
        """Run name without WANDB_PREFIX; also the --base-config spelling."""
        return self.stem.removeprefix(CONFIG_PREFIX)

    @property
    def run(self) -> str:
        return f"{WANDB_PREFIX}{self.suffix}"

    @property
    def finetune_run(self) -> str:
        return f"{self.run}{FINETUNE_SUFFIX}"

    @property
    def is_finetuned(self) -> bool:
        return self.regime in FINETUNE_REGIMES

    @property
    def is_fm(self) -> bool:
        return self.regime == "fm"


@dataclasses.dataclass(frozen=True)
class Job:
    name: str
    stage: str
    cell: Cell
    #: Job names that must have succeeded before this one can be submitted.
    needs: tuple[str, ...] = ()
    #: Extra arguments to the stage's submit script selecting this job's cell.
    select: tuple[str, ...] = ()


def cells() -> list[Cell]:
    """The training cells in scope: every non-degenerate unmasked cell of the
    architectures in ARCHS, plus the mask10 twins of the A1 cells.
    """
    out = []
    for arch in ARCHS:
        for masking in (UNMASKED, MASK10):
            for regime in REGIMES:
                for arm in ARMS:
                    for conditional in (False, True):
                        if degenerate_reason(regime, arm, conditional) is not None:
                            continue
                        if masking == MASK10 and arm not in MASK10_ARMS:
                            continue
                        out.append(Cell(arch, regime, arm, conditional, masking))
    return out


def expected_jobs() -> list[Job]:
    jobs: list[Job] = []
    for cell in cells():
        cond_flag = "--conditional" if cell.conditional else "--no-conditional"
        masking_flag = cell.masking or "none"
        jobs.append(
            Job(
                cell.run,
                "train",
                cell,
                select=(
                    "--arch",
                    cell.arch,
                    "--regime",
                    cell.regime,
                    "--arm",
                    cell.arm,
                    "--masking",
                    masking_flag,
                    cond_flag,
                ),
            )
        )
        for suffix in EVAL_CHECKPOINT_NAME_SUFFIXES:
            jobs.append(
                Job(
                    f"{cell.run}{suffix}",
                    "eval",
                    cell,
                    needs=(cell.run,),
                    select=("--arch", cell.arch, "--run", cell.run),
                )
            )
        for grid in GRIDS_BY_REGIME[cell.regime]:
            for level in LEVELS:
                jobs.append(
                    Job(
                        f"{cell.run}-sst-{grid}-{level}",
                        "sst",
                        cell,
                        needs=(cell.run,),
                        select=("--run", cell.run),
                    )
                )
        if cell.is_fm:
            for part in FIXED_VARIANT_PARTS:
                jobs.append(
                    Job(
                        f"{WANDB_PREFIX}fixed-{part}{FIXED_VARIABLE}-{cell.suffix}"
                        "-bestinf",
                        "q0",
                        cell,
                        needs=(cell.run,),
                        select=("--base-config", cell.suffix),
                    )
                )
        if cell.is_finetuned:
            jobs.append(
                Job(
                    cell.finetune_run,
                    "finetune",
                    cell,
                    needs=(cell.run,),
                    select=(
                        "--regime",
                        cell.regime,
                        "--arch",
                        cell.arch,
                        "--run",
                        cell.finetune_run,
                    ),
                )
            )
            for grid in FINETUNE_GRIDS:
                for level in LEVELS:
                    for epoch in EPOCHS:
                        needs = (cell.run,) if epoch == 0 else (cell.finetune_run,)
                        jobs.append(
                            Job(
                                f"{cell.finetune_run}-sst-{grid}-{level}-e{epoch:02d}",
                                "ft-sst",
                                cell,
                                needs=needs,
                                select=(
                                    "--run",
                                    cell.finetune_run,
                                    "--epoch",
                                    str(epoch),
                                ),
                            )
                        )
    return jobs


# --- state -----------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"attempts": {}, "exhausted": [], "seed_bumps": {}, "seen_failed": {}}


def save_state(state: dict) -> None:
    state["last_tick"] = datetime.datetime.now(datetime.UTC).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


# --- shell helpers ---------------------------------------------------------


def run(cmd: Sequence[str], *, cwd: pathlib.Path = HERE, check: bool = True) -> str:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr)
        if check:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, proc.stdout, proc.stderr
            )
    return proc.stdout


def python(script: str, *args: str, check: bool = True) -> str:
    return run([sys.executable, str(HERE / script), *args], check=check)


class Git:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def pull(self) -> None:
        if self.enabled:
            run(["git", "pull", "--ff-only", REMOTE, REMOTE_BRANCH], cwd=REPO_ROOT)

    def commit_and_push(self, paths: Iterable[pathlib.Path], message: str) -> bool:
        """Commit `paths` if any changed; push. True if a commit was made."""
        targets = [str(p) for p in paths]
        if not self.enabled:
            print(f"(no-git) would commit: {message}")
            return False
        run(["git", "add", "-A", "--", *targets], cwd=REPO_ROOT)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *targets], cwd=REPO_ROOT
        )
        if staged.returncode == 0:
            return False
        run(["git", "commit", "-q", "-m", message], cwd=REPO_ROOT)
        run(["git", "push", REMOTE, f"HEAD:{REMOTE_BRANCH}"], cwd=REPO_ROOT)
        return True


# --- failure handling ------------------------------------------------------


def is_swin_training(job: Job) -> bool:
    return job.stage in ("train", "finetune") and job.cell.arch.startswith("nc-swin")


def log_shows_non_finite(experiment_id: str) -> bool:
    proc = subprocess.run(
        ["beaker", "experiment", "logs", experiment_id, "--tail", str(LOG_TAIL_LINES)],
        capture_output=True,
        text=True,
    )
    text = (proc.stdout + proc.stderr).lower()
    return any(marker in text for marker in NON_FINITE_MARKERS)


def current_seed(config_stem: str) -> int:
    path = RUN_CONFIGS_DIR / f"{config_stem}.yaml"
    for line in path.read_text().splitlines():
        if line.startswith("seed:"):
            return int(line.split(":", 1)[1])
    return 0


def bump_seed(job: Job, reason: str) -> int:
    """Record seed + 1 for the job's config in seed_overrides.json."""
    stem = (
        job.cell.stem if job.stage == "train" else f"{job.cell.stem}{FINETUNE_SUFFIX}"
    )
    key = stem.removeprefix(CONFIG_PREFIX)
    overrides = (
        json.loads(SEED_OVERRIDES_FILE.read_text())
        if SEED_OVERRIDES_FILE.exists()
        else {}
    )
    seed = current_seed(stem) + 1
    overrides[key] = {"seed": seed, "reason": reason}
    SEED_OVERRIDES_FILE.write_text(
        json.dumps(overrides, indent=2, sort_keys=True) + "\n"
    )
    return seed


def regenerate_cell(job: Job) -> None:
    cell = job.cell
    if job.stage == "train":
        python(
            "generate_norm_ablation_configs.py",
            "--arch",
            cell.arch,
            "--regime",
            cell.regime,
            "--arm",
            cell.arm,
            "--masking",
            cell.masking or "none",
            "--conditional" if cell.conditional else "--no-conditional",
        )
    else:
        python(
            "generate_norm_ablation_finetune_configs.py",
            "--arch",
            cell.arch,
            "--regime",
            cell.regime,
        )


# --- the tick --------------------------------------------------------------


def status_of(name: str, listing: dict[str, NamedExperiment]) -> str:
    named = listing.get(name)
    return "missing" if named is None else named.status


def plan(
    jobs: list[Job],
    listing: dict[str, NamedExperiment],
    state: dict,
    stages: set[str],
    git: Git,
    dry_run: bool,
) -> dict[str, list[Job]]:
    """Decide which jobs to (re)submit this tick; update attempts and seeds."""
    to_submit: dict[str, list[Job]] = {stage: [] for stage in STAGE_SCRIPTS}
    exhausted = set(state["exhausted"])
    seed_changed = False
    for job in jobs:
        if job.stage not in stages or job.name in exhausted:
            continue
        status = status_of(job.name, listing)
        if status in (OK, RUNNING, CANCELED):
            continue
        if not all(status_of(need, listing) == OK for need in job.needs):
            continue
        if status in (FAILED, NO_JOB):
            named = listing[job.name]
            # Count one attempt per distinct failed experiment, so a failure
            # already counted on an earlier tick is not counted twice.
            if state["seen_failed"].get(job.name) != named.id:
                state["seen_failed"][job.name] = named.id
                state["attempts"][job.name] = state["attempts"].get(job.name, 0) + 1
            attempts = state["attempts"][job.name]
            if attempts > RETRY_LIMIT:
                exhausted.add(job.name)
                print(f"NOTIFY: {job.name} failed {attempts} times; giving up.")
                continue
            print(f"retry {attempts}/{RETRY_LIMIT}: {job.name} ({status})")
            if is_swin_training(job) and log_shows_non_finite(named.id):
                if not dry_run:
                    seed = bump_seed(
                        job,
                        f"non-finite loss in experiment {named.id}, attempt {attempts}",
                    )
                    regenerate_cell(job)
                    seed_changed = True
                    state["seed_bumps"][job.name] = seed
                    print(f"  non-finite loss: seed -> {seed}")
                else:
                    print("  non-finite loss: would bump seed")
        to_submit[job.stage].append(job)
    state["exhausted"] = sorted(exhausted)
    if seed_changed:
        git.commit_and_push(
            [SEED_OVERRIDES_FILE, RUN_CONFIGS_DIR],
            "Bump the seed of FM jobs that died on a non-finite loss",
        )
    return to_submit


def generate_dependents(
    to_submit: dict[str, list[Job]], git: Git, dry_run: bool
) -> None:
    """Write the configs the eval, q0 and fine-tune stages read; commit."""
    archs = sorted({job.cell.arch for job in to_submit["eval"]})
    for arch in archs:
        python("generate_eval_configs.py", "--arch", arch)
    q0_cells = sorted({job.cell.suffix for job in to_submit["q0"]})
    if q0_cells:
        python(
            "generate_fixed_var_configs.py",
            "--variable",
            FIXED_VARIABLE,
            "--variant",
            "both",
            "--base-config",
            *q0_cells,
        )
    finetune_archs = sorted({job.cell.arch for job in to_submit["finetune"]})
    for arch in finetune_archs:
        python("generate_norm_ablation_finetune_configs.py", "--arch", arch)
    if archs or q0_cells or finetune_archs:
        git.commit_and_push(
            [RUN_CONFIGS_DIR], "Add the FM eval, fixed-variable and fine-tune configs"
        )


def submit(
    to_submit: dict[str, list[Job]], exhausted: Sequence[str], dry_run: bool
) -> dict[str, int]:
    """Run each stage's submit script once per distinct selection."""
    counts: dict[str, int] = {}
    exclude = ["--exclude-job", *exhausted] if exhausted else []
    common = [*BEAKER_ARGS, "--skip-if-in-beaker", *exclude]
    if dry_run:
        common.append("--dry-run")
    for stage, jobs in to_submit.items():
        if not jobs:
            continue
        script = STAGE_SCRIPTS[stage]
        selections = _merge_selections(stage, jobs)
        for select in selections:
            out = python(script, *select, *common, check=False)
            counts[stage] = counts.get(stage, 0) + out.count("Submitting:")
    return counts


def _merge_selections(stage: str, jobs: list[Job]) -> list[tuple[str, ...]]:
    """Collapse per-job selections into as few script invocations as the
    script's flags allow: --run / --base-config take many values; the train
    and fine-tune scripts are per regime (and arch).
    """
    if stage == "train":
        return sorted({job.select for job in jobs})
    if stage == "eval":
        by_arch: dict[str, set[str]] = {}
        for job in jobs:
            by_arch.setdefault(job.cell.arch, set()).add(job.cell.run)
        return [
            ("--arch", arch, "--run", *sorted(runs)) for arch, runs in by_arch.items()
        ]
    if stage == "sst":
        return [("--run", *sorted({job.cell.run for job in jobs}))]
    if stage == "q0":
        return [("--base-config", *sorted({job.cell.suffix for job in jobs}))]
    if stage == "finetune":
        by_key: dict[tuple[str, str], set[str]] = {}
        for job in jobs:
            by_key.setdefault((job.cell.regime, job.cell.arch), set()).add(
                job.cell.finetune_run
            )
        return [
            ("--regime", regime, "--arch", arch, "--run", *sorted(runs))
            for (regime, arch), runs in sorted(by_key.items())
        ]
    if stage == "ft-sst":
        by_epochs: dict[tuple[str, ...], set[str]] = {}
        # Group runs by the set of epochs they need, so a run waiting on its
        # fine-tune (epoch 0 only) and one whose fine-tune is done (1-10) each
        # get one invocation.
        epochs_by_run: dict[str, set[str]] = {}
        for job in jobs:
            epoch = job.select[job.select.index("--epoch") + 1]
            epochs_by_run.setdefault(job.cell.finetune_run, set()).add(epoch)
        for run_name, epochs in epochs_by_run.items():
            by_epochs.setdefault(tuple(sorted(epochs, key=int)), set()).add(run_name)
        return [
            ("--run", *sorted(runs), "--epoch", *epochs)
            for epochs, runs in sorted(by_epochs.items())
        ]
    raise ValueError(stage)


def report(
    jobs: list[Job],
    listing: dict[str, NamedExperiment],
    counts: dict[str, int],
    state: dict,
) -> None:
    print("\n=== FM job watcher summary ===")
    by_stage: dict[str, dict[str, int]] = {}
    for job in jobs:
        status = status_of(job.name, listing)
        by_stage.setdefault(job.stage, {})
        by_stage[job.stage][status] = by_stage[job.stage].get(status, 0) + 1
    for stage, statuses in by_stage.items():
        line = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
        print(f"{stage:9s} {line}")
    print("submitted this tick:", counts or "nothing")
    if state["exhausted"]:
        print("exhausted:", ", ".join(state["exhausted"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=sorted(STAGE_SCRIPTS),
        default=None,
        help="Only these stages (default: all).",
    )
    args = parser.parse_args()
    stages = set(args.stage or STAGE_SCRIPTS)
    git = Git(enabled=not args.no_git and not args.dry_run)

    git.pull()
    listing = fetch_experiments_by_name()
    if refresh_map(DEFAULT_MAP, listing, dry_run=args.dry_run):
        git.commit_and_push([DEFAULT_MAP], "Refresh the FM run map from Beaker")

    state = load_state()
    jobs = expected_jobs()
    to_submit = plan(jobs, listing, state, stages, git, args.dry_run)
    if not args.dry_run:
        generate_dependents(to_submit, git, args.dry_run)
    counts = submit(to_submit, state["exhausted"], args.dry_run)
    if not args.dry_run:
        save_state(state)
    report(jobs, listing, counts, state)


if __name__ == "__main__":
    main()
