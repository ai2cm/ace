"""Pre-submission checks shared by the ``submit_*_jobs.py`` scripts.

Every ``run-ace-*.sh`` in this directory launches with gantry, which sends the
*current HEAD commit* to Beaker; the job then clones the repository at that
commit and reads its config from ``run_configs/`` inside the clone.
``--allow-dirty`` silences gantry's own dirty-tree check but does not upload
anything, so the working tree never reaches the job. Two consequences, each of
which silently wastes a whole sweep:

  - a generated config that is untracked or modified relative to HEAD is
    absent (or stale) in the clone, and every job dies looking for it;
  - a HEAD commit that has not been pushed cannot be cloned at all.

A third, cheaper failure: ``gantry`` missing from ``PATH`` makes the run script
exit 127 on the first job, after which the submitter aborts partway through.

``check_submit_preconditions`` catches all three before the first job is
submitted. Callers declare the paths their jobs read *from the clone*: the
configs being submitted, plus any script the job's entrypoint runs by
repository path (run_eval_suite.py for the evaluator; the train and inference
scripts instead run ``-m fme.ace.*`` from the pip-installed clone, so their
configs are the only declaration needed).
"""

import pathlib
import shutil
import subprocess
from collections.abc import Sequence

HERE = pathlib.Path(__file__).parent
REMOTE = "origin"


class PreflightError(RuntimeError):
    """A submission precondition that would waste the whole sweep."""


def _git(*args: str) -> str:
    """Run a git command in this repository and return its stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _repo_root() -> pathlib.Path:
    return pathlib.Path(_git("rev-parse", "--show-toplevel"))


def _gantry_problem() -> str | None:
    if shutil.which("gantry") is not None:
        return None
    return (
        "gantry is not on PATH, so run-ace-*.sh would exit 127 on the first "
        "job. Activate the environment gantry is installed in (e.g. the fme "
        "conda environment) before submitting."
    )


def _unpushed_head_problem() -> str | None:
    """HEAD must exist on the remote: the Beaker job clones it by sha."""
    fetch = subprocess.run(
        ["git", "fetch", REMOTE, "--quiet"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        return (
            f"could not fetch {REMOTE} to check whether HEAD is pushed, so "
            "submission cannot be verified as safe: "
            f"{fetch.stderr.strip() or 'git fetch failed'}"
        )
    if _git("branch", "-r", "--contains", "HEAD"):
        return None
    sha = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return (
        f"HEAD ({sha}) is not on any {REMOTE} branch, so the Beaker job cannot "
        f"clone it. Push first:\n    git push {REMOTE} {branch}"
    )


def _uncommitted_problem(paths: Sequence[pathlib.Path]) -> str | None:
    """Paths the job reads from the clone must match HEAD exactly.

    ``git status --porcelain`` reports untracked, unstaged and staged-but-
    uncommitted paths alike, which is precisely the set that differs from the
    commit gantry hands to Beaker.
    """
    if not paths:
        return None
    absolute_paths = [str(path.resolve()) for path in paths]
    status = _git("status", "--porcelain", "--", *absolute_paths)
    if not status:
        return None
    # Porcelain lines are "XY <path>", with paths relative to the repository
    # root; only these offending paths belong in the suggested fix, not every
    # declared path.
    offending_paths = sorted(line[3:] for line in status.splitlines())
    listing = "\n".join(f"    {line}" for line in status.splitlines())
    repo_root = _repo_root()
    return (
        "these paths differ from HEAD, so the Beaker job would clone a commit "
        f"without them:\n{listing}\n"
        "Commit and push them first:\n"
        f"    git -C {repo_root} add {' '.join(offending_paths)}\n"
        f'    git -C {repo_root} commit -m "Submitting jobs"\n'
        f"    git -C {repo_root} push {REMOTE} HEAD"
    )


def check_submit_preconditions(
    paths: Sequence[pathlib.Path],
    dry_run: bool,
) -> None:
    """Verify a sweep can actually run before any job is submitted.

    ``paths`` are the files the jobs read from the cloned repository. Under
    ``dry_run`` the problems are reported and submission continues, so that the
    usual generate → dry-run → commit → submit loop is not blocked by configs
    that are about to be committed; a real submission raises instead.
    """
    problems = [
        problem
        for problem in (
            _gantry_problem(),
            _unpushed_head_problem(),
            _uncommitted_problem(paths),
        )
        if problem is not None
    ]
    if not problems:
        return
    report = "\n".join(f"  - {problem}" for problem in problems)
    if dry_run:
        print(f"Preflight warnings (submission would fail):\n{report}")
        return
    raise PreflightError(f"refusing to submit:\n{report}")
