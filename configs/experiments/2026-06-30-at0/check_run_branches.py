"""Report wandb runs whose source git branch does not match their project.

Each project is expected to hold runs launched from one branch:

  AirTemp0 -> exp/alexey7

Resolution path for each run:
  wandb run name
    -> run.notes  (a https://beaker.org/ex/<experiment_id> link)
    -> beaker experiment  -> job envVar GIT_REF (a commit SHA)
    -> which of the expected branches the commit is an ancestor of

A run is flagged when GIT_REF is not an ancestor of the project's expected
branch (i.e. it was launched from the wrong branch, or the commit is missing
locally so the branch can't be determined).

Usage:
    python check_run_branches.py
"""

import argparse
import json
import re
import subprocess

WANDB_ENTITY = "ai2cm"
PROJECT_BRANCHES = {
    "AirTemp0": "exp/alexey7",
}
# Only training runs are versioned with these suffixes; eval/export runs and
# stragglers without one are skipped.
KNOWN_BRANCHES = list(PROJECT_BRANCHES.values())

EXPERIMENT_RE = re.compile(r"beaker\.org/ex/([0-9A-Za-z]+)")


def _experiment_id_from_notes(notes: str | None) -> str | None:
    if not notes:
        return None
    match = EXPERIMENT_RE.search(notes)
    return match.group(1) if match else None


def _git_ref(experiment_id: str) -> str | None:
    """Return the GIT_REF commit SHA recorded by gantry, else None."""
    proc = subprocess.run(
        ["beaker", "experiment", "get", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    experiment = json.loads(proc.stdout)[0]
    for job in experiment.get("jobs", []):
        env_vars = job.get("execution", {}).get("spec", {}).get("envVars", [])
        for env in env_vars:
            if env.get("name") == "GIT_REF":
                return env.get("value")
    return None


def _commit_present(sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


def _is_ancestor(sha: str, branch: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, branch],
            capture_output=True,
        ).returncode
        == 0
    )


def _branches_for(sha: str) -> list[str]:
    """Which known branches contain this commit (as an ancestor)."""
    return [branch for branch in KNOWN_BRANCHES if _is_ancestor(sha, branch)]


def _fetch_run_notes(project: str) -> dict[str, str | None]:
    import wandb  # lazy import: keeps the module importable without wandb

    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{project}")
    return {run.name: run.notes for run in runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every run's resolved branch, not just mismatches.",
    )
    args = parser.parse_args()

    for project, expected_branch in PROJECT_BRANCHES.items():
        print(f"\n=== {project} (expected branch: {expected_branch}) ===")
        run_notes = _fetch_run_notes(project)
        for run_name, notes in sorted(run_notes.items()):
            experiment_id = _experiment_id_from_notes(notes)
            if experiment_id is None:
                print(f"  {run_name}: no beaker link in notes")
                continue
            sha = _git_ref(experiment_id)
            if sha is None:
                print(f"  {run_name}: no GIT_REF / experiment unavailable")
                continue
            if not _commit_present(sha):
                print(f"  {run_name}: commit {sha[:12]} not present locally (fetch)")
                continue
            matches = _is_ancestor(sha, expected_branch)
            if matches and not args.verbose:
                continue  # matches the project
            branches = _branches_for(sha) or ["<none of the known branches>"]
            print(f"  {run_name}: {sha[:12]} -> {', '.join(branches)}")


if __name__ == "__main__":
    main()
