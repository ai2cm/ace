"""Make checkpoint C (uniform EMA soup) of each anneal run and upload it to Beaker.

Checkpoint C of an anneal run is the uniform average (fme.core.checkpoint_soup)
of the EMA checkpoints `training_checkpoints/ema_ckpt_0016.tar` ...
`ema_ckpt_0020.tar` in the run's Beaker result dataset. For each selected run
whose training experiment succeeded, this script

1. fetches the epoch checkpoints (`beaker dataset fetch --prefix`) into
   SCRATCH/{run}/;
2. runs `python -m fme.core.checkpoint_soup` (this checkout's fme, via
   PYTHONPATH) into SCRATCH/{run}/upload/SOUP_FILENAME;
3. checks the soup loads with torch.load, carries a `soup` entry with the
   expected sources and uniform weights, and has no `optimization` state;
4. uploads SCRATCH/{run}/upload/ as Beaker dataset `{run}-ckptC` (workspace
   ai2/ace) and records its id in anneal_soups.json.

Runs whose `{run}-ckptC` dataset already exists are skipped (and recorded in
anneal_soups.json if missing there) unless --force, which deletes nothing but
uploads a new dataset under a timestamped name and records that instead.
Runs still training are skipped with a message, so the script can be rerun
as runs finish.

Usage:
    python make_soup_checkpoints.py [--arms a0 a1 ...] [--seeds 0 1 ...]
                                    [--only RUN ...] [--smoke]
                                    [--epochs 16 17 18 19 20] [--force]
                                    [--dry-run] [--keep-scratch]
                                    [--scratch-dir DIR]
                                    [--beaker-workspace WORKSPACE]
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

from _beaker_listing import DEFAULT_WORKSPACE, fetch_experiments_by_name
from _inference_common import finished_runs
from create_anneal_jobs import ARMS, SEEDS, SMOKE_ARMS, SMOKE_SEED, anneal_job_name

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parents[2]
SOUPS_PATH = HERE / "anneal_soups.json"
SOUP_EPOCHS = (16, 17, 18, 19, 20)
SOUP_FILENAME = "soup_ckpt.tar"
SOUP_SUFFIX = "-ckptC"
DEFAULT_SCRATCH = pathlib.Path("/tmp/ace-anneal-soups")
BUDGET = "ai2/atec-climate"


def ema_checkpoint(epoch: int) -> str:
    return f"training_checkpoints/ema_ckpt_{epoch:04d}.tar"


def soup_dataset_name(run: str) -> str:
    return f"{run}{SOUP_SUFFIX}"


def select_runs(
    arms: list[str], seeds: list[int], only: list[str] | None, smoke: bool
) -> list[str]:
    """Anneal run names for --arms/--seeds (or the smoke runs), narrowed by --only."""
    if smoke:
        runs = [anneal_job_name(a, SMOKE_SEED, True) for a in SMOKE_ARMS if a in arms]
    else:
        runs = [anneal_job_name(a, s) for a in arms for s in seeds]
    if only is not None:
        unknown = sorted(set(only) - set(runs))
        if unknown:
            raise SystemExit(f"--only names not selected: {unknown}")
        runs = [r for r in runs if r in only]
    return runs


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="RUN",
        help="Only these anneal run names (after --arms/--seeds/--smoke).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Use the smoke runs (arms {SMOKE_ARMS}, seed {SMOKE_SEED}).",
    )


def load_soups() -> dict[str, dict]:
    if SOUPS_PATH.exists():
        return json.loads(SOUPS_PATH.read_text())
    return {}


def save_soups(soups: dict[str, dict]) -> None:
    SOUPS_PATH.write_text(json.dumps(dict(sorted(soups.items())), indent=2) + "\n")


def beaker_account() -> str:
    out = subprocess.run(
        ["beaker", "account", "whoami", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)[0]["name"]


def existing_dataset(full_name: str) -> dict | None:
    proc = subprocess.run(
        ["beaker", "dataset", "get", full_name, "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or "not found" in proc.stderr:
        return None
    try:
        return json.loads(proc.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        return None


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def fetch_checkpoints(dataset: str, epochs: list[int], dest: pathlib.Path) -> list[str]:
    paths = []
    for epoch in epochs:
        rel = ema_checkpoint(epoch)
        path = dest / rel
        if not path.exists():
            subprocess.run(
                [
                    "beaker",
                    "dataset",
                    "fetch",
                    dataset,
                    "--prefix",
                    rel,
                    "--output",
                    str(dest),
                    "--quiet",
                ],
                check=True,
            )
        if not path.exists():
            raise FileNotFoundError(f"{rel} not in dataset {dataset}")
        paths.append(str(path))
    return paths


def run_soup(inputs: list[str], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    subprocess.run(
        [sys.executable, "-m", "fme.core.checkpoint_soup", str(output), *inputs],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def verify_soup(path: pathlib.Path, inputs: list[str]) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "soup" not in checkpoint:
        raise ValueError(f"{path}: no 'soup' entry")
    if "optimization" in checkpoint:
        raise ValueError(f"{path}: still holds 'optimization' state")
    if "stepper" not in checkpoint:
        raise ValueError(f"{path}: no 'stepper' entry")
    soup = checkpoint["soup"]
    if soup["sources"] != inputs:
        raise ValueError(f"{path}: soup sources {soup['sources']} != {inputs}")
    n = len(inputs)
    if any(abs(w - 1.0 / n) > 1e-9 for w in soup["weights"]):
        raise ValueError(f"{path}: weights {soup['weights']} are not uniform")
    print(f"  verified {path.name}: keys {sorted(checkpoint)}")


def upload(
    directory: pathlib.Path, full_name: str, description: str, workspace: str
) -> str:
    """Create the dataset and return its id (looked up by name: the create
    command's output is not reliably JSON).
    """
    name = full_name.split("/", 1)[1]
    subprocess.run(
        [
            "beaker",
            "dataset",
            "create",
            str(directory),
            "--name",
            name,
            "--workspace",
            workspace,
            "--budget",
            BUDGET,
            "--desc",
            description,
        ],
        check=True,
    )
    created = existing_dataset(full_name)
    if created is None:
        raise RuntimeError(f"dataset {full_name} not found after upload")
    return created["id"]


def record_from_dataset(run: str, dataset: dict, source: str) -> dict:
    """anneal_soups.json entry of an already uploaded soup, from its Beaker
    description.
    """
    description = dataset.get("description", "")
    epochs = [int(e) for e in re.findall(r"ema_ckpt_(\d{4})\.tar", description)]
    commit = re.search(r"commit ([0-9a-f]{40})", description)
    return {
        "dataset_id": dataset["id"],
        "dataset_name": dataset["name"],
        "file": SOUP_FILENAME,
        "source_dataset": source,
        "epochs": epochs or None,
        "commit": commit.group(1) if commit else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_selection_args(parser)
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=list(SOUP_EPOCHS),
        help=f"EMA checkpoint epochs to average (default: {list(SOUP_EPOCHS)}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Make and upload a soup even if {run}-ckptC already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep the fetched checkpoints and soup in the scratch dir.",
    )
    parser.add_argument("--scratch-dir", type=pathlib.Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--beaker-workspace", default=DEFAULT_WORKSPACE)
    args = parser.parse_args()

    runs = select_runs(args.arms, args.seeds, args.only, args.smoke)
    print(f"Listing experiments in {args.beaker_workspace}...")
    experiments = fetch_experiments_by_name(args.beaker_workspace)
    ready, skipped = finished_runs(runs, experiments)
    for run, reason in sorted(skipped.items()):
        print(f"Skipping (training not finished): {run} ({reason})")

    account = beaker_account()
    commit = git_commit()
    soups = load_soups()
    epochs = sorted(args.epochs)
    made = 0
    for run in runs:
        if run not in ready:
            continue
        source = ready[run]
        name = soup_dataset_name(run)
        existing = existing_dataset(f"{account}/{name}")
        if existing is not None and not args.force:
            print(f"Skipping (dataset exists): {name} ({existing['id']})")
            if soups.get(run, {}).get("dataset_id") != existing["id"]:
                soups[run] = record_from_dataset(run, existing, source)
            continue
        if existing is not None:
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
            name = f"{name}-{stamp}"
        sources = [ema_checkpoint(e) for e in epochs]
        description = (
            f"Checkpoint C of {run}: uniform soup (fme.core.checkpoint_soup) of "
            f"{', '.join(sources)} from Beaker dataset {source}; "
            f"ai2cm/ace commit {commit}"
        )
        print(f"\n{run}\n  source {source}, epochs {epochs} -> {name}")
        if args.dry_run:
            continue
        work = args.scratch_dir / run
        inputs = fetch_checkpoints(source, epochs, work)
        output = work / "upload" / SOUP_FILENAME
        run_soup(inputs, output)
        verify_soup(output, inputs)
        dataset_id = upload(
            output.parent, f"{account}/{name}", description, args.beaker_workspace
        )
        print(f"  uploaded {name}: {dataset_id}")
        soups[run] = {
            "dataset_id": dataset_id,
            "dataset_name": name,
            "file": SOUP_FILENAME,
            "source_dataset": source,
            "epochs": epochs,
            "commit": commit,
        }
        save_soups(soups)
        made += 1
        if not args.keep_scratch:
            shutil.rmtree(work)
    if not args.dry_run:
        save_soups(soups)
    print(f"\n{made} soup(s) made; {len(skipped)} run(s) not finished.")


if __name__ == "__main__":
    main()
