"""Fetch beaker config.yaml for every run of every wandb project in projects.yaml.

Each entry under ``wandb_projects`` is either a plain project name or a
mapping with ``name`` (the wandb project) and an optional ``description``.
For each project, create a folder named after it and write one
``<run_name>.yaml`` per run, holding the ``config.yaml`` from that run's
Beaker experiment result dataset.

Lookup chain per run:
    wandb run
      -> run.config["environment"]["BEAKER_EXPERIMENT_ID"]   (beaker experiment)
      -> beaker.workload.get_results(...)                    (result dataset)
      -> beaker.dataset.stream_file(..., "config.yaml")      (config.yaml)

Usage:
    python fetch_wandb_configs.py [--projects PATH] [--entity ENTITY] \
        [--out-dir DIR] [--force]
"""

import argparse
import pathlib

import wandb
import yaml
from beaker import Beaker

HERE = pathlib.Path(__file__).parent
DEFAULT_PROJECTS = HERE / "projects.yaml"
DEFAULT_OUT = HERE
DEFAULT_ENTITY = "ai2cm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects",
        type=pathlib.Path,
        default=DEFAULT_PROJECTS,
        help=f"YAML file listing wandb_projects (default: {DEFAULT_PROJECTS}).",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"wandb entity to fetch projects from (default: {DEFAULT_ENTITY}).",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help=(
            "Directory under which per-project folders are created "
            f"(default: {DEFAULT_OUT})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch configs that already exist locally.",
    )
    args = parser.parse_args()

    projects = yaml.safe_load(args.projects.read_text())["wandb_projects"]

    api = wandb.Api()
    with Beaker.from_env() as beaker:
        for entry in projects:
            # Entry is either a plain project name or a mapping with
            # `name` (required) and `description` (optional).
            if isinstance(entry, str):
                project, description = entry, ""
            else:
                project, description = entry["name"], entry.get("description", "")
            project_dir = args.out_dir / project
            project_dir.mkdir(parents=True, exist_ok=True)
            print(f"{project}:" + (f" {description}" if description else ""))
            runs = api.runs(f"{args.entity}/{project}")
            for run in runs:
                out_path = project_dir / f"{run.name}.yaml"
                if out_path.exists() and not args.force:
                    print(f"  skip {run.name}: already fetched")
                    continue
                experiment_id = run.config.get("environment", {}).get(
                    "BEAKER_EXPERIMENT_ID"
                )
                if not experiment_id:
                    print(f"  skip {run.name}: no BEAKER_EXPERIMENT_ID in config")
                    continue
                try:
                    workload = beaker.workload.get(experiment_id)
                except Exception as exc:  # experiment deleted / no access
                    print(f"  skip {run.name}: cannot get {experiment_id}: {exc}")
                    continue
                result_dataset = beaker.workload.get_results(workload)
                if result_dataset is None:
                    print(f"  skip {run.name}: no result dataset for {experiment_id}")
                    continue
                try:
                    config_bytes = b"".join(
                        beaker.dataset.stream_file(result_dataset, "config.yaml")
                    )
                except Exception as exc:  # no config.yaml in dataset
                    print(
                        f"  skip {run.name}: no config.yaml in "
                        f"{result_dataset.id}: {exc}"
                    )
                    continue
                out_path.write_bytes(config_bytes)
                print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
