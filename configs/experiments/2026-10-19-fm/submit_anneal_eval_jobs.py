"""Submit the post-training evaluation jobs of the anneal sweep.

Checkpoints (--ckpts, default C,B):

  C  {run}-ckptC Beaker dataset (make_soup_checkpoints.py, anneal_soups.json),
     file soup_ckpt.tar: uniform soup of the run's ema_ckpt_0016..0020.tar
  B  the run's training result dataset, training_checkpoints/best_inference_ckpt.tar
  P  the parent's training result dataset, training_checkpoints/ema_ckpt_0150.tar
     (the anneal start point). P jobs are per parent, not per anneal run:
     every mask-on arm of a seed shares its parent. P's J1 is the parent's
     existing `{parent}-lastema` eval suite (submit_inference_jobs.py --ckpt
     lastema), so only J3/J4 are submitted for P, unless that job has not
     succeeded in Beaker.

Families (--families, default J1,J3,J4), each with the inference config the
parents' -bestinf jobs ran (generate_inference_configs.py):

  J1  {base}-ckpt{X}-bestinf             eval suite (2026-06-26 windows), suite mode
  J3  {base}-ckpt{X}-sst-{grid}-p{K}k    SST +0/2/4 K on era5 / c96, inference mode
  J4  {base}-ckpt{X}-{kind}              six SOM / ERA5 4xCO2 and control kinds,
                                         evaluator mode

{base} is the anneal run name for C and B and the parent run name for P.
The "-bestinf" suffix of J1 names the eval suite (as in the parents' job
names), not the checkpoint.

--entries NAME ... reduces J1 to those suite entries: the reduced suite is
written to inference_configs/ (it must be committed before a real
submission, since gantry clones HEAD) and the J1 job name gets
"-only-{entries}" appended, so it never stands in for the full suite.

wandb groups: J1 ace2-fm-anneal-eval-2026-10-19, J3
ace2-fm-anneal-sst-perts-2026-10-19, J4 ace2-fm-anneal-paper-2026-10-19.

Usage:
    python submit_anneal_eval_jobs.py [--arms a0 ...] [--seeds 0 ...]
                                      [--only RUN ...] [--smoke]
                                      [--ckpts C,B[,P]] [--families J1,J3,J4]
                                      [--entries ENTRY ...] [--dry-run]
                                      [--skip-if-in-beaker] [--exclude-job NAME ...]
                                      [--beaker-workspace WORKSPACE]
                                      [--beaker-cluster CLUSTER [CLUSTER ...]]
                                      [--beaker-priority PRIORITY]
"""

import argparse
import os
import pathlib
from typing import NamedTuple

import yaml
from _beaker_listing import OK, fetch_experiments_by_name
from _inference_common import CHECKPOINT_PATH, INFERENCE_CONFIGS_DIR, WANDB_PROJECT
from _submit_common import check_configs_at_head, drop_jobs_in_beaker, submit_job
from create_anneal_jobs import ARMS, PARENT_CHECKPOINT, SMOKE_SEED, parent_job_name
from generate_inference_configs import (
    EVAL_SUITE_CONFIGS,
    PAPER_KINDS,
    SST_GRIDS,
    SST_PERTURBATIONS,
    paper_config_filename,
    sst_config_filename,
)
from make_soup_checkpoints import REPO_ROOT, add_selection_args, load_soups, select_runs
from submit_fm_jobs import add_default_beaker_args
from submit_inference_jobs import validate_configs

HERE = pathlib.Path(__file__).parent
RUN_SCRIPT = HERE / "run-ace-inference-job.sh"
CONFIGS_DIRNAME = INFERENCE_CONFIGS_DIR.name

CKPTS = ("C", "B", "P")
FAMILIES = ("J1", "J3", "J4")
WANDB_GROUPS = {
    "J1": "ace2-fm-anneal-eval-2026-10-19",
    "J3": "ace2-fm-anneal-sst-perts-2026-10-19",
    "J4": "ace2-fm-anneal-paper-2026-10-19",
}
# run-ace-inference-job.sh MODE and the submit_inference_jobs family whose
# validation applies.
MODES = {"J1": "suite", "J3": "inference", "J4": "evaluator"}
VALIDATE_FAMILY = {"J1": "eval", "J3": "sst", "J4": "paper-era5"}
J1_SUITE = EVAL_SUITE_CONFIGS["eval"]
J1_SUFFIX = "-bestinf"
P_J1_SUFFIX = "-lastema"  # submit_inference_jobs.py --ckpt lastema


class Job(NamedTuple):
    name: str
    family: str
    config: str  # filename in inference_configs/
    dataset_id: str
    checkpoint_path: str  # inside dataset_id, mounted at /ckpt.tar


def family_jobs(base: str, family: str, j1_config: str, j1_suffix: str):
    """(job name, config) of one checkpoint's jobs in one family."""
    if family == "J1":
        return [(f"{base}{j1_suffix}", j1_config)]
    if family == "J3":
        return [
            (f"{base}-sst-{grid}-{level}", sst_config_filename(grid, level))
            for grid in SST_GRIDS
            for level in SST_PERTURBATIONS
        ]
    if family == "J4":
        return [(f"{base}-{kind}", paper_config_filename(kind)) for kind in PAPER_KINDS]
    raise ValueError(family)


def reduced_suite(entries: list[str]) -> str:
    """Write the J1 suite with only `entries`; return its filename."""
    source = INFERENCE_CONFIGS_DIR / J1_SUITE
    suite = yaml.safe_load(source.read_text())
    names = [entry["name"] for entry in suite["inferences"]]
    unknown = sorted(set(entries) - set(names))
    if unknown:
        raise SystemExit(f"--entries not in {J1_SUITE}: {unknown}; have {names}")
    suite["inferences"] = [e for e in suite["inferences"] if e["name"] in entries]
    filename = source.name.replace(".yaml", f"-only-{'-'.join(entries)}.yaml")
    path = INFERENCE_CONFIGS_DIR / filename
    header = (
        "# Generated by submit_anneal_eval_jobs.py --entries from "
        f"{J1_SUITE}; do not edit by hand.\n"
    )
    text = header + yaml.safe_dump(suite, default_flow_style=False, sort_keys=False)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
        print(f"Wrote {CONFIGS_DIRNAME}/{filename}")
    return filename


def parse_list(value: str, choices: tuple[str, ...]) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = sorted(set(items) - set(choices))
    if bad:
        raise argparse.ArgumentTypeError(f"{bad} not in {choices}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_selection_args(parser)
    parser.add_argument(
        "--ckpts",
        type=lambda v: parse_list(v, CKPTS),
        default=["C", "B"],
        help="Comma-separated checkpoints, from C,B,P (default: C,B).",
    )
    parser.add_argument(
        "--families",
        type=lambda v: parse_list(v, FAMILIES),
        default=list(FAMILIES),
        help="Comma-separated families, from J1,J3,J4 (default: all).",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        default=None,
        metavar="ENTRY",
        help="J1 only: submit a reduced suite with only these entries.",
    )
    add_default_beaker_args(parser)
    args = parser.parse_args()
    # Validation and the soup use this checkout's fme, not whatever the venv has.
    os.environ["PYTHONPATH"] = str(REPO_ROOT)

    runs = select_runs(args.arms, args.seeds, args.only, args.smoke)
    j1_config, j1_suffix = J1_SUITE, J1_SUFFIX
    if args.entries is not None:
        j1_config = reduced_suite(args.entries)
        j1_suffix = f"{J1_SUFFIX}-only-{'-'.join(args.entries)}"

    print(f"Listing experiments in {args.beaker_workspace}...")
    experiments = fetch_experiments_by_name(args.beaker_workspace)

    def result_dataset(name: str) -> str | None:
        named = experiments.get(name)
        if named is None or named.status != OK:
            return None
        return named.result_dataset

    soups = load_soups()
    jobs: list[Job] = []
    for run in runs:
        dataset = result_dataset(run)
        if dataset is None:
            status = experiments[run].status if run in experiments else "missing"
            print(f"Skipping (training not finished, {status}): {run}")
            continue
        for ckpt in ("C", "B"):
            if ckpt not in args.ckpts:
                continue
            if ckpt == "C":
                soup = soups.get(run)
                if soup is None:
                    print(f"Skipping C (no soup in anneal_soups.json): {run}")
                    continue
                if soup["source_dataset"] != dataset:
                    print(
                        f"WARNING: {run} soup was made from {soup['source_dataset']}"
                        f", Beaker now has {dataset}"
                    )
                ckpt_dataset, ckpt_path = soup["dataset_id"], soup["file"]
            else:
                ckpt_dataset, ckpt_path = dataset, CHECKPOINT_PATH
            for family in args.families:
                for name, config in family_jobs(
                    f"{run}-ckpt{ckpt}", family, j1_config, j1_suffix
                ):
                    jobs.append(Job(name, family, config, ckpt_dataset, ckpt_path))

    if "P" in args.ckpts:
        if args.smoke:
            parents = sorted({parent_job_name(SMOKE_SEED, "on")})
        else:
            parents = sorted(
                {
                    parent_job_name(s, ARMS[a].parent_mask)
                    for a in args.arms
                    for s in args.seeds
                }
            )
        for parent in parents:
            dataset = result_dataset(parent)
            if dataset is None:
                print(f"Skipping P (parent not succeeded): {parent}")
                continue
            for family in args.families:
                if family == "J1" and args.entries is None:
                    lastema = f"{parent}{P_J1_SUFFIX}"
                    if result_dataset(lastema) is not None:
                        print(f"P J1 covered by {lastema}")
                        continue
                for name, config in family_jobs(
                    f"{parent}-ckptP", family, j1_config, j1_suffix
                ):
                    jobs.append(Job(name, family, config, dataset, PARENT_CHECKPOINT))

    jobs = drop_jobs_in_beaker(jobs, lambda job: job.name, args)
    if not jobs:
        print("Nothing to submit.")
        return

    needed = sorted({(job.family, job.config) for job in jobs})
    if not args.dry_run:
        check_configs_at_head([INFERENCE_CONFIGS_DIR / config for _, config in needed])
    validate_configs(
        sorted({(VALIDATE_FAMILY[family], config) for family, config in needed})
    )

    for job in jobs:
        submit_job(
            RUN_SCRIPT,
            [
                MODES[job.family],
                f"{CONFIGS_DIRNAME}/{job.config}",
                job.name,
                WANDB_GROUPS[job.family],
                job.dataset_id,
                job.checkpoint_path,
            ],
            wandb_project=WANDB_PROJECT,
            args=args,
            cwd=HERE,
            extra_env={"SKIP_VALIDATE": "1"},
        )

    counts: dict[str, int] = {}
    for job in jobs:
        key = f"{job.family} {job.name.split('-ckpt')[1][0]}"
        counts[key] = counts.get(key, 0) + 1
    verb = "would be submitted" if args.dry_run else "submitted"
    print(
        f"\n{len(jobs)} job(s) {verb}: "
        + ", ".join(f"{k}: {n}" for k, n in sorted(counts.items()))
    )
    longest = max(jobs, key=lambda job: len(job.name))
    print(f"Longest job name ({len(longest.name)} chars): {longest.name}")


if __name__ == "__main__":
    main()
