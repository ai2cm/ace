"""Script to delete all the beaker results datasets except for the result
dataset corresponding to the last job of an experiment. Particularly useful
to reduce storage costs associated with experiments which were preempted
many times. Use with care! This has not been thoroughly tested!

Requires beaker-py>=2.
"""

import argparse
import datetime

import beaker

SAFETY_PERIOD = datetime.timedelta(days=90)
TODAY = datetime.datetime.now(datetime.timezone.utc)


def handle_experiment(client: beaker.Beaker, wl, dry_run: bool = False):
    """Delete result datasets all jobs except the last for a given experiment."""
    print(f"Processing experiment: {client.workload.url(wl)}")
    # we assume one task for experiment, which is case for all ACE workloads
    task = wl.experiment.tasks[0]
    jobs = list(
        client.job.list(
            task=task,
            sort_field="created",
            sort_order=beaker.types.BeakerSortOrder.descending,
        )
    )
    if len(jobs) <= 1:
        # experiment was never preempted, so nothing to do
        print("This experiment has a single job, so nothing to delete.")
        return

    # we want to save the last results dataset that has at least 1 file
    # and delete all results datasets from prior jobs.
    # We can't just use the result dataset from the last job because there
    # is an edge case where a job may fail to start and so have an empty
    # results dataset, and it's possible this might happen for the last job.
    delete_all_earlier_datasets = False
    datasets_to_delete = []

    # assuming the jobs are sorted starting from most recent
    for job in jobs:
        ds = client.job.get_results(job)
        if not delete_all_earlier_datasets:
            file_generator = client.dataset.list_files(ds)
            ds_has_at_least_one_file = False
            for _ in file_generator:
                ds_has_at_least_one_file = True
                break
            if ds_has_at_least_one_file:
                delete_all_earlier_datasets = True
        else:
            datasets_to_delete.append(ds)

    if dry_run:
        print(f"  [DRY RUN] Would be deleting {len(datasets_to_delete)} datasets...")
        for ds in datasets_to_delete:
            print(f"  [DRY RUN] Would delete dataset {client.dataset.url(ds)}")
    else:
        print(f"  Deleting {len(datasets_to_delete)} datasets...")
        client.dataset.delete(*datasets_to_delete)


def delete_non_last_job_datasets(
    workspace_name: str, dry_run: bool = False, username: str | None = None
):
    with beaker.Beaker.from_env() as client:
        if username is None:
            username = client.user_name
        print(f"Scanning workspace: {workspace_name} for experiments by {username}")
        workloads = client.workload.list(
            workspace=workspace_name,
            author=username,
            created_before=TODAY - SAFETY_PERIOD,
            workload_type=beaker.types.BeakerWorkloadType.experiment,
        )
        for wl in workloads:
            handle_experiment(client, wl, dry_run=dry_run)

        print("\nDone scanning workspace.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete results dataset for all jobs except the last for all "
        "experiments in a workspace. Will not delete results dataset for any "
        "experiments in the last 90 days."
    )
    parser.add_argument(
        "workspace", type=str, help="Workspace name or id (e.g. 'ai2/my-workspace')"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Username for which to delete datasets. If none "
        "provided, will use current active user.",
        required=False,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without actually deleting datasets.",
    )
    args = parser.parse_args()
    delete_non_last_job_datasets(
        args.workspace, dry_run=args.dry_run, username=args.username
    )
