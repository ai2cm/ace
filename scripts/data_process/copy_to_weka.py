"""Copy zarr stores from GCS to WEKA by submitting a gantry job.

Replaces the earlier ``copy_zarr_to_weka.sh`` and ``gcs_to_weka.sh`` scripts. The
destination store is always ``<destination>/<source store name>``, and the job
refuses to touch a destination that already exists unless ``--overwrite`` is
given.
"""

import dataclasses
import logging
import os
import shlex
import subprocess
import sys

import click
import dacite
import fsspec
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_stats import Config

WEKA_MOUNT = "/climate-default"
WEKA_SOURCE = "climate-default"
CREDENTIALS = "/tmp/google_application_credentials.json"


@dataclasses.dataclass
class CopyPair:
    """
    A single store to copy from GCS to WEKA.

    Attributes:
        source: The gs:// path of the store to copy.
        destination: The absolute WEKA path the store is copied to.
    """

    source: str
    destination: str


def _sources_from_config(
    config: Config, coarsened: bool, runs: tuple[str, ...]
) -> list[str]:
    """Resolve the zarr stores to copy for the runs selected from a config."""
    if coarsened and config.time_coarsen is None:
        raise click.ClickException(
            "--coarsened was given but the config has no time_coarsen section."
        )
    selected = list(runs) if runs else config.run_names()
    unknown = [run for run in selected if run not in config.runs]
    if unknown:
        raise click.ClickException(
            f"Runs {unknown} are not in the config, which defines "
            f"{config.run_names()}."
        )
    if coarsened:
        return [config.coarsened_store(run) for run in selected]
    return [config.raw_store(run) for run in selected]


def _copy_pairs(sources: list[str], destination: str) -> list[CopyPair]:
    """Pair each source store with its destination, keeping the store's name."""
    if not destination.startswith(WEKA_MOUNT + "/"):
        raise click.ClickException(
            f"Destination {destination} must be a path under {WEKA_MOUNT}/, which is "
            "the only WEKA directory this job mounts."
        )
    pairs = []
    for source in sources:
        if not source.startswith("gs://"):
            raise click.ClickException(f"Source {source} must start with gs://.")
        name = source.rstrip("/").rsplit("/", 1)[-1]
        pairs.append(
            CopyPair(
                source=source.rstrip("/"),
                destination=destination.rstrip("/") + "/" + name,
            )
        )
    return pairs


def _missing_sources(pairs: list[CopyPair]) -> list[str]:
    """Sources that could be checked and do not exist.

    Sources whose existence cannot be determined, e.g. because no GCS
    credentials are available locally, are reported as present so that a
    credentials problem here does not block submitting the job.
    """
    missing = []
    for pair in pairs:
        # The GCS client logs its own tracebacks when credentials are stale, which
        # would drown out the warning below for what is only a best effort check.
        logging.disable(logging.ERROR)
        try:
            filesystem, _, (path,) = fsspec.get_fs_token_paths(pair.source)
            exists = filesystem.exists(path)
        except Exception as err:
            error: Exception | None = err
        else:
            error = None
        finally:
            logging.disable(logging.NOTSET)
        if error is not None:
            logging.warning(f"Could not check whether {pair.source} exists: {error}")
        elif not exists:
            missing.append(pair.source)
    return missing


def _bash_command(pairs: list[CopyPair], overwrite: bool) -> str:
    """Build the bash run inside the job.

    Every destination is checked before anything is copied, so a job that would
    clobber one store does not leave the earlier ones half copied.
    """
    lines = ["set -e"]
    for pair in pairs:
        destination = shlex.quote(pair.destination)
        if overwrite:
            lines.append(
                f"if [ -e {destination} ]; then "
                f'echo "Removing existing {pair.destination}"; '
                f"rm -rf {destination}; fi"
            )
        else:
            lines.append(
                f"if [ -e {destination} ]; then "
                f'echo "ERROR: {pair.destination} already exists on WEKA. '
                f'Rerun with --overwrite to replace it." >&2; exit 1; fi'
            )
    for pair in pairs:
        destination = shlex.quote(pair.destination)
        lines.append(f"mkdir -p {destination}")
        lines.append(
            f"gsutil -m -o Credentials:gs_service_key_file={CREDENTIALS} "
            f"rsync -r {shlex.quote(pair.source)} {destination}"
        )
    return "\n".join(lines)


def _gantry_arguments(
    pairs: list[CopyPair],
    command: str,
    name: str,
    workspace: str,
    cluster: str,
    priority: str,
    budget: str,
) -> list[str]:
    description = f"Copy {len(pairs)} zarr store(s) from GCS to WEKA"
    return [
        "gantry",
        "run",
        "--name",
        name,
        "--task-name",
        name,
        "--description",
        description,
        "--docker-image",
        "google/cloud-sdk:slim",
        "--workspace",
        workspace,
        "--priority",
        priority,
        "--cluster",
        cluster,
        "--dataset-secret",
        f"google-credentials:{CREDENTIALS}",
        "--gpus",
        "0",
        "--shared-memory",
        "40GiB",
        "--weka",
        f"{WEKA_SOURCE}:{WEKA_MOUNT}",
        "--budget",
        budget,
        "--no-python",
        "--install",
        "echo 'skipping installation step'",
        "--",
        "bash",
        "-c",
        command,
    ]


def _repository_root() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _load_config(config_yaml: str) -> Config:
    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    return dacite.from_dict(data_class=Config, data=config_data)


@click.command()
@click.option(
    "--config", "config_yaml", type=str, help="Config YAML to take the stores from."
)
@click.option(
    "--source",
    "explicit_sources",
    type=str,
    multiple=True,
    help="A gs:// store to copy. May be repeated. Cannot be used with --config.",
)
@click.option(
    "--destination",
    type=str,
    required=True,
    help=f"Directory under {WEKA_MOUNT} to copy the stores into. The name of each "
    "source store is appended to it.",
)
@click.option(
    "--coarsened",
    is_flag=True,
    help="Copy the time coarsened stores rather than the native resolution ones.",
)
@click.option(
    "--run",
    "runs",
    type=str,
    multiple=True,
    help="Only copy this run from the config. May be repeated. Defaults to all runs.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Delete and replace destination stores that already exist on WEKA.",
)
@click.option(
    "--dry-run", is_flag=True, help="Print what would be copied without submitting."
)
@click.option("--workspace", type=str, default="ai2/ace", show_default=True)
@click.option("--cluster", type=str, default="ai2/phobos", show_default=True)
@click.option("--priority", type=str, default="normal", show_default=True)
@click.option("--budget", type=str, default="ai2/atec-climate", show_default=True)
def main(
    config_yaml: str | None,
    explicit_sources: tuple[str, ...],
    destination: str,
    coarsened: bool,
    runs: tuple[str, ...],
    overwrite: bool,
    dry_run: bool,
    workspace: str,
    cluster: str,
    priority: str,
    budget: str,
):
    """
    Copy zarr stores from GCS to WEKA by submitting a gantry job.

    Each store is copied to <destination>/<store name>. Destinations that
    already exist are not overwritten unless --overwrite is given.
    """
    logging.basicConfig(level=logging.INFO)

    if bool(config_yaml) == bool(explicit_sources):
        raise click.ClickException("Provide exactly one of --config or --source.")
    if config_yaml is not None:
        sources = _sources_from_config(_load_config(config_yaml), coarsened, runs)
        job_name = os.path.basename(config_yaml).removesuffix(".yaml")
    else:
        if coarsened or runs:
            raise click.ClickException(
                "--coarsened and --run only apply when copying from a --config."
            )
        sources = list(explicit_sources)
        job_name = sources[0].rstrip("/").rsplit("/", 1)[-1].removesuffix(".zarr")

    pairs = _copy_pairs(sources, destination)
    click.echo(f"Copying {len(pairs)} store(s) to WEKA:")
    for pair in pairs:
        click.echo(f"  {pair.source} -> {pair.destination}")
    if overwrite:
        click.echo("Existing destination stores will be deleted and replaced.")

    missing = _missing_sources(pairs)
    if missing:
        raise click.ClickException(f"These sources do not exist: {missing}")
    if not overwrite and os.path.isdir(WEKA_MOUNT):
        present = [
            pair.destination for pair in pairs if os.path.exists(pair.destination)
        ]
        if present:
            raise click.ClickException(
                f"These destinations already exist on WEKA: {present}. "
                "Rerun with --overwrite to replace them."
            )

    command = _bash_command(pairs, overwrite)
    if dry_run:
        click.echo("\nDry run, not submitting. Job would run:\n")
        click.echo(command)
        return

    arguments = _gantry_arguments(
        pairs=pairs,
        command=command,
        name=f"copy-to-weka-{job_name}",
        workspace=workspace,
        cluster=cluster,
        priority=priority,
        budget=budget,
    )
    subprocess.run(arguments, cwd=_repository_root(), check=True)


if __name__ == "__main__":
    main()
