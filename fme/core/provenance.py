"""Record where an inference job's data output came from.

When data is written to an ``output_dir`` separate from the job's
``experiment_dir`` (e.g. a remote bucket), the job's config and logs are not
next to the data. This module writes a copy of the effective config and a
small ``provenance.json`` into the output directory so the data can be traced
back to the job that produced it.
"""

import dataclasses
import datetime
import json
import os
from typing import Any

import fsspec
import yaml

import fme
from fme.core.cloud import makedirs
from fme.core.distributed import Distributed
from fme.core.wandb import WandB

CONFIG_FILENAME = "config.yaml"
PROVENANCE_FILENAME = "provenance.json"


def write_provenance(output_dir: str, config: Any, experiment_dir: str) -> None:
    """Write ``config.yaml`` and ``provenance.json`` into ``output_dir``.

    Only the root rank writes. Call after logging is configured so the wandb
    run id, if any, is recorded.

    Args:
        output_dir: Directory receiving the files. May be a remote fsspec path.
        config: The effective config dataclass instance of the job.
        experiment_dir: The job directory, recorded in ``provenance.json``.
    """
    if not Distributed.get_instance().is_root():
        return
    makedirs(output_dir, exist_ok=True)
    config_data = dataclasses.asdict(config)
    with fsspec.open(os.path.join(output_dir, CONFIG_FILENAME), "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    logging_config = config_data.get("logging") or {}
    provenance = {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
        "beaker_job_id": os.environ.get("BEAKER_JOB_ID"),
        "wandb": {
            "entity": logging_config.get("entity"),
            "project": logging_config.get("project"),
            "run_id": WandB.get_instance().get_id(),
        },
        "fme_version": fme.__version__,
        "written_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with fsspec.open(os.path.join(output_dir, PROVENANCE_FILENAME), "w") as f:
        json.dump(provenance, f, indent=2)
