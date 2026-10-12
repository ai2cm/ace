"""Record where a job's data output came from.

When a job writes its data output to a directory separate from its job
directory (e.g. a remote bucket), the job's config and logs are not next to
the data. This module writes a copy of the config and a small provenance file
into the output directory so the data can be traced back to the job.
"""

import json
import os
from collections.abc import Mapping
from typing import Any

import fsspec
import yaml

from fme.core.distributed import Distributed
from fme.core.logging_utils import retrieve_env_vars
from fme.core.wandb import WandB

CONFIG_FILENAME = "config.yaml"
PROVENANCE_FILENAME = "provenance.json"


def write_provenance(
    output_dir: str, config_data: Mapping[str, Any], experiment_dir: str
) -> None:
    """Write ``config.yaml`` and ``provenance.json`` into ``output_dir``.

    Only the root rank writes. Call after wandb is initialized so the run id
    is recorded.

    Args:
        output_dir: Directory receiving the files. May be a remote fsspec path.
        config_data: The job's configuration, dumped as ``config.yaml``.
        experiment_dir: The job directory, recorded in ``provenance.json``.
    """
    if not Distributed.get_instance().is_root():
        return
    with fsspec.open(os.path.join(output_dir, CONFIG_FILENAME), "w") as f:
        yaml.dump(dict(config_data), f, default_flow_style=False, sort_keys=False)
    provenance = {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "beaker_experiment_id": os.environ.get("BEAKER_EXPERIMENT_ID"),
        "wandb_run_id": WandB.get_instance().get_id(),
        "environment": retrieve_env_vars(),
    }
    with fsspec.open(os.path.join(output_dir, PROVENANCE_FILENAME), "w") as f:
        json.dump(provenance, f, indent=2)
