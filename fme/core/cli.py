import argparse
import dataclasses
import os
import shutil
from collections.abc import Sequence

import yaml

from fme.core.distributed import Distributed
from fme.core.wandb import WANDB_RUN_ID_FILE, WandB

from .config import update_dict_with_dotlist


@dataclasses.dataclass
class ResumeResultsConfig:
    """Configuration for resuming a previously stopped or finished job.

    Typically only useful for training jobs which have already finished (e.g., to train
    for a larger value of max_epochs than originally configured) or which were stopped
    (e.g., to resume training on different hardware or to change data loader settings
    such as number of data workers).

    WARNING: We typically don't guarantee backwards compatibility for training, so this
    may not work well when resuming old experiments.

    Arguments:
        existing_dir: Directory with existing results to resume from.
        resume_wandb: If true, log to the same WandB job as given in the
            wandb_job_id file in existing_dir, if any.
    """

    existing_dir: str
    resume_wandb: bool = False

    def prepare_directory(self, experiment_dir: str):
        """Recursively copies existing_dir to experiment_dir.

        Arguments:
            experiment_dir: Directory to which existing_dir will be copied. Typically,
                this will be an empty directory which has been configured for saving a
                training job's outputs, such as model checkpoints.
        """
        if not os.path.isdir(self.existing_dir):
            raise ValueError(f"The directory {self.existing_dir} does not exist.")
        dist = Distributed.get_instance()
        if dist.is_root():
            # recursively copy all files in existing_dir to experiment_dir
            shutil.copytree(self.existing_dir, experiment_dir, dirs_exist_ok=True)
            wandb_run_id_path = os.path.join(experiment_dir, WANDB_RUN_ID_FILE)
            if not self.resume_wandb and os.path.exists(wandb_run_id_path):
                os.remove(wandb_run_id_path)

    def verify_wandb_resumption(self, experiment_dir: str):
        wandb = WandB.get_instance()
        if self.resume_wandb and wandb.enabled:
            with open(os.path.join(experiment_dir, WANDB_RUN_ID_FILE)) as f:
                wandb_run_id = f.read().strip()
            if wandb.get_id() != wandb_run_id:
                raise ValueError(
                    f"Expected WandB job ID for resumption is {wandb_run_id} "
                    f"but the actual ID is {wandb.get_id()}. "
                    "Is there a bug in ResumeResultsConfig?"
                )


def prepare_config(path: str, override: Sequence[str] | None = None) -> dict:
    """Get config and update with possible dotlist override."""
    with open(path) as f:
        data = yaml.safe_load(f)
    data = update_dict_with_dotlist(data, override)
    return data


def prepare_directory(
    path: str, config_data: dict, resume_results: ResumeResultsConfig | None = None
) -> ResumeResultsConfig | None:
    """Create experiment directory and dump config_data to it."""
    dist = Distributed.get_instance()
    if not os.path.isdir(path) and dist.is_root():
        os.makedirs(path, exist_ok=True)
    if resume_results is not None and not os.path.isdir(
        os.path.join(path, "training_checkpoints")
    ):
        resume_results.prepare_directory(path)
    else:
        # either not given or ignored because we already resumed once before
        resume_results = None
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    return resume_results


def get_parser():
    """Standard arg parser for ACE entrypoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--override",
        nargs="*",
        help="A dotlist of key=value pairs to override the config. "
        "For example, --override a.b=1 c=2, where a dot indicates nesting.",
    )
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")

    return parser
