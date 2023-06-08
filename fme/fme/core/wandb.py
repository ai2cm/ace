from typing import Any, Mapping, Optional
import wandb

from fme.core.distributed import Distributed

singleton: Optional["WandB"] = None


class WandB:
    """
    A singleton class to interface with Weights and Biases (WandB).
    """

    @classmethod
    def get_instance(cls) -> "WandB":
        """
        Get the singleton instance of the WandB class.
        """
        global singleton
        if singleton is None:
            singleton = cls()
        return singleton

    def __init__(self):
        self._enabled = False
        self._configured = False

    def configure(self, log_to_wandb: bool):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True

    def init(
        self,
        config: Mapping[str, Any],
        project: str,
        entity: str,
        resume: bool,
        dir: str,
    ):
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            wandb.init(
                config=config,
                project=project,
                entity=entity,
                resume=resume,
                dir=dir,
            )

    def watch(self, modules):
        if self._enabled:
            wandb.watch(modules)

    def log(self, data: Mapping[str, Any], step=None):
        if self._enabled:
            wandb.log(data, step=step)

    @property
    def Image(self):
        return wandb.Image

    @property
    def Video(self):
        return wandb.Video
