import collections
from typing import Any, Mapping, Dict
from fme.core.distributed import Distributed
import contextlib
import wandb as upstream_wandb
from fme.core import wandb


class MockWandB:
    def __init__(self):
        self._enabled = False
        self._configured = False
        self._logs: Dict[int, Dict[str, Any]] = collections.defaultdict(dict)

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
            # wandb.init(
            #     config=config,
            #     project=project,
            #     entity=entity,
            #     resume=resume,
            #     dir=dir,
            # )
            pass

    def watch(self, modules):
        if self._enabled:
            # wandb.watch(modules)
            pass

    def log(self, data: Mapping[str, Any], step: int):
        if self._enabled:
            self._logs[step].update(data)

    def get_logs(self) -> Dict[int, Dict[str, Any]]:
        return self._logs

    @property
    def Image(self):
        return upstream_wandb.Image

    @property
    def Video(self):
        return upstream_wandb.Video

    @property
    def Table(self):
        return upstream_wandb.Table


@contextlib.contextmanager
def mock_wandb():
    """
    Mock the distributed singleton to return a MockDistributed object.

    This is useful for testing that metrics are reduced across processes.

    It will make it so that when any tensor is reduced, it is filled with
    the given fill_value, which can be checked for in tests.
    """
    original = wandb.singleton
    wandb.singleton = MockWandB()  # type: ignore
    try:
        yield wandb.singleton
    finally:
        wandb.singleton = original
