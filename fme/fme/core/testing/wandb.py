import collections
import contextlib
from typing import Any, Dict, List, Mapping

from fme.core import wandb
from fme.core.distributed import Distributed


class MockWandB:
    def __init__(self):
        self._enabled = False
        self._configured = False
        self._logs: Dict[int, Dict[str, Any]] = collections.defaultdict(dict)
        self._last_step = 0

    def configure(self, log_to_wandb: bool):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True

    def init(self, **kwargs):
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            pass

    def watch(self, modules):
        if self._enabled:
            # wandb.watch(modules)
            pass

    def log(self, data: Mapping[str, Any], step: int, sleep=None):
        if step < self._last_step:
            raise ValueError(
                f"step {step} is less than last step {self._last_step}, "
                "steps must be logged in order"
            )
        self._last_step = step
        # sleep arg is ignored since we don't want to sleep in tests
        if self._enabled:
            self._logs[step].update(data)

    def get_logs(self) -> List[Dict[str, Any]]:
        if len(self._logs) == 0:
            return []
        n_logs = max(self._logs.keys())
        return_value: List[Dict[str, Any]] = [dict() for _ in range(n_logs + 1)]
        for step, log in self._logs.items():
            return_value[step] = log
        return return_value

    def clean_wandb_dir(self, experiment_dir: str):
        pass

    def Image(self, *args, **kwargs) -> wandb.Image:
        return wandb.Image(*args, direct_access=False, **kwargs)

    def Video(self, *args, **kwargs) -> wandb.Video:
        return wandb.Video(*args, direct_access=False, **kwargs)

    def Table(self, *args, **kwargs) -> wandb.Table:
        return wandb.Table(*args, direct_access=False, **kwargs)

    def Histogram(self, *args, **kwargs) -> wandb.Histogram:
        return wandb.Histogram(*args, direct_access=False, **kwargs)

    @property
    def enabled(self) -> bool:
        return self._enabled


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
