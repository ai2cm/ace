import collections
import contextlib
import random
import string
from collections.abc import Mapping
from typing import Any, Literal

from fme.core import wandb
from fme.core.distributed import Distributed


class MockWandB:
    def __init__(self):
        self._enabled = False
        self._configured = False
        self._logs: dict[int, dict[str, Any]] = collections.defaultdict(dict)
        self._last_step = 0
        self._id: str | None = None

    def configure(self, log_to_wandb: bool):
        dist = Distributed.get_instance()
        self._enabled = log_to_wandb and dist.is_root()
        self._configured = True

    def init(
        self,
        resumable: bool = False,
        experiment_dir: str | None = None,
        **kwargs,
    ):
        if not self._configured:
            raise RuntimeError(
                "must call WandB.configure before WandB init can be called"
            )
        if self._enabled:
            if resumable:
                if experiment_dir is None:
                    raise ValueError(
                        "must provide `experiment_dir` when `resumable` is True"
                    )
                else:
                    wandb.init_wandb_with_resumption(
                        experiment_dir,
                        direct_access=False,
                        wandb_init=self._wandb_init,
                        wandb_id=self.get_id,
                        **kwargs,
                    )
            else:
                self._wandb_init(resume="never", **kwargs)

    def _wandb_init(
        self, resume: Literal["must", "never"], id: str | None = None, **kwargs
    ):
        """
        Mocks the `wandb.init` behavior, specifically around initializing
        a run with `resume` and `id`.
        See https://docs.wandb.ai/guides/runs/resuming/.
        """
        if resume == "must":
            if id is None:
                raise ValueError("resume='must' and id is None")
            else:
                if id != self._id:
                    raise ValueError("resume='must' and id does not match previous id")
        else:
            if id is not None:
                raise ValueError("resume='never' and id is not None")
            else:
                if self._id is not None:
                    raise ValueError(
                        "resume='never' and id is None but previous id exists"
                    )
            self._id = _mock_wandb_id()

    def get_id(self) -> str:
        if self._id is None:
            raise ValueError("mock wandb id is None")
        return self._id

    def set_id(self, id: str):
        self._id = id

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

    def get_logs(self) -> list[dict[str, Any]]:
        if len(self._logs) == 0:
            return []
        n_logs = max(self._logs.keys())
        return_value: list[dict[str, Any]] = [dict() for _ in range(n_logs + 1)]
        for step, log in self._logs.items():
            return_value[step] = log
        return return_value

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

    @property
    def configured(self) -> bool:
        return self._configured


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


def _mock_wandb_id(n_chars: int = 8) -> str:
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(n_chars)
    )
