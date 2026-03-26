import contextlib
import os
from collections.abc import Generator

import torch

from .typing_ import TensorDict, TensorMapping

_FORCE_CPU: bool = os.environ.get("FME_FORCE_CPU", "0") == "1"


@contextlib.contextmanager
def force_cpu(force: bool = True) -> Generator[None, None, None]:
    """Force the use of CPU even if a GPU is available. This is useful for
    testing and debugging.

    Args:
        force: If True, force the use of CPU. If False, allow the use of GPU if
            available.
    """
    global _FORCE_CPU
    previous = _FORCE_CPU
    try:
        _FORCE_CPU = force
        yield
    finally:
        _FORCE_CPU = previous


def using_gpu() -> bool:
    return get_device().type == "cuda"


def using_srun() -> bool:
    """If using srun instead of torchrun, set FME_USE_SRUN=1 in the environment."""
    if os.environ.get("FME_USE_SRUN", "0") == "1":
        return True
    return False


def get_device() -> torch.device:
    """If CUDA is available, return a CUDA device. Otherwise, return a CPU device
    unless FME_USE_MPS is set, in which case return an MPS device if available.
    """
    if _FORCE_CPU:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    else:
        mps_available = torch.backends.mps.is_available()
        if mps_available and os.environ.get("FME_USE_MPS", "0") == "1":
            return torch.device("mps", 0)
        else:
            return torch.device("cpu")


def move_tensordict_to_device(data: TensorMapping) -> TensorDict:
    device = get_device()
    return {name: value.to(device) for name, value in data.items()}
