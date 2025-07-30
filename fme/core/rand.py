import contextlib
import os
import random

import numpy as np
import torch

from fme.core.distributed import Distributed

USE_CPU_RANDN = False


def set_seed(seed: int):
    """
    Set the seed for all random number generators, including numpy, random, torch,
    and Distributed.

    Args:
        seed: The seed to set.
    """
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    torch.manual_seed(seed + 3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 4)
    dist = Distributed.get_instance()
    dist.set_seed(seed + 5)


def randn_like(x: torch.Tensor, **kwargs):
    if USE_CPU_RANDN:
        device = kwargs.pop("device", x.device)
        return torch.randn_like(x, device="cpu", **kwargs).to(device)
    else:
        return torch.randn_like(x, **kwargs)


def randn(shape: torch.Size, **kwargs):
    if USE_CPU_RANDN:
        device = kwargs.pop("device", None)
        return torch.randn(shape, device="cpu", **kwargs).to(device)
    else:
        return torch.randn(shape, **kwargs)


@contextlib.contextmanager
def use_cpu_randn():
    """
    Context manager to use CPU when generating random numbers for
    randn and randn_like.

    This is likely less performant than generating them directly on the GPU,
    but it allows comparing regression outputs between machines.
    """
    global USE_CPU_RANDN
    old_use_cpu_randn = USE_CPU_RANDN
    USE_CPU_RANDN = True
    yield
    USE_CPU_RANDN = old_use_cpu_randn
