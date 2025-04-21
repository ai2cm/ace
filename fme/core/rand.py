import contextlib

import torch

USE_CPU_RANDN = False


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
