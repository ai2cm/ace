import os

import torch

from .typing_ import TensorDict, TensorMapping


def using_gpu() -> bool:
    return get_device().type == "cuda"


def get_device() -> torch.device:
    """If CUDA is available, return a CUDA device. Otherwise, return a CPU device
    unless FME_USE_MPS is set, in which case return an MPS device if available.
    """
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
