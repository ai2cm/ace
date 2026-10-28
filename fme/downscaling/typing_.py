import dataclasses
from collections.abc import Callable
from typing import Generic, TypeVar

import torch

T = TypeVar("T")

# net(x, x_lr, sigma) -> denoised x, as called by the EDM sampler
Denoiser = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclasses.dataclass
class FineResCoarseResPair(Generic[T]):
    fine: T
    coarse: T
