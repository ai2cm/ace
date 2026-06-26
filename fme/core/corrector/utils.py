import dataclasses

import torch

from fme.core.corrector.state import CorrectorState
from fme.core.typing_ import TensorDict, TensorMapping


def force_positive(data: TensorMapping, names: list[str]) -> TensorDict:
    """Clamp all tensors defined by `names` to be greater than or equal to zero."""
    out = {**data}
    for name in names:
        out[name] = torch.clamp(data[name], min=0.0)
    return out


@dataclasses.dataclass
class ForcePositive:
    """Correction that clamps the named generated fields to be non-negative.

    Implements the ``Correction`` protocol; ``input_data``, ``forcing_data`` and
    ``corrector_state`` are unused and passed through.
    """

    names: list[str]

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> tuple[TensorDict, CorrectorState | None]:
        return force_positive(gen_data, self.names), corrector_state
