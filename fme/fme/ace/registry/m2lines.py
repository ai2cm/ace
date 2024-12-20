import dataclasses
from typing import List, Tuple

from fme.ace.models.ocean.m2lines.samudra import Samudra
from fme.ace.registry.registry import ModuleConfig, ModuleSelector


@ModuleSelector.register("Samudra")
@dataclasses.dataclass
class SamudraBuilder(ModuleConfig):
    """
    Configuration for the M2Lines Samudra architecture.
    """

    ch_width: List[int] = dataclasses.field(
        default_factory=lambda: [200, 250, 300, 400]
    )
    n_layers: List[int] = dataclasses.field(default_factory=lambda: [1, 1, 1, 1])
    dilation: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 4, 8])
    pred_residuals: bool = False
    pad: str = "circular"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ):
        return Samudra(
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            ch_width=self.ch_width,
            dilation=self.dilation,
            n_layers=self.n_layers,
            pred_residuals=self.pred_residuals,
            pad=self.pad,
        )
