import dataclasses
from typing import Literal

from fme.ace.models.land.land_net import LandNet
from fme.ace.registry.registry import ModuleConfig, ModuleSelector


@ModuleSelector.register("LandNet")
@dataclasses.dataclass
class LandNetBuilder(ModuleConfig):
    """
    Configuration for the LandNet architecture.
    """

    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [64, 64])
    network_type: Literal["MLP"] = "MLP"
    use_positional_embedding: bool = False

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
    ):
        assert self.network_type in ["MLP"], "network_type must be MLP"

        return LandNet(
            img_shape=img_shape,
            input_channels=n_in_channels,
            output_channels=n_out_channels,
            hidden_dims=self.hidden_dims,
            network_type=self.network_type,
            use_positional_embedding=self.use_positional_embedding,
        )
