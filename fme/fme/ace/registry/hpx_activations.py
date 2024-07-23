import dataclasses
from typing import Literal

import torch.nn as nn

from fme.ace.models.healpix.healpix_activations import AvgPool, CappedGELU, MaxPool
from fme.ace.models.healpix.healpix_recunet import HEALPixBlockConfig

# Helper configs - activations


@dataclasses.dataclass
class DownsamplingBlockConfig(HEALPixBlockConfig):
    """
    Configuration for the downsampling block.
    Generally, either a pooling block or a striding conv block.

    Attributes:
        block_type: Type of recurrent block, either "MaxPool" or "AvgPool"
        pooling: Pooling size
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.

    """

    block_type: Literal["MaxPool", "AvgPool"]
    pooling: int = 2
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the recurrent block model.

        Returns:
            Recurrent block.
        """
        if self.block_type == "MaxPool":
            return MaxPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )

        elif self.block_type == "AvgPool":
            return AvgPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class CappedGELUConfig(HEALPixBlockConfig):
    """
    Configuration for the CappedGELU activation function.

    Attributes:
        cap_value: Cap value for the GELU function, default is 10.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
    """

    cap_value: int = 10
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the CappedGELU activation function.

        Returns:
            CappedGELU activation function.
        """
        return CappedGELU(cap_value=self.cap_value)
