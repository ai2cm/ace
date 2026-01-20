import inspect
from typing import List, Tuple

from physicsnemo.models.diffusion import SongUNet as _SongUNet


class NonDivisibleShapeError(ValueError):
    pass


def validate_shape(x_shape: Tuple[int, int], levels: int):
    """
    Validates that the input shape is divisible by the number of downsampling levels.

    Note that the SongUnet does not downsample the first level, so the
    number of downsamplings considered is the len of channel_mult - 1.
    """
    next_shape = (x_shape[0] // 2, x_shape[1] // 2)
    if next_shape[0] * next_shape[1] * 4 != x_shape[0] * x_shape[1]:
        raise NonDivisibleShapeError(f"Shape {x_shape} is not divisible by {levels} levels")
    elif levels > 2:
        try:
            validate_shape(next_shape, levels - 1)
        except NonDivisibleShapeError:
            raise NonDivisibleShapeError(
                f"Shape {x_shape} is not divisible by {levels} levels"
            )

def check_level_compatibility(
        img_resolution: int,
        channel_mult: List[int],
        attn_resolutions: List[int],
):

    matched_attn = set()
    for i in range(len(channel_mult)):
        res = img_resolution >> i
        if res == 0:
            raise ValueError(
                "Image resolution is not divisible by the number of number of"
                " levels in the U-Net architecture specified by channel_mult"
                f" {channel_mult}."
            )
        if res in attn_resolutions:
            matched_attn.add(res)

    if matched_attn != set(attn_resolutions):
        raise ValueError(
            "Requested attn_resolutions are not compatible with the input"
            f" image resolution. Matched attention resolutions {matched_attn}"
            f" but requested {attn_resolutions}."
        )
    

class SongUNet(_SongUNet):
    def __init__(self, img_resolution: int,*args, **kwargs):
        # defaults from parent class
        self.channel_mult = kwargs.get("channel_mult", [1, 2, 2, 2])
        self.attn_resolution = kwargs.get("attn_resolutions", [16])
        check_level_compatibility(img_resolution, self.channel_mult, self.attn_resolution)
        super().__init__(img_resolution, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        validate_shape(
            x.shape[2:],
            levels=len(self.channel_mult),
        )
        return super().forward(x, *args, **kwargs)
    
    # repeat the docstrings and signature info from SongUNet
    __init__.__doc__ = _SongUNet.__init__.__doc__
    __init__.__signature__ = inspect.signature(_SongUNet.__init__)
    forward.__doc__ = _SongUNet.forward.__doc__
    forward.__signature__ = inspect.signature(_SongUNet.forward)
