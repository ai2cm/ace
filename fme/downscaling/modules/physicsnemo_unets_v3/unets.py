"""
SongUNetv3: SongUNetv2 with optional torch.compile support.

This is a thin wrapper around SongUNetv2 that adds the ability to compile
the forward pass using torch.compile for potential speedups on supported
hardware.
"""

import torch

from fme.core.benchmark.timer import NullTimer, Timer
from fme.downscaling.modules.physicsnemo_unets_v2.unets import SongUNetv2


class SongUNetv3(SongUNetv2):
    """SongUNetv2 with optional torch.compile support.

    All parameters are identical to SongUNetv2, with the addition of
    ``compile_model`` which, when True, compiles the parent's forward
    method using ``torch.compile(fullgraph=False)``.

    Parameters
    ----------
    *args
        Positional arguments forwarded to SongUNetv2.
    compile_model : bool, optional
        If True, the forward pass is compiled via torch.compile.
        Default is False.
    **kwargs
        Keyword arguments forwarded to SongUNetv2.
    """

    def __init__(self, *args, compile_model: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_model = compile_model
        if compile_model:
            self._compiled_forward = torch.compile(
                super().forward, fullgraph=False
            )

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        augment_labels=None,
        timer: Timer = NullTimer(),
    ):
        if self._compile_model:
            return self._compiled_forward(
                x, noise_labels, class_labels, augment_labels, timer=timer
            )
        return super().forward(
            x, noise_labels, class_labels, augment_labels, timer=timer
        )
