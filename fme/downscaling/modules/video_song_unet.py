"""Video backbone: ACE/HiRO's ``SongUNetv2`` with cBottle-style temporal
attention injected via forward hooks on the attention-resolution ``UNetBlock``s.
"""

import torch
import torch.nn as nn

from fme.downscaling.modules.physicsnemo_unets_v2 import SongUNetv2
from fme.downscaling.modules.physicsnemo_unets_v2.layers import Conv2d as _Conv2d
from fme.downscaling.modules.physicsnemo_unets_v2.layers import UNetBlock
from fme.downscaling.modules.video_modules import CalendarEmbedding, TemporalAttention


class VideoSongUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_resolution: list[int],
        seq_length: int,
        model_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        num_blocks: int = 2,
        n_heads: int = 8,
        attn_resolutions: tuple[int, ...] = (22,),
        num_freqs: int = 4,
        periodic: bool = True,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.calendar = CalendarEmbedding(num_freqs)
        self.unet = SongUNetv2(
            img_resolution=list(img_resolution),
            in_channels=in_channels + self.calendar.out_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=list(channel_mult),
            num_blocks=num_blocks,
            attn_resolutions=list(attn_resolutions),
            use_apex_gn=False,
        )
        if periodic:
            for m in self.unet.modules():
                if isinstance(m, _Conv2d):
                    m.periodic = True

        self._temporal = nn.ModuleList()
        self._bt = (1, 1)
        for name, block in list(self.unet.enc.items()) + list(self.unet.dec.items()):
            if isinstance(block, UNetBlock) and getattr(block, "attention", False):
                ta = TemporalAttention(block.out_channels, n_heads, seq_length)
                self._temporal.append(ta)
                block.register_forward_hook(self._make_hook(ta))

    def _make_hook(self, temporal: nn.Module):
        def hook(module, inputs, output):
            b, t = self._bt
            bt, c, h, w = output.shape
            x = output.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            x = temporal(x)
            return x.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)

        return hook

    def forward(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        day_of_year: torch.Tensor,
        second_of_day: torch.Tensor,
        lon: torch.Tensor,
    ) -> torch.Tensor:
        b, _, t, h, w = x.shape
        cal = self.calendar(day_of_year, second_of_day, lon, h)
        inp = torch.cat([x, cal], dim=1)
        folded = inp.permute(0, 2, 1, 3, 4).reshape(b * t, inp.shape[1], h, w)
        noise = c_noise.reshape(-1)
        if noise.shape[0] == b:
            noise = noise.repeat_interleave(t)
        self._bt = (b, t)
        out = self.unet(folded, noise, None)
        return out.reshape(b, t, -1, h, w).permute(0, 2, 1, 3, 4)
