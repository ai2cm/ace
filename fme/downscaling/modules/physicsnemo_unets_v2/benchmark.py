from typing import Self

import torch

from fme.core.benchmark.benchmark import BenchmarkABC, register_benchmark
from fme.core.device import get_device
from fme.core.typing_ import TensorDict

from .unets import SongUNetv2


class SongUNetv2Benchmark(BenchmarkABC):
    def __init__(
        self,
        model: SongUNetv2,
        x: torch.Tensor,
        noise_labels: torch.Tensor,
        class_labels: torch.Tensor | None,
    ):
        self.model = model
        self.x = x
        self.noise_labels = noise_labels
        self.class_labels = class_labels

    def run_instance(self, timer) -> TensorDict:
        result = self.model(
            self.x,
            self.noise_labels,
            self.class_labels,
            timer=timer,
        )
        return {"output": result.detach()}

    @classmethod
    def new(cls) -> Self:
        return cls._new_with_params(
            img_resolution=128,
            B=4,
            in_channels=4,
            out_channels=4,
            label_dim=2,
            model_channels=64,
            channel_mult=[1, 2, 2, 2],
        )

    @classmethod
    def _new_with_params(
        cls,
        img_resolution: int,
        B: int,
        in_channels: int,
        out_channels: int,
        label_dim: int,
        model_channels: int,
        channel_mult: list[int],
    ) -> Self:
        device = get_device()
        model = SongUNetv2(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
            use_apex_gn=False,
        ).to(device)
        model.eval()

        H = W = img_resolution
        x = torch.randn(B, in_channels, H, W, device=device)
        noise_labels = torch.rand(B, device=device)
        class_labels = (
            torch.randn(B, label_dim, device=device) if label_dim > 0 else None
        )

        return cls(
            model=model,
            x=x,
            noise_labels=noise_labels,
            class_labels=class_labels,
        )

    @classmethod
    def new_for_regression(cls) -> Self | None:
        return cls._new_with_params(
            img_resolution=16,
            B=2,
            in_channels=3,
            out_channels=3,
            label_dim=0,
            model_channels=4,
            channel_mult=[1, 2, 2],
        )


register_benchmark("songunetv2")(SongUNetv2Benchmark)
