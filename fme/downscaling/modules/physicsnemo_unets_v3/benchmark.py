"""
Benchmarks for SongUNetv3, including a compiled variant that tracks
torch.compile overhead separately.
"""

import time
from typing import Self

import torch

from fme.core.benchmark.benchmark import BenchmarkABC, register_benchmark
from fme.core.device import get_device
from fme.core.typing_ import TensorDict

from .unets import SongUNetv3


class SongUNetv3Benchmark(BenchmarkABC):
    """Benchmark for SongUNetv3 without torch.compile."""

    def __init__(
        self,
        model: SongUNetv3,
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
            img_resolution=64,
            B=1,
            in_channels=3,
            out_channels=2,
            label_dim=0,
            model_channels=16,
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
        compile_model: bool = False,
    ) -> Self:
        device = get_device()
        model = SongUNetv3(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_blocks=4,
            attn_resolutions=[],
            dropout=0.0,
            use_apex_gn=False,
            compile_model=compile_model,
        ).to(device)
        model.eval()

        H = W = img_resolution
        x = torch.randn(B, in_channels, H, W, device=device)
        noise_labels = torch.rand(B, device=device)
        class_labels = (
            torch.randn(B, label_dim, device=device) if label_dim > 0 else None
        )

        instance = cls(
            model=model,
            x=x,
            noise_labels=noise_labels,
            class_labels=class_labels,
        )

        # For compiled models, trigger compilation by running a warmup forward
        # and record the compilation time.
        if compile_model:
            with torch.no_grad():
                t0 = time.perf_counter()
                model(x, noise_labels, class_labels)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
            instance._compile_time_s = t1 - t0
        else:
            instance._compile_time_s = None

        return instance

    @classmethod
    def new_for_regression(cls) -> Self | None:
        return cls._new_with_params(
            img_resolution=16,
            B=1,
            in_channels=3,
            out_channels=2,
            label_dim=0,
            model_channels=16,
            channel_mult=[1, 2],
        )


class SongUNetv3BenchmarkCompiled(SongUNetv3Benchmark):
    """Benchmark for SongUNetv3 with torch.compile enabled."""

    def run_instance(self, timer) -> TensorDict:
        result = self.model(
            self.x,
            self.noise_labels,
            self.class_labels,
            timer=timer,
        )
        out = {"output": result.detach()}
        if self._compile_time_s is not None:
            out["compile_time_s"] = torch.tensor(self._compile_time_s)
        return out

    @classmethod
    def new(cls) -> Self:
        return cls._new_with_params(
            img_resolution=64,
            B=1,
            in_channels=3,
            out_channels=2,
            label_dim=0,
            model_channels=16,
            channel_mult=[1, 2, 2, 2],
            compile_model=True,
        )

    @classmethod
    def new_for_regression(cls) -> Self | None:
        return cls._new_with_params(
            img_resolution=16,
            B=1,
            in_channels=3,
            out_channels=2,
            label_dim=0,
            model_channels=16,
            channel_mult=[1, 2],
            compile_model=True,
        )


register_benchmark("songunetv3")(SongUNetv3Benchmark)
register_benchmark("songunetv3_compiled")(SongUNetv3BenchmarkCompiled)
