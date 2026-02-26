from typing import Self

import torch
from torch_harmonics import InverseRealSHT, RealSHT

from fme.core.benchmark.benchmark import BenchmarkABC, register_benchmark
from fme.core.benchmark.timer import Timer
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.conditional_sfno.sfnonet import FourierNeuralOperatorBlock
from fme.core.typing_ import TensorDict
from fme.sht_fix import InverseRealSHT, RealSHT


def get_block_benchmark(filter_num_groups: int) -> type[BenchmarkABC]:
    class BlockBenchmark(BenchmarkABC):
        def __init__(
            self, block: FourierNeuralOperatorBlock, x: torch.Tensor, context: Context
        ):
            self.block = block
            self.x = x
            self.context = context

        def run_instance(self, timer: Timer) -> TensorDict:
            result = self.block(self.x, self.context, timer=timer)
            return {"output": result.detach()}

        @classmethod
        def new(cls) -> Self:
            B = 2
            C = 512
            H = 180
            L = 360
            G = filter_num_groups
            conditional_embed_dim_noise = 64
            conditional_embed_dim_labels = 3
            conditional_embed_dim_pos = 32
            return cls._new_with_params(
                B=B,
                C=C,
                H=H,
                L=L,
                G=G,
                conditional_embed_dim_noise=conditional_embed_dim_noise,
                conditional_embed_dim_labels=conditional_embed_dim_labels,
                conditional_embed_dim_pos=conditional_embed_dim_pos,
            )

        @classmethod
        def _new_with_params(
            cls,
            B: int,
            C: int,
            H: int,
            L: int,
            G: int,
            conditional_embed_dim_noise: int,
            conditional_embed_dim_labels: int,
            conditional_embed_dim_pos: int,
        ) -> Self:
            G = filter_num_groups
            device = get_device()
            conditional_embed_dim_scalar = 0
            embedding_scalar = None
            context_embedding_noise = torch.randn(
                B, conditional_embed_dim_noise, H, L
            ).to(device)
            context_embedding_labels = torch.randn(B, conditional_embed_dim_labels).to(
                device
            )
            context_embedding_pos = torch.randn(B, conditional_embed_dim_pos, H, L).to(
                device
            )
            context = Context(
                embedding_scalar=embedding_scalar,
                embedding_pos=context_embedding_pos,
                noise=context_embedding_noise,
                labels=context_embedding_labels,
            )
            x = torch.randn(B, C, H, L, device=get_device())
            forward = RealSHT(nlat=H, nlon=L)
            inverse = InverseRealSHT(nlat=H, nlon=L)
            context_config = ContextConfig(
                embed_dim_scalar=conditional_embed_dim_scalar,
                embed_dim_noise=conditional_embed_dim_noise,
                embed_dim_labels=conditional_embed_dim_labels,
                embed_dim_pos=conditional_embed_dim_pos,
            )
            block = FourierNeuralOperatorBlock(
                forward_transform=forward,
                inverse_transform=inverse,
                img_shape=(H, L),
                embed_dim=C,
                filter_type="linear",
                operator_type="dhconv",
                use_mlp=True,
                context_config=context_config,
                filter_num_groups=G,
            ).to(device)
            return cls(block=block, x=x, context=context)

        @classmethod
        def new_for_regression(cls):
            B = 1
            C = 16
            H = 9
            L = 18
            G = 2
            conditional_embed_dim_noise = 4
            conditional_embed_dim_labels = 3
            conditional_embed_dim_pos = 2
            return cls._new_with_params(
                B=B,
                C=C,
                H=H,
                L=L,
                G=G,
                conditional_embed_dim_noise=conditional_embed_dim_noise,
                conditional_embed_dim_labels=conditional_embed_dim_labels,
                conditional_embed_dim_pos=conditional_embed_dim_pos,
            )

    return BlockBenchmark


register_benchmark("csfno_block")(get_block_benchmark(filter_num_groups=1))
register_benchmark("csfno_block_8_groups")(get_block_benchmark(filter_num_groups=8))
