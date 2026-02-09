import torch

from fme.core.benchmark.benchmark import BenchmarkFn, register_benchmark
from fme.core.benchmark.timer import Timer
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.conditional_sfno.sfnonet import FourierNeuralOperatorBlock
from fme.core.models.conditional_sfno.sht import InverseRealSHT, RealSHT
from fme.core.typing_ import TensorDict


def run_block(filter_num_groups: int) -> BenchmarkFn:
    def _run_block(timer: Timer) -> TensorDict:
        with timer.child("setup_inputs"):
            B = 2
            C = 512
            H = 180
            L = 360
            G = filter_num_groups
            device = get_device()
            conditional_embed_dim_scalar = 0
            conditional_embed_dim_noise = 64
            conditional_embed_dim_labels = 3
            conditional_embed_dim_pos = 32
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
        with timer.child("setup_block"):
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
        with timer.child("call_block") as child_timer:
            result = block(x, context, timer=child_timer)
        return {"output": result[0].detach(), "residual": result[1].detach()}

    return _run_block


register_benchmark("csfno_block")(run_block(filter_num_groups=1))
register_benchmark("csfno_block_8_groups")(run_block(filter_num_groups=8))
