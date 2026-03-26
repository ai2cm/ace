from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor_dict
from fme.downscaling.samplers import stochastic_sampler


class TinyDeterministicNet(torch.nn.Module):
    """
    A small, deterministic "denoiser" net with the required signature:
        net(x, x_lr, t) -> Tensor  (same shape as x)
    Keeps things stable across devices/dtypes by doing float32 math internally.
    """

    def __init__(self, c_out: int, c_lr: int):
        super().__init__()
        # 1x1 conv over concatenated [x, x_lr] -> c_out
        self.proj = torch.nn.Conv2d(c_out + c_lr, c_out, kernel_size=1, bias=True)
        # Fixed parameters for determinism.
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0.0)

    def forward(self, x: Tensor, x_lr: Tensor, t: Tensor | float) -> Tensor:
        x_f = x.float()
        xlr_f = x_lr.float()
        # Ensure t broadcasts to (B,1,1,1)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if not isinstance(t, float):
            t = t.to(device=x.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.view(1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1, 1, 1)
        # Mild, deterministic conditioning on noise level
        x_in = torch.cat([x_f, xlr_f], dim=1)
        out = self.proj(x_in) * (1.0 / (1.0 + t))
        return out.to(dtype=x.dtype)


def test_stochastic_sampler_regression() -> None:
    device = get_device()
    if device.type == "cuda":
        pytest.skip("Skipping regression test on CUDA to avoid non-determinism issues")

    torch.set_default_dtype(torch.float32)

    B, C_out, C_lr, H, W = 2, 3, 2, 8, 8
    latents = (
        torch.arange(B * C_out * H * W, device=device, dtype=torch.float32).reshape(
            B, C_out, H, W
        )
        / 1000.0
    )
    img_lr = (
        torch.arange(B * C_lr * H * W, device=device, dtype=torch.float32).reshape(
            B, C_lr, H, W
        )
        / 500.0
    )

    net = TinyDeterministicNet(c_out=C_out, c_lr=C_lr).to(device).eval()

    seeded_generator = torch.Generator(device).manual_seed(1234)

    def _deterministic_randn_like(x: Tensor) -> Tensor:
        # Force the sampler to be deterministic regardless of RNG state/device.
        return torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=seeded_generator
        )

    out, latent_steps = stochastic_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        randn_like=_deterministic_randn_like,
        num_steps=6,  # keep it fast
        sigma_min=0.01,  # avoid extreme ratios
        sigma_max=1.0,
        rho=3.0,
        S_churn=0.0,
        S_noise=1.0,
    )

    # Basic invariants
    assert out.shape == latents.shape
    assert out.dtype == latents.dtype
    assert len(latent_steps) == 6 + 1
    assert latent_steps[0].shape == latents.shape
    assert latent_steps[-1].shape == latents.shape

    # Regression checks (store under a stable path near this test file)
    reg_dir = Path(__file__).with_suffix("").parent / "testdata"
    reg_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "out": out,
        "latent_steps": torch.stack(latent_steps, dim=0),
    }

    validate_tensor_dict(data, str(reg_dir / f"test_stochastic_sampler_regression.pt"))


def test_stochastic_sampler_raises_on_batch_mismatch() -> None:
    device = get_device()
    net = TinyDeterministicNet(c_out=1, c_lr=1).to(device).eval()
    latents = torch.zeros(2, 1, 4, 4, device=device)
    img_lr = torch.zeros(1, 1, 4, 4, device=device)  # mismatched batch

    with pytest.raises(ValueError, match=r"same batch size"):
        stochastic_sampler(
            net=net,
            latents=latents,
            img_lr=img_lr,
            num_steps=3,
            sigma_min=0.01,
            sigma_max=1.0,
            rho=3.0,
        )
