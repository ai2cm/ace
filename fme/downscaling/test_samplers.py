from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from fme.core.device import get_device
from fme.core.testing.regression import validate_tensor_dict
from fme.downscaling.samplers import fastgen_sampler, stochastic_sampler


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


class SigmaCapturingNet(torch.nn.Module):
    """Returns zeros and records every (scalar) sigma it was called with."""

    def __init__(self, c_out: int) -> None:
        super().__init__()
        self.c_out = c_out
        self.sigmas_called: list[float] = []

    def forward(self, x: Tensor, x_lr: Tensor, t: Tensor | float) -> Tensor:
        if isinstance(t, Tensor):
            self.sigmas_called.append(float(t.detach().reshape(-1)[0].item()))
        else:
            self.sigmas_called.append(float(t))
        return torch.zeros(x.shape[0], self.c_out, *x.shape[-2:], device=x.device)


def _fastgen_reference_t_list(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    schedule_num_steps: int = 1000,
    min_step_percent: float = 0.002,
    max_step_percent: float = 0.998,
) -> Tensor:
    """Stand-alone reproduction of FastGen's EDMNoiseSchedule.get_t_list."""
    ramp = torch.linspace(0.0, 1.0, schedule_num_steps, dtype=torch.float64)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = torch.flip(
        (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho, dims=[0]
    )
    min_step = int(min_step_percent * schedule_num_steps)
    max_step = int(max_step_percent * schedule_num_steps)
    indices = torch.linspace(max_step, min_step, num_steps + 1).long()
    t_list = sigmas[indices].clone()
    t_list[-1] = 0.0
    return t_list


def test_fastgen_sampler_one_step_matches_net_call() -> None:
    """num_steps=1: output is exactly net(noise * sigma_max, img_lr, sigma_max)."""
    device = get_device()
    B, C_out, C_lr, H, W = 2, 3, 2, 4, 4
    torch.manual_seed(0)
    latents = torch.randn(B, C_out, H, W, device=device)
    img_lr = torch.randn(B, C_lr, H, W, device=device)
    net = TinyDeterministicNet(c_out=C_out, c_lr=C_lr).to(device).eval()

    sigma_max = 80.0
    out, latent_steps = fastgen_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        num_steps=1,
        sigma_min=0.002,
        sigma_max=sigma_max,
        rho=7.0,
    )

    # Reference: a single forward at the first sigma in the t_list, with
    # input = noise * t_list[0]. t_list[0] is near sigma_max but not exactly
    # equal because of FastGen's discrete-index schedule.
    t_list = _fastgen_reference_t_list(1, 0.002, sigma_max, 7.0).to(device)
    expected_input = latents.to(torch.float64) * t_list[0]
    expected = net(expected_input.to(latents.dtype), img_lr, t_list[0])

    assert out.shape == latents.shape
    assert out.dtype == latents.dtype
    assert len(latent_steps) == 2  # initial + final
    torch.testing.assert_close(out, expected.to(out.dtype), rtol=1e-5, atol=1e-5)


def test_fastgen_sampler_calls_net_at_fastgen_sigmas() -> None:
    """The net is called once per step at the first num_steps sigmas of t_list."""
    device = get_device()
    B, C_out, H, W = 2, 1, 4, 4
    latents = torch.zeros(B, C_out, H, W, device=device)
    img_lr = torch.zeros(B, 1, H, W, device=device)
    net = SigmaCapturingNet(c_out=C_out).to(device).eval()

    fastgen_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        num_steps=4,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    )

    expected = _fastgen_reference_t_list(4, 0.002, 80.0, 7.0)[:-1]  # drop trailing 0
    assert len(net.sigmas_called) == 4
    for got, want in zip(net.sigmas_called, expected.tolist()):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


def test_fastgen_sampler_sde_vs_ode_differ() -> None:
    """SDE re-noising and ODE re-noising should produce different trajectories."""
    device = get_device()
    B, C_out, C_lr, H, W = 1, 2, 2, 4, 4
    torch.manual_seed(7)
    latents = torch.randn(B, C_out, H, W, device=device)
    img_lr = torch.randn(B, C_lr, H, W, device=device)
    net = TinyDeterministicNet(c_out=C_out, c_lr=C_lr).to(device).eval()

    gen = torch.Generator(device).manual_seed(42)

    def deterministic_randn_like(x: Tensor) -> Tensor:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)

    out_sde, _ = fastgen_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        randn_like=deterministic_randn_like,
        num_steps=4,
        sigma_min=0.002,
        sigma_max=80.0,
        sde=True,
    )
    out_ode, _ = fastgen_sampler(
        net=net,
        latents=latents,
        img_lr=img_lr,
        num_steps=4,
        sigma_min=0.002,
        sigma_max=80.0,
        sde=False,
    )
    assert not torch.allclose(out_sde, out_ode)


def test_fastgen_sampler_raises_on_batch_mismatch() -> None:
    device = get_device()
    net = TinyDeterministicNet(c_out=1, c_lr=1).to(device).eval()
    latents = torch.zeros(2, 1, 4, 4, device=device)
    img_lr = torch.zeros(1, 1, 4, 4, device=device)

    with pytest.raises(ValueError, match=r"same batch size"):
        fastgen_sampler(
            net=net,
            latents=latents,
            img_lr=img_lr,
            num_steps=2,
            sigma_min=0.01,
            sigma_max=1.0,
        )
