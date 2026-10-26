"""Training-step microbenchmark for the nc-swin-v2.2 speed study.

Builds the noise-conditioned Swin (and the SFNO twin) through the registry at
the FM training shape (4-degree 45x90 grid, 89 input / 50 output channels, 4
labels) and times compiled forward + backward + fused AdamW on random data.
"Samples" counts input samples; each one is run as ``n_ensemble`` (2) model
members, as in training, so the model batch is ``2 * samples``.

Usage:
    python bench_swin.py --flops-only             # FLOPs/params for all variants
    python bench_swin.py base base_bf16 ...       # time the named variants
    python bench_swin.py --profile base           # also print a kernel profile
"""

import argparse
import copy
import dataclasses
import json
import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode

from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector

IMG_SHAPE = (45, 90)
N_IN = 89
N_OUT = 50
LABELS = {"era5", "c96", "cm4", "xshield"}
N_ENSEMBLE = 2

SWIN_BASE: dict[str, Any] = dict(
    depth_multiplier=2,
    drop_path_rate=0.2,
    embed_dim=192,
    mlp_layer="swiglu",
    mlp_ratio=4.0,
    noise_embed_dim=32,
    num_heads=[3, 6, 6, 3],
    num_levels=2,
    padding_conf=dict(activate=True, mode="earth", pad_lat=[2, 1], pad_lon=[3, 3]),
    patch_size=[1, 1],
    skip_projection=True,
    use_skip=True,
    window_size=[4, 8],
)
SFNO_BASE: dict[str, Any] = dict(
    affine_norms=True,
    clip_latent_global_means=True,
    embed_dim=512,
    filter_num_groups=16,
    filter_type="linear",
    noise_embed_dim=32,
    noise_type="isotropic",
    normalize_big_skip=True,
    num_layers=8,
    operator_type="dhconv",
    spectral_ratio=0.125,
    use_mlp=True,
)


@dataclasses.dataclass
class Variant:
    model: dict[str, Any]
    type: str = "NoiseConditionedSwinTransformer"
    samples: int = 4  # input samples per GPU per step (training: 8 global / 2)
    amp: bool = False  # bf16 autocast
    compile_mode: str | None = None


def _swin(samples: int = 4, amp: bool = False, mode=None, **overrides) -> Variant:
    model = copy.deepcopy(SWIN_BASE)
    model.update(overrides)
    return Variant(model=model, samples=samples, amp=amp, compile_mode=mode)


VARIANTS: dict[str, Variant] = {
    "sfno": Variant(model=SFNO_BASE, type="NoiseConditionedSFNO"),
    "sfno_bf16": Variant(model=SFNO_BASE, type="NoiseConditionedSFNO", amp=True),
    "base": _swin(),
    "base_bf16": _swin(amp=True),
    "base_s8": _swin(samples=8),
    "base_s8_bf16": _swin(samples=8, amp=True),
    "base_nolat": _swin(use_cpb_scaling=False),
    "base_bf16_nolat": _swin(amp=True, use_cpb_scaling=False),
    "base_gelu": _swin(mlp_layer="mlp"),
    "base_maxat": _swin(mode="max-autotune-no-cudagraphs"),
    "base_cg": _swin(mode="max-autotune"),
    "base_bf16_cg": _swin(amp=True, mode="max-autotune"),
    # fewer, wider blocks (16 instead of 32)
    "dm1_e256": _swin(depth_multiplier=1, embed_dim=256, num_heads=[4, 8, 8, 4]),
    "dm1_e256_bf16": _swin(
        amp=True, depth_multiplier=1, embed_dim=256, num_heads=[4, 8, 8, 4]
    ),
    "dm1_e256_s8": _swin(
        samples=8, depth_multiplier=1, embed_dim=256, num_heads=[4, 8, 8, 4]
    ),
    "dm1_e256_cg": _swin(
        mode="max-autotune", depth_multiplier=1, embed_dim=256, num_heads=[4, 8, 8, 4]
    ),
    "dm1_e256_ro": _swin(
        mode="reduce-overhead",
        depth_multiplier=1,
        embed_dim=256,
        num_heads=[4, 8, 8, 4],
    ),
    "base_ro": _swin(mode="reduce-overhead"),
    "dm1_e288": _swin(depth_multiplier=1, embed_dim=288, num_heads=[3, 6, 6, 3]),
    # patch 2x2: top level on a 24x48 token grid
    "p2_nl1_e256": _swin(
        patch_size=[2, 2], num_levels=1, embed_dim=256, num_heads=[4, 8, 8, 4]
    ),
    "p2_nl1_e256_bf16": _swin(
        amp=True,
        patch_size=[2, 2],
        num_levels=1,
        embed_dim=256,
        num_heads=[4, 8, 8, 4],
    ),
    "p2_nl1_e256_s8_bf16": _swin(
        samples=8,
        amp=True,
        patch_size=[2, 2],
        num_levels=1,
        embed_dim=256,
        num_heads=[4, 8, 8, 4],
    ),
    "p2_nl1_e288": _swin(
        patch_size=[2, 2], num_levels=1, embed_dim=288, num_heads=[3, 6, 6, 3]
    ),
    "p2_nl1_e320": _swin(
        patch_size=[2, 2], num_levels=1, embed_dim=320, num_heads=[5, 10, 10, 5]
    ),
    "p2_nl1_e320_bf16": _swin(
        amp=True,
        patch_size=[2, 2],
        num_levels=1,
        embed_dim=320,
        num_heads=[5, 10, 10, 5],
    ),
    "p2_nl1_e384": _swin(
        patch_size=[2, 2], num_levels=1, embed_dim=384, num_heads=[6, 12, 12, 6]
    ),
    "p2_nl1_e384_bf16": _swin(
        amp=True,
        patch_size=[2, 2],
        num_levels=1,
        embed_dim=384,
        num_heads=[6, 12, 12, 6],
    ),
    # bigger windows (72 tokens) at the same width
    "w6x12": _swin(window_size=[6, 12]),
    "w6x12_bf16": _swin(amp=True, window_size=[6, 12]),
    # head dim 96/128 (fewer heads)
    "base_hd96": _swin(num_heads=[2, 4, 4, 2]),
}


def _dataset_info(device: torch.device) -> DatasetInfo:
    lat = torch.linspace(-88.0, 88.0, IMG_SHAPE[0], device=device)
    lon = torch.linspace(0.0, 356.0, IMG_SHAPE[1], device=device)
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(lat=lat, lon=lon),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(9, device=device, dtype=torch.float32),
            bk=torch.arange(9, device=device, dtype=torch.float32),
        ),
        all_labels=LABELS,
    )


def build(v: Variant, device: torch.device) -> torch.nn.Module:
    selector = ModuleSelector(type=v.type, config=v.model, conditional=True)
    module = selector.build(N_IN, N_OUT, _dataset_info(device))
    return module.torch_module.to(device)


def _inputs(v: Variant, device: torch.device):
    b = v.samples * N_ENSEMBLE
    x = torch.randn(b, N_IN, *IMG_SHAPE, device=device)
    labels = torch.nn.functional.one_hot(
        torch.arange(b, device=device) % len(LABELS), len(LABELS)
    ).float()
    target = torch.randn(b, N_OUT, *IMG_SHAPE, device=device)
    return x, labels, target


def flops(name: str, v: Variant, device: torch.device) -> dict[str, Any]:
    net = build(v, device)
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(1, N_IN, *IMG_SHAPE, device=device)
    labels = torch.zeros(1, len(LABELS), device=device)
    labels[0, 0] = 1
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        net(x, labels)
    return dict(
        name=name,
        params_M=round(n_params / 1e6, 2),
        fwd_gflop_per_member=round(counter.get_total_flops() / 1e9, 1),
    )


def _categorize(kernel: str) -> str:
    k = kernel.lower()
    if any(s in k for s in ("fmha", "flash", "attention", "efficient_attention")):
        return "attention"
    if any(s in k for s in ("gemm", "cutlass", "sm90_", "sm80_", "cublas", "ampere")):
        return "matmul"
    if "conv" in k or "implicit" in k:
        return "conv"
    if "triton" in k:
        return "triton (fused elementwise/norm/reduction)"
    if "adam" in k or "multi_tensor" in k:
        return "optimizer"
    if "memcpy" in k or "memset" in k:
        return "memcpy/memset"
    if "fft" in k:
        return "fft"
    return "other aten"


def time_variant(
    name: str,
    v: Variant,
    device: torch.device,
    steps: int,
    profile: bool,
    ema: str = "none",
    hooks: bool = False,
) -> dict[str, Any]:
    torch._dynamo.reset()
    torch.manual_seed(0)
    net = build(v, device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, fused=True, weight_decay=0.01)
    params = [p for p in net.parameters() if p.requires_grad]
    if ema == "repo":
        from fme.core.ema import EMATracker

        tracker = EMATracker(net, decay=0.999, faster_decay_at_start=True)
    ema_params = [p.detach().clone() for p in params]
    n_hook_calls = [0]
    if hooks:  # stand-in for wandb.watch's per-parameter gradient hooks

        def _hook(grad):
            n_hook_calls[0] += 1
            return None

        for p in params:
            p.register_hook(_hook)
    compiled = torch.compile(net, mode=v.compile_mode)
    x, labels, target = _inputs(v, device)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=v.amp):
            out = compiled(x, labels)
        loss = (out.float() - target).abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ema == "repo":
            tracker(net)
        elif ema == "foreach":
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, params, 1.0 - 0.999)
        return loss

    t0 = time.time()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        t0 = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        times.append((time.time() - t0) / steps)
    ms = 1000 * min(times)
    result = dict(
        name=name,
        samples_per_gpu_step=v.samples,
        amp=v.amp,
        mode=v.compile_mode,
        ema=ema,
        hooks=hooks,
        ms_per_step=round(ms, 1),
        samples_per_s=round(v.samples / (ms / 1000), 1),
        peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 1),
        compile_s=round(compile_s),
    )
    if profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprofile

        n_prof = 10
        with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(n_prof):
                step()
            torch.cuda.synchronize()
        by_cat: dict[str, float] = defaultdict(float)
        by_kernel: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        n_kernels = 0
        for evt in p.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                dur = evt.device_time / 1000.0 / n_prof  # ms per step
                by_cat[_categorize(evt.name)] += dur
                by_kernel[evt.name][0] += dur
                by_kernel[evt.name][1] += 1
                n_kernels += 1
        busy = sum(by_cat.values())
        result["gpu_busy_ms_per_step"] = round(busy, 1)
        result["kernels_per_step"] = n_kernels // n_prof
        result["by_category_ms"] = {
            k: round(val, 1) for k, val in sorted(by_cat.items(), key=lambda t: -t[1])
        }
        top = sorted(by_kernel.items(), key=lambda t: -t[1][0])[:25]
        print(f"--- top kernels for {name} (ms/step, launches/step) ---")
        for kname, (d, n) in top:
            print(f"{d:8.2f} {n // n_prof:5d}  {kname[:150]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--flops-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ema", choices=["none", "repo", "foreach"], default="none")
    parser.add_argument("--hooks", action="store_true")
    args = parser.parse_args()
    names = args.variants or list(VARIANTS)
    if args.flops_only:
        device = torch.device("cpu")
        for name in names:
            print(json.dumps(flops(name, VARIANTS[name], device)), flush=True)
        return
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    print(torch.__version__, torch.cuda.get_device_name(), flush=True)
    for name in names:
        try:
            info = flops(name, VARIANTS[name], device)
            info.update(
                time_variant(
                    name,
                    VARIANTS[name],
                    device,
                    args.steps,
                    args.profile,
                    ema=args.ema,
                    hooks=args.hooks,
                )
            )
            print("RESULT " + json.dumps(info), flush=True)
            torch.cuda.empty_cache()
        except Exception as err:  # keep going on OOM or compile failure
            print(f"FAILED {name}: {type(err).__name__}: {err}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
