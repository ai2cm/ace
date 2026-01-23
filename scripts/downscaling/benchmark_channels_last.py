"""
Benchmark script to test channels_last memory format configurations
and diagnose performance issues.

Usage:
    python -m fme.downscaling.benchmark_channels_last

Note: Run on a machine with a GPU for meaningful benchmark results.
"""

import time
from dataclasses import dataclass

import torch

from fme.core.device import get_device
from fme.downscaling.modules.diffusion_registry import DiffusionModuleRegistrySelector


@dataclass
class BenchmarkConfig:
    name: str
    model_channels_last: bool
    input_channels_last: bool


def check_tensor_format(x: torch.Tensor, name: str = "") -> dict:
    """Check memory format of a tensor."""
    return {
        "name": name,
        "shape": tuple(x.shape),
        "channels_last": x.is_contiguous(memory_format=torch.channels_last),
        "contiguous": x.is_contiguous(),
        "stride": x.stride(),
    }


def print_format_info(info: dict):
    """Pretty print tensor format info."""
    cl = "✅" if info["channels_last"] else "❌"
    c = "✅" if info["contiguous"] else "❌"
    print(f"  {info['name']}: shape={info['shape']}, channels_last={cl}, contiguous={c}")


def build_model(
    model_channels: int = 160,
    channel_mult: list[int] = None,
    fine_shape: tuple[int, int] = (64, 64),
    n_channels: int = 3,
    use_channels_last: bool = True,
):
    """Build a diffusion model for benchmarking."""
    if channel_mult is None:
        channel_mult = [1, 2, 2, 2]

    config = {
        "model_channels": model_channels,
        "channel_mult": channel_mult,
        "attn_resolutions": [],  # Disable attention for simpler benchmark
        "num_blocks": 2,
    }

    selector = DiffusionModuleRegistrySelector(
        type="unet_diffusion_song",
        config=config,
        use_channels_last=use_channels_last,
    )

    # Calculate coarse shape (assuming downscale_factor=1 for simplicity)
    coarse_shape = fine_shape
    downscale_factor = 1

    module = selector.build(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        coarse_shape=coarse_shape,
        downscale_factor=downscale_factor,
        sigma_data=1.0,
    )

    return module


def run_benchmark(
    config: BenchmarkConfig,
    fine_shape: tuple[int, int],
    n_channels: int,
    batch_size: int,
    model_channels: int = 160,
    channel_mult: list[int] = None,
    n_warmup: int = 5,
    n_iterations: int = 20,
) -> dict:
    """Run a single benchmark configuration."""
    if channel_mult is None:
        channel_mult = [1, 2, 2, 2]
    device = get_device()

    print(f"\n{'='*60}")
    print(f"Config: {config.name}")
    print(f"  model_channels_last={config.model_channels_last}")
    print(f"  input_channels_last={config.input_channels_last}")
    print(f"{'='*60}")

    # Build model fresh for each config
    # Note: use_channels_last controls both model parameter format AND
    # whether forward() converts inputs to channels_last
    model = build_model(
        model_channels=model_channels,
        channel_mult=channel_mult,
        fine_shape=fine_shape,
        n_channels=n_channels,
        use_channels_last=config.model_channels_last,
    )

    # Check model parameter formats
    print("\nModel parameter formats:")
    n_cl = 0
    n_total = 0
    for name, param in model.named_parameters():
        if param.ndim == 4:
            n_total += 1
            if param.is_contiguous(memory_format=torch.channels_last):
                n_cl += 1
    print(f"  4D params in channels_last: {n_cl}/{n_total}")

    # Create input tensors (on device via model forward which calls get_device())
    latent = torch.randn(batch_size, n_channels, *fine_shape, device=device)
    conditioning = torch.randn(batch_size, n_channels, *fine_shape, device=device)
    noise_level = torch.randn(batch_size, 1, 1, 1, device=device)

    # Apply channels_last to inputs if requested
    if config.input_channels_last:
        latent = latent.to(memory_format=torch.channels_last)
        conditioning = conditioning.to(memory_format=torch.channels_last)

    # Check input formats
    print("\nInput tensor formats:")
    print_format_info(check_tensor_format(latent, "latent"))
    print_format_info(check_tensor_format(conditioning, "conditioning"))

    # Warmup
    print(f"\nWarmup ({n_warmup} iterations)...")
    model.train()
    for _ in range(n_warmup):
        output = model(latent, conditioning, noise_level)
        loss = output.mean()
        loss.backward()
        model.zero_grad()

    # Check output format after forward
    print("\nOutput tensor format (after forward):")
    with torch.no_grad():
        output = model(latent, conditioning, noise_level)
    print_format_info(check_tensor_format(output, "output"))

    # Benchmark forward pass
    print(f"\nBenchmarking forward pass ({n_iterations} iterations)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_times = []
    with torch.no_grad():
        for _ in range(n_iterations):
            start = time.perf_counter()
            output = model(latent, conditioning, noise_level)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

    forward_mean = sum(forward_times) / len(forward_times) * 1000
    forward_std = (sum((t - forward_mean/1000)**2 for t in forward_times) / len(forward_times))**0.5 * 1000
    print(f"  Forward: {forward_mean:.2f} ± {forward_std:.2f} ms")

    # Benchmark forward + backward pass
    print(f"\nBenchmarking forward+backward ({n_iterations} iterations)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    full_times = []
    for _ in range(n_iterations):
        model.zero_grad()
        start = time.perf_counter()
        output = model(latent, conditioning, noise_level)
        loss = output.mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        full_times.append(time.perf_counter() - start)

    full_mean = sum(full_times) / len(full_times) * 1000
    full_std = (sum((t - full_mean/1000)**2 for t in full_times) / len(full_times))**0.5 * 1000
    print(f"  Forward+Backward: {full_mean:.2f} ± {full_std:.2f} ms")

    return {
        "config": config.name,
        "forward_ms": forward_mean,
        "forward_std_ms": forward_std,
        "full_ms": full_mean,
        "full_std_ms": full_std,
    }


def trace_format_through_forward(model, latent, conditioning, noise_level):
    """
    Trace where channels_last format is lost during forward pass.
    This hooks into intermediate layers to check format.
    """
    print("\n" + "="*60)
    print("Tracing format through forward pass...")
    print("="*60)

    format_breaks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                is_cl = output.is_contiguous(memory_format=torch.channels_last)
                if not is_cl:
                    format_breaks.append(name)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Count total 4D outputs
    total_4d = []
    def count_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                total_4d.append(name)
        return hook
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(count_hook(name)))

    # Run forward pass
    with torch.no_grad():
        _ = model(latent, conditioning, noise_level)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Summary
    n_breaks = len(format_breaks)
    n_total = len(total_4d) // 2  # We registered twice
    print(f"\nFormat breaks: {n_breaks} layers output non-channels_last tensors")

    if format_breaks:
        print("\nFirst 10 layers that break channels_last format:")
        for name in format_breaks[:10]:
            print(f"  ⚠️  {name}")
        
        # Categorize breaks
        norm_breaks = [n for n in format_breaks if "norm" in n.lower()]
        other_breaks = [n for n in format_breaks if "norm" not in n.lower()]
        print(f"\n  GroupNorm breaks: {len(norm_breaks)}")
        print(f"  Other breaks: {len(other_breaks)}")

    return format_breaks


def main():
    device = get_device()
    print(f"Device: {device}")

    if device.type != "cuda":
        print("\n⚠️  WARNING: Running on CPU. Results will not reflect GPU performance.")
        print("   For meaningful benchmarks, run on a machine with a GPU.\n")

    fine_shape = (512, 512)  # Adjust based on your use case
    n_channels = 4
    batch_size = 1  # Reduced for larger model
    model_channels = 128
    channel_mult = [1, 2, 2, 2, 2, 2, 2]

    print(f"Fine shape: {fine_shape}")
    print(f"Channels: {n_channels}")
    print(f"Batch size: {batch_size}")

    # Define benchmark configurations
    configs = [
        BenchmarkConfig("baseline (NCHW)", model_channels_last=False, input_channels_last=False),
        BenchmarkConfig("both (NHWC)", model_channels_last=True, input_channels_last=True),
    ]

    # Run benchmarks
    results = []
    for config in configs:
        result = run_benchmark(
            config=config,
            fine_shape=fine_shape,
            n_channels=n_channels,
            batch_size=batch_size,
            model_channels=model_channels,
            channel_mult=channel_mult,
            n_warmup=2,
            n_iterations=5,
        )
        results.append(result)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config':<20} {'Forward (ms)':<20} {'Full (ms)':<20}")
    print("-"*60)

    baseline_full = results[0]["full_ms"]
    for r in results:
        speedup = (baseline_full - r["full_ms"]) / baseline_full * 100
        speedup_str = f"({speedup:+.1f}%)" if r["config"] != "baseline (NCHW)" else ""
        print(f"{r['config']:<20} {r['forward_ms']:.2f} ± {r['forward_std_ms']:.2f}      {r['full_ms']:.2f} ± {r['full_std_ms']:.2f}  {speedup_str}")

    # Trace format through forward pass for both configs
    for trace_config in configs:
        print(f"\n\nTracing: {trace_config.name}")
        model = build_model(
            model_channels=model_channels,
            channel_mult=channel_mult,
            fine_shape=fine_shape,
            n_channels=n_channels,
            use_channels_last=trace_config.model_channels_last,
        )

        latent = torch.randn(batch_size, n_channels, *fine_shape, device=device)
        conditioning = torch.randn(batch_size, n_channels, *fine_shape, device=device)
        if trace_config.input_channels_last:
            latent = latent.to(memory_format=torch.channels_last)
            conditioning = conditioning.to(memory_format=torch.channels_last)
        noise_level = torch.randn(batch_size, 1, 1, 1, device=device)

        trace_format_through_forward(model, latent, conditioning, noise_level)


if __name__ == "__main__":
    main()
