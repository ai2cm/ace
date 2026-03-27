import math

import torch

from fme.ace.aggregator.inference.regional_spectrum import compute_isotropic_spectrum


def test_compute_isotropic_spectrum():
    """3D (B, H, W) input with detrend='linear', window='hann', truncate=True."""
    H, W = 64, 64
    B = 3
    kx_cycles = 10
    expected_k = kx_cycles / W

    x = torch.arange(W, dtype=torch.float64)
    y = torch.arange(H, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    signal = torch.sin(2 * math.pi * kx_cycles * X / W)
    data = signal.unsqueeze(0).expand(B, -1, -1).clone()

    k_bins, spectrum = compute_isotropic_spectrum(
        data, detrend="linear", window="hann", truncate=True
    )

    num_bins = min(H, W) // 2 - 1
    assert k_bins.shape == (num_bins,)
    assert spectrum.shape == (B, num_bins)

    torch.testing.assert_close(spectrum[0], spectrum[1])
    torch.testing.assert_close(spectrum[0], spectrum[2])

    spec = spectrum[0].clone()
    spec[torch.isnan(spec)] = 0.0
    peak_idx = spec.argmax()
    peak_k = k_bins[peak_idx].item()
    bin_width = (k_bins[1] - k_bins[0]).item()
    assert abs(peak_k - expected_k) <= 0.5 * bin_width
