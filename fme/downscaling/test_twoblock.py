import pytest
import torch

from fme.downscaling.twoblock import (
    assemble_fine_output,
    block_replicate_upsample,
    coarse_temporal_interp,
    conservative_downsample,
    d_target,
    null_space_projector,
    r_target,
)


@pytest.mark.parametrize("factor", [2, 4, 8])
@pytest.mark.parametrize("shape", [(2, 3, 5, 8, 12), (4, 6)])
def test_DU_is_exact_identity(factor, shape):
    """D @ U = I to machine precision, for any leading batch shape."""
    torch.manual_seed(0)
    c = torch.randn(*shape)
    upsampled = block_replicate_upsample(c, factor)
    assert upsampled.shape[-2:] == (shape[-2] * factor, shape[-1] * factor)
    recovered = conservative_downsample(upsampled, factor)
    assert torch.allclose(
        recovered, c, atol=1e-6, rtol=1e-5
    ), "D(U(c)) must equal c to machine precision"


@pytest.mark.parametrize("factor", [2, 4])
def test_conservative_downsample_requires_divisibility(factor):
    x = torch.randn(1, 1, factor + 1, factor + 1)
    with pytest.raises(ValueError, match="divisible"):
        conservative_downsample(x, factor)


def test_conservative_downsample_is_box_average():
    factor = 2
    x = torch.tensor(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    d = conservative_downsample(x, factor)
    expected = torch.tensor(
        [
            [(1 + 3 + 2 + 4) / 4, (5 + 7 + 6 + 8) / 4],
            [(9 + 9 + 1 + 1) / 4, (9 + 9 + 1 + 1) / 4],
        ]
    )
    assert torch.allclose(d, expected)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_null_space_projector_D_is_exactly_zero(factor):
    """D(Pi(x)) == 0 to machine precision -- Pi projects onto null(D)."""
    torch.manual_seed(1)
    x = torch.randn(3, 2, 4, 8 * factor, 8 * factor)
    projected = null_space_projector(x, factor)
    downsampled = conservative_downsample(projected, factor)
    assert torch.allclose(downsampled, torch.zeros_like(downsampled), atol=1e-6)


@pytest.mark.parametrize("factor", [2, 4])
def test_null_space_projector_is_idempotent(factor):
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4 * factor, 4 * factor)
    once = null_space_projector(x, factor)
    twice = null_space_projector(once, factor)
    assert torch.allclose(once, twice, atol=1e-6)


def test_null_space_projector_removes_coarse_representable_signal():
    """A field that's already block-constant (i.e. in the row space of U, the
    orthogonal complement of null(D) under this inner product) projects to
    exactly zero."""
    factor = 4
    coarse = torch.randn(2, 3, 5, 6)
    block_constant = block_replicate_upsample(coarse, factor)
    projected = null_space_projector(block_constant, factor)
    assert torch.allclose(projected, torch.zeros_like(projected), atol=1e-6)


def test_coarse_temporal_interp_pins_endpoints():
    torch.manual_seed(3)
    clip = torch.randn(2, 3, 5, 4, 4)
    interp = coarse_temporal_interp(clip)
    assert torch.equal(interp[:, :, 0], clip[:, :, 0])
    assert torch.equal(interp[:, :, -1], clip[:, :, -1])


def test_coarse_temporal_interp_respects_custom_tau():
    clip = torch.zeros(1, 1, 3, 1, 1)
    clip[:, :, 0] = 0.0
    clip[:, :, -1] = 10.0
    tau = torch.tensor([0.0, 0.25, 1.0])
    interp = coarse_temporal_interp(clip, tau)
    assert torch.allclose(interp[0, 0, 1, 0, 0], torch.tensor(2.5))


def test_r_target_vanishes_at_endpoints():
    torch.manual_seed(4)
    coarse_clip = torch.randn(2, 3, 5, 4, 4)
    r, interp = r_target(coarse_clip)
    assert torch.allclose(r[:, :, 0], torch.zeros_like(r[:, :, 0]), atol=1e-6)
    assert torch.allclose(r[:, :, -1], torch.zeros_like(r[:, :, -1]), atol=1e-6)
    assert torch.allclose(interp + r, coarse_clip, atol=1e-6)


def test_d_target_present_at_endpoints():
    """Prop 1: unlike r, d is NOT pinned -- it's generically nonzero at the
    endpoint frames too."""
    torch.manual_seed(5)
    factor = 4
    coarse_clip = torch.randn(2, 3, 5, 4, 4)
    fine_clip = torch.randn(2, 3, 5, 16, 16)
    d = d_target(fine_clip, coarse_clip, factor)
    assert not torch.allclose(d[:, :, 0], torch.zeros_like(d[:, :, 0]))
    assert not torch.allclose(d[:, :, -1], torch.zeros_like(d[:, :, -1]))
    # D(d) == 0 exactly at every frame, endpoints included.
    assert torch.allclose(
        conservative_downsample(d, factor), torch.zeros_like(coarse_clip), atol=1e-6
    )


def test_d_target_independent_of_coarse_clip():
    """d_target = Pi(fine_clip - U(coarse_clip)) algebraically reduces to
    Pi(fine_clip): Pi(U(x_c)) == 0 for ANY x_c, so two completely different
    coarse clips must give the identical d target."""
    torch.manual_seed(5)
    factor = 4
    fine_clip = torch.randn(2, 3, 5, 16, 16)
    coarse_clip_a = torch.randn(2, 3, 5, 4, 4)
    coarse_clip_b = torch.randn(2, 3, 5, 4, 4) * 100.0
    d_a = d_target(fine_clip, coarse_clip_a, factor)
    d_b = d_target(fine_clip, coarse_clip_b, factor)
    assert torch.allclose(d_a, d_b, atol=1e-4)
    assert torch.allclose(d_a, null_space_projector(fine_clip, factor), atol=1e-6)


def test_assemble_fine_output_exact_coarse_recovery():
    """Prop 4: D(x_hat_f(tau)) == I(tau) + r_hat(tau) exactly, at every tau,
    even when d_hat is a raw (non-null(D)) prediction -- assemble_fine_output
    re-projects it."""
    torch.manual_seed(6)
    factor = 4
    coarse_clip = torch.randn(2, 3, 5, 4, 4)
    r, interp = r_target(coarse_clip)
    d_hat_raw = torch.randn(2, 3, 5, 16, 16)  # NOT projected into null(D)

    x_hat = assemble_fine_output(interp, r, d_hat_raw, factor)
    recovered = conservative_downsample(x_hat, factor)
    assert torch.allclose(recovered, interp + r, atol=1e-6)

    # And at the true endpoints specifically: D(x_hat(0)) == x_0_c exactly.
    assert torch.allclose(recovered[:, :, 0], coarse_clip[:, :, 0], atol=1e-6)
    assert torch.allclose(recovered[:, :, -1], coarse_clip[:, :, -1], atol=1e-6)


def test_assemble_fine_output_reconstructs_truth_from_true_targets():
    """Sanity check tying r_target/d_target/assemble_fine_output together:
    assembling the TRUE targets reproduces the true fine clip exactly --
    PROVIDED ``coarse_clip`` really is the conservative downsample of
    ``fine_clip`` (``coarse_clip = D(fine_clip)``), which this test
    constructs explicitly.

    This does NOT hold for arbitrary independently-sourced coarse/fine
    pairs (e.g. this codebase's actual 100km/25km stores, which are
    separate datasets, not a literal ``D``-downsample of one another) --
    in that case ``d_target`` still equals ``Pi(fine_clip)`` exactly (``Pi``
    kills the ``U(coarse_clip)`` term identically, see the derivation in
    ``d_target``'s docstring), but ``U(coarse_clip) != U(D(fine_clip))`` in
    general, so assembling the true targets recovers ``fine_clip`` only up
    to that coarse/fine data mismatch. Prop 4's exactness guarantee is about
    the MODEL's own self-consistency (``D(x_hat_f) == I + r_hat``,
    see ``test_assemble_fine_output_exact_coarse_recovery``), not about
    recovering ground truth from independently-collected coarse data.
    """
    torch.manual_seed(7)
    factor = 4
    fine_clip = torch.randn(2, 3, 5, 16, 16)
    coarse_clip = conservative_downsample(fine_clip, factor)
    r, interp = r_target(coarse_clip)
    d = d_target(fine_clip, coarse_clip, factor)
    reconstructed = assemble_fine_output(interp, r, d, factor)
    assert torch.allclose(reconstructed, fine_clip, atol=1e-6)
