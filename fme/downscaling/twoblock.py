"""Two-block (r, d) decomposition for coarse-endpoint spatiotemporal diffusion.

See ``idea/spatiotemoral/twoblock_theory.md`` (repo root, sibling of ``ace/``)
for the full derivation and proofs. Summary: the single-stage residual
``x_f(tau) - upsample(coarse_interp(tau))`` is not spatially/temporally
separable (Prop 2 of that doc) because it mixes two structurally different
components:

- ``r``: a **pinned** coarse-temporal residual, exactly zero at the two
  endpoint frames by construction, living on the coarse spatial grid.
- ``d``: an **unpinned** fine-detail residual, present (nonzero) at every
  frame including the endpoints, living on the fine spatial grid.

This module implements the exact linear algebra needed to keep the two
blocks separate and to recombine them into a fine-resolution field with an
EXACT (not loss-penalized) coarse/fine consistency guarantee:

    D(x_hat_f(tau)) == I(tau) + r_hat(tau)   for every tau, exactly
    D(x_hat_f(0))    == x_0_c                 exactly (same at tau=T)

where ``D`` is a fixed conservative (area-weighted) downsample and ``U`` is
its exact left inverse (``D @ U = I``), so ``Pi = I - U @ D`` is the
projector onto ``null(D)`` that the fine-detail block ``d`` is parameterized
in. All three functions are pure tensor ops with no learned parameters and
no dependency on ``video_models``, so they can be unit-tested in isolation
(``test_twoblock.py``) and reused by both training (target construction) and
inference (output assembly).
"""

import torch


def conservative_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Conservative (area-weighted box-average) downsample by ``factor``
    along the last two (spatial) dims of ``x``. This is ``D``: the exact
    left inverse of ``block_replicate_upsample`` (``D @ U = I``, see that
    function's docstring for why the identity is exact, not approximate).

    Args:
        x: Any tensor whose last two dims are spatial and evenly divisible
            by ``factor`` (e.g. ``(B, C, T, H, W)``).
        factor: The downscale factor (fine pixels per coarse pixel, per
            side).
    """
    *lead, height, width = x.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Spatial dims {(height, width)} must be evenly divisible by "
            f"factor={factor}."
        )
    x = x.reshape(*lead, height // factor, factor, width // factor, factor)
    return x.mean(dim=(-3, -1))


def block_replicate_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Block-replicate (nearest-neighbor) upsample by ``factor`` along the
    last two (spatial) dims of ``x``. This is ``U``: the exact left inverse
    of ``conservative_downsample`` -- ``D @ U = I`` to machine precision,
    since averaging ``factor**2`` identical replicated values returns that
    value exactly (multiplying and dividing by a power of two is exact in
    IEEE-754 floating point, barring overflow/underflow -- verified in
    ``test_twoblock.py`` for the ``factor=4`` case used throughout this
    codebase's 100km/25km configs).

    Unlike ``metrics_and_maths.interpolate`` (bicubic, used for the
    conditioning/legacy baseline), this ``U`` is deliberately "unattractive"
    (piecewise-constant) -- its only job is the exact ``D @ U = I`` identity
    that makes the null-space parameterization below exact; visual fine
    detail is entirely the ``d`` block's job, not ``U``'s.

    Args:
        x: Any tensor whose last two dims are the coarse spatial dims.
        factor: The upscale factor.
    """
    return x.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


def null_space_projector(x: torch.Tensor, factor: int) -> torch.Tensor:
    """``Pi(x) = x - U(D(x))``: projects ``x`` onto ``null(D)``, removing the
    coarse-block-mean component and leaving only the fine detail beyond what
    the coarse grid can represent. ``Pi`` is idempotent (``Pi(Pi(x)) ==
    Pi(x)``) and ``D(Pi(x)) == 0`` exactly (Prop 4 of the two-block theory
    doc) -- both checked to machine precision in ``test_twoblock.py``.
    """
    return x - block_replicate_upsample(conservative_downsample(x, factor), factor)


def coarse_temporal_interp(
    coarse_clip: torch.Tensor, tau: torch.Tensor | None = None
) -> torch.Tensor:
    """``I(tau)``: temporal linear interpolation between the two coarse
    endpoint frames (index 0 and -1 along the time axis, axis -3).

    Resolution-agnostic -- this is the same operation as
    ``video_models._linear_interp_endpoints``, duplicated here (rather than
    imported) so this module has no dependency on ``video_models``; callers
    in ``video_models.py`` should prefer their own copy for the fine-grid
    baseline and use this one only for the coarse grid (or reuse either,
    since the math is identical).

    Args:
        coarse_clip: ``(..., T, H, W)``-shaped tensor.
        tau: Normalized frame times in ``[0, 1]``, shape ``(T,)``. Defaults
            to a uniform grid (``linspace(0, 1, T)``) when omitted.
    """
    n_times = coarse_clip.shape[-3]
    shape = [1] * coarse_clip.dim()
    shape[-3] = n_times
    if tau is None:
        w = torch.linspace(0.0, 1.0, n_times, device=coarse_clip.device)
    else:
        w = tau.to(device=coarse_clip.device, dtype=coarse_clip.dtype)
    w = w.reshape(shape)
    x0 = coarse_clip[..., 0:1, :, :]
    xT = coarse_clip[..., n_times - 1 : n_times, :, :]
    return (1 - w) * x0 + w * xT


def r_target(
    coarse_clip: torch.Tensor, tau: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pinned coarse-temporal-residual target ``r(tau) = x_c(tau) -
    I(tau)``, and ``I(tau)`` itself (needed again at assembly time, see
    ``assemble_fine_output``). ``r`` is exactly zero at the two endpoint
    frames by construction (``I`` interpolates exactly to ``x_c`` there).

    Args:
        coarse_clip: True coarse-resolution clip, ``(B, C, T, Hc, Wc)``.
        tau: See ``coarse_temporal_interp``.

    Returns:
        ``(r, interp)``, both ``(B, C, T, Hc, Wc)``.
    """
    interp = coarse_temporal_interp(coarse_clip, tau)
    return coarse_clip - interp, interp


def d_target(
    fine_clip: torch.Tensor, coarse_clip: torch.Tensor, factor: int
) -> torch.Tensor:
    """The unpinned fine-detail-residual target ``d(tau) =
    Pi(x_f(tau) - U(x_c(tau)))``, present (nonzero) at every frame including
    the endpoints (Prop 1 of the theory doc: the total residual is NOT
    pinned because this component isn't).

    Note: ``Pi(U(x_c)) == 0`` for ANY ``x_c`` (``U``'s image is exactly
    ``Pi``'s null space), so this simplifies to ``d(tau) = Pi(x_f(tau))``
    algebraically -- the ``- U(x_c)`` term never survives the projection.
    It's kept in the formula (rather than dropped) purely for
    interpretability (``d`` as "fine detail beyond the coarse anchor") and
    because a future cross-block-correlation extension (Remark 3.2 of the
    theory doc) may condition ``d``'s *noise* kernel on ``x_c``, at which
    point this residual framing would matter again. One consequence worth
    knowing: since ``coarse_clip`` and ``fine_clip`` in this codebase are
    independently-sourced datasets (not a literal ``D``-downsample pair,
    see ``twoblock.py``'s module docstring), ``d_target`` does not depend on
    exactly how well ``coarse_clip`` approximates ``D(fine_clip)`` -- it's
    always exactly ``Pi(fine_clip)`` regardless.

    Args:
        fine_clip: True fine-resolution clip, ``(B, C, T, Hf, Wf)``.
        coarse_clip: True coarse-resolution clip, ``(B, C, T, Hc, Wc)``,
            same ``T`` as ``fine_clip`` (both endpoints and interior).
        factor: ``Hf / Hc == Wf / Wc``, the coarse-to-fine downscale factor.
    """
    upsampled_coarse = block_replicate_upsample(coarse_clip, factor)
    return null_space_projector(fine_clip - upsampled_coarse, factor)


def assemble_fine_output(
    interp: torch.Tensor,
    r_hat: torch.Tensor,
    d_hat: torch.Tensor,
    factor: int,
) -> torch.Tensor:
    """``x_hat_f(tau) = U(I(tau) + r_hat(tau)) + Pi(d_hat(tau))``: the
    fine-resolution output assembled from the two blocks' predictions.

    ``d_hat`` is re-projected through ``null_space_projector`` here (rather
    than trusted to already lie in ``null(D)``) so that the exact-recovery
    guarantee (Prop 4: ``conservative_downsample(x_hat_f, factor) ==
    interp + r_hat``, to machine precision) holds unconditionally -- even
    during early training, before the network has learned to keep its raw
    ``d`` output in ``null(D)`` on its own. Since ``Pi`` is idempotent, this
    re-projection is a no-op whenever the network's output is already
    (approximately) in ``null(D)`` and otherwise only removes the
    coarse-representable component the ``r`` block already carries -- it
    never discards genuine fine detail.

    Args:
        interp: ``I(tau)``, coarse-resolution, ``(B, C, T, Hc, Wc)``.
        r_hat: Predicted coarse-temporal residual, same shape as ``interp``.
        d_hat: Predicted fine-detail residual, ``(B, C, T, Hf, Wf)``.
        factor: Coarse-to-fine downscale factor.

    Returns:
        Fine-resolution assembled output, ``(B, C, T, Hf, Wf)``.
    """
    return block_replicate_upsample(interp + r_hat, factor) + null_space_projector(
        d_hat, factor
    )
