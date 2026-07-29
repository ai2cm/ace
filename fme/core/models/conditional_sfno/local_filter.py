"""The local (DISCO) filter that stands in for the SFNO spectral filter.

``SpectralConvS2`` (``filter_type: linear``, ``operator_type: dhconv``)
multiplies each total wavenumber ``l`` by an independent complex
channel-mixing matrix, shared across zonal wavenumber ``m``. The real SHT
stores only ``m >= 0``, with ``m < 0`` implied by conjugate symmetry, so
multiplying the stored coefficients by ``w_l = a_l + i b_l`` acts over the full
``m`` range as::

    a_l + i b_l sign(m)

The real part is genuinely isotropic. The imaginary part is not: ``i sign(m)``
advances every zonal harmonic by a quarter cycle, which is tied to the rotation
axis and distinguishes eastward from westward.

By Funk-Hecke, convolving with a real radius-only kernel multiplies ``f_lm`` by
a real number identically in ``m``, so the spectral filter decomposes exactly
into two isotropic convolutions plus one fixed operator::

    dhconv  ==  K_a * f  +  H[K_b * f]

where ``K_a`` and ``K_b`` are learned real radius-only kernels supplying the
free profiles ``a_l`` and ``b_l``, and ``H`` is parameter-free multiplication by
``-i sign(m)`` (:func:`zonal_quarter_cycle_shift`). No isotropic kernel can
express ``H``: any real span of radius-only kernels is mirror-symmetric, while
``H`` is not, so quadrature has to be supplied explicitly.

``LocalFilter`` implements that decomposition with DISCO convolutions, which
makes the receptive field a continuous knob (``theta_cutoff``) where the
spectral filter has none: shrinking ``theta_cutoff`` localizes the ``K``
branches, while ``H`` stays zonally global (its spatial kernel decays like
``1 / delta_phi``) or is dropped by setting ``two_branch: false``.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from fme.core.benchmark.timer import NullTimer, Timer
from fme.core.disco import (
    BasisNormMode,
    BasisType,
    compute_cutoff_radius,
    get_filter_basis,
    kernel_shape_for_basis_count,
)
from fme.core.distributed import Distributed


def zonal_quarter_cycle_shift(x: torch.Tensor) -> torch.Tensor:
    """Advance every zonal harmonic of ``x`` by a quarter cycle.

    Multiplies the zonal Fourier coefficient at wavenumber ``m`` by
    ``-i sign(m)``, i.e. a Hilbert transform along each latitude circle. This is
    exact without any spherical harmonic transform: spherical harmonics have
    pure ``exp(i m phi)`` longitude dependence, so a per-latitude-row FFT phase
    flip *is* the harmonic-space operator.

    Two coefficients are annihilated rather than shifted:

    - ``m = 0``, which has no quadrature partner. This matches the spectral
      filter, whose ``m = 0`` bin only ever sees ``Re(w_l)``.
    - the Nyquist wavenumber when ``nlon`` is even, whose partner
      ``sin(nlon / 2 * phi)`` vanishes at every grid point and so is not
      representable. The spectral filter's imaginary part is likewise a null
      direction there.

    Args:
        x: Real tensor with longitude as the last dimension.

    Returns:
        A real tensor of the same shape and dtype as ``x``.
    """
    nlon = x.shape[-1]
    # FFTs are not supported in reduced precision, and the filter runs with
    # autocast disabled around the spectral path for the same reason.
    coeffs = torch.fft.rfft(x.float(), dim=-1)
    m = torch.arange(coeffs.shape[-1], device=x.device)
    shifted = m > 0
    if nlon % 2 == 0:
        shifted = shifted & (m < nlon // 2)
    factor = torch.zeros(coeffs.shape[-1], dtype=coeffs.dtype, device=x.device)
    factor[shifted] = -1j
    return torch.fft.irfft(coeffs * factor, n=nlon, dim=-1).to(x.dtype)


@dataclasses.dataclass
class LocalFilterConfig:
    """Configuration for the local (DISCO) filter used by ``local_blocks``.

    The defaults reproduce the filter's long-standing hardcoded behaviour: a
    single 3x3 Morlet branch with the heuristic support radius and DISCO's own
    weight initialization.

    Attributes:
        kernel_shape: Shape of the DISCO filter basis. ``"lmax"`` resolves to
            the shape giving exactly ``lmax`` radius-only basis functions, one
            degree of freedom per total wavenumber, which is the span-equality
            condition for reproducing an arbitrary per-``l`` profile. Note that
            span equality is necessary but not sufficient: a family whose
            ``l``-transfer matrix is ill-conditioned at that size cannot
            realize arbitrary profiles in practice even though it nominally
            spans them.
        basis_type: Filter basis family. ``two_branch`` requires a family that
            is purely radial at this ``kernel_shape`` (``"isotropic morlet"``,
            or ``"piecewise linear"`` with a single azimuthal bin), since the
            decomposition assumes isotropic branches.
        theta_cutoff: Support radius of the basis, in radians. ``"global"``
            resolves to ``pi``, a globally-supported filter. ``None`` keeps the
            heuristic that scales the radius with the radial mode count, which
            is the only behaviour available historically.
        basis_norm_mode: How basis function magnitudes are normalized.
        two_branch: Whether to use the ``K_a * f + H[K_b * f]`` decomposition of
            the dhconv spectral filter (see the module docstring) rather than a
            single isotropic convolution. A single isotropic branch cannot
            express the filter's zonal phase shift, so a difference measured
            against the spectral filter without this would be ambiguous.
        match_spectral_init: Whether to rescale the branch weights so the filter
            has the same output magnitude at initialization as
            ``SpectralConvS2`` does. DISCO's own initialization is far weaker at
            large ``kernel_size``, which would confound a comparison of
            training dynamics with an initialization-magnitude difference.
    """

    kernel_shape: list[int] | str = dataclasses.field(default_factory=lambda: [3, 3])
    basis_type: BasisType = "morlet"
    theta_cutoff: float | str | None = None
    basis_norm_mode: BasisNormMode = "mean"
    two_branch: bool = False
    match_spectral_init: bool = False

    def __post_init__(self):
        if isinstance(self.kernel_shape, str) and self.kernel_shape != "lmax":
            raise ValueError(
                "local filter kernel_shape must be a list of ints or the string "
                f'"lmax", got {self.kernel_shape!r}.'
            )
        if isinstance(self.theta_cutoff, str) and self.theta_cutoff != "global":
            raise ValueError(
                "local filter theta_cutoff must be a number, the string "
                f'"global", or null, got {self.theta_cutoff!r}.'
            )
        if isinstance(self.theta_cutoff, float | int) and not (
            0.0 < self.theta_cutoff <= math.pi
        ):
            raise ValueError(
                "local filter theta_cutoff must be in (0, pi], got "
                f"{self.theta_cutoff}. pi is a globally-supported filter, the "
                "largest geodesic distance on the sphere."
            )

    def resolved_kernel_shape(self, lmax: int) -> tuple[int, ...]:
        """The concrete ``kernel_shape``, resolving the ``"lmax"`` shorthand."""
        if self.kernel_shape == "lmax":
            return kernel_shape_for_basis_count(lmax, self.basis_type)
        # A str kernel_shape is rejected in __post_init__, so this is a list.
        return tuple(self.kernel_shape)  # type: ignore[arg-type]

    def resolved_theta_cutoff(self, nlat: int, kernel_shape: tuple[int, ...]) -> float:
        """The concrete support radius in radians."""
        if self.theta_cutoff == "global":
            return math.pi
        if self.theta_cutoff is None:
            # Historically the only available value, and doubled at the call
            # site; fold the factor of 2 in so the default is unchanged.
            return 2 * compute_cutoff_radius(nlat, kernel_shape, self.basis_type)
        return float(self.theta_cutoff)  # type: ignore[arg-type]

    def build(
        self,
        embed_dim: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        grid_in: str,
        grid_out: str,
        lmax: int,
    ) -> "LocalFilter":
        return LocalFilter(
            self,
            embed_dim=embed_dim,
            in_shape=in_shape,
            out_shape=out_shape,
            grid_in=grid_in,
            grid_out=grid_out,
            lmax=lmax,
        )


def _set_spectral_matched_init(conv: nn.Module, n_branches: int) -> None:
    """Re-initialize a DISCO conv to match ``SpectralConvS2``'s output magnitude.

    ``SpectralConvS2``'s ``sqrt(1 / channels)`` weight is a unit-gain
    initialization: on a spatially white unit-variance input its output has unit
    variance. DISCO's ``sqrt(1 / (channels * kernel_size))`` default is far
    weaker once ``kernel_size`` is large -- about 50x at ``kernel_size = 45``,
    ``channels = 512`` -- so leaving it would confound a comparison of training
    dynamics against the spectral filter with an initialization difference.

    For a spatially white unit-variance input, basis function ``k`` produces
    output variance ``g_k^2 = sum(psi_k^2) / nlat_out``, where ``psi`` is the
    precomputed convolution tensor. Independent weights of variance ``v`` then
    give an output variance of ``groupsize * v * sum_k g_k^2``. Setting that to
    ``1 / n_branches`` makes the branches sum to unit gain, mirroring how the
    complex weight's real and imaginary parts each carry half the variance.
    """
    required = ("psi_ker_idx", "psi_vals", "nlat_out", "kernel_size", "groupsize")
    missing = [name for name in required if not hasattr(conv, name)]
    if missing:
        raise NotImplementedError(
            "match_spectral_init needs the precomputed convolution tensor of "
            f"fme's DISCO fork, but {type(conv).__name__} is missing "
            f"{missing}. Spatial model parallelism dispatches to "
            "torch-harmonics instead, which does not expose it."
        )
    gain_sq = torch.zeros(conv.kernel_size, dtype=torch.float64)
    gain_sq.index_add_(
        0,
        conv.psi_ker_idx.detach().cpu(),
        conv.psi_vals.detach().cpu().double() ** 2,
    )
    total_gain_sq = float(gain_sq.sum()) / conv.nlat_out
    if total_gain_sq <= 0.0:
        raise ValueError(
            "DISCO basis has zero total gain, so it cannot be scaled to match "
            "the spectral filter. Check kernel_shape and theta_cutoff."
        )
    std = math.sqrt(1.0 / (n_branches * conv.groupsize * total_gain_sq))
    with torch.no_grad():
        conv.weight.normal_(0.0, std)


class LocalFilter(nn.Module):
    """DISCO local filter, optionally in the two-branch dhconv-equivalent form.

    Returns ``(filtered, residual)`` to match the protocol
    ``FourierNeuralOperatorBlock`` expects of a spectral filter, where the
    residual is the unmodified input.
    """

    def __init__(
        self,
        config: LocalFilterConfig,
        embed_dim: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        grid_in: str,
        grid_out: str,
        lmax: int,
    ):
        super().__init__()

        kernel_shape = config.resolved_kernel_shape(lmax)
        theta_cutoff = config.resolved_theta_cutoff(in_shape[0], kernel_shape)
        basis = get_filter_basis(
            kernel_shape=kernel_shape, basis_type=config.basis_type
        )

        self.two_branch = config.two_branch
        n_branches = 2 if config.two_branch else 1

        if config.two_branch:
            if not basis.is_isotropic:
                raise ValueError(
                    "the two-branch local filter requires isotropic branches, "
                    f"but basis_type {config.basis_type!r} with kernel_shape "
                    f"{list(kernel_shape)} is not purely radial. Use "
                    "'isotropic morlet', or 'piecewise linear' with a single "
                    "azimuthal bin."
                )
            Distributed.get_instance().require_no_spatial_parallelism(
                "the two-branch local filter's zonal quarter-cycle shift is an "
                "FFT over the full longitude circle"
            )

        dist = Distributed.get_instance()
        self.branches = nn.ModuleList(
            [
                dist.get_disco_conv_s2(
                    embed_dim,
                    embed_dim,
                    in_shape=in_shape,
                    out_shape=out_shape,
                    kernel_shape=kernel_shape,
                    basis_type=config.basis_type,
                    basis_norm_mode=config.basis_norm_mode,
                    groups=1,
                    grid_in=grid_in,
                    grid_out=grid_out,
                    bias=False,
                    theta_cutoff=theta_cutoff,
                )
                for _ in range(n_branches)
            ]
        )

        if config.match_spectral_init:
            for branch in self.branches:
                _set_spectral_matched_init(branch, n_branches)

    def forward(
        self, x: torch.Tensor, timer: Timer = NullTimer()
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with timer.child("disco"):
            out = self.branches[0](x)
        if self.two_branch:
            with timer.child("disco_quarter_cycle"):
                out = out + zonal_quarter_cycle_shift(self.branches[1](x))
        return out, x
