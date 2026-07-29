import dataclasses
import math

import pytest
import torch

from fme.core.disco._disco_utils import _disco_s2_contraction_fft
from fme.core.distributed import Distributed
from fme.core.models.conditional_sfno.local_filter import (
    LocalFilter,
    LocalFilterConfig,
    zonal_quarter_cycle_shift,
)
from fme.core.models.conditional_sfno.s2convolutions import SpectralConvS2

# A grid small enough that the DISCO convolution tensor precompute (a Python
# loop over kernel_size x nlat) stays fast, but large enough for the l-transfer
# properties under test to be meaningful.
NLAT, NLON = 16, 32
LMAX, MMAX = NLAT, NLON // 2 + 1
GRID = "legendre-gauss"

# The radial family the parity experiment uses: piecewise-linear radial bumps
# with one azimuthal bin, sized to give exactly lmax basis functions.
PARITY_FILTER = LocalFilterConfig(
    kernel_shape="lmax",
    basis_type="piecewise linear",
    theta_cutoff="global",
    two_branch=True,
    match_spectral_init=True,
)


def _transforms():
    dist = Distributed.get_instance()
    return (
        dist.get_sht(NLAT, NLON, lmax=LMAX, mmax=MMAX, grid=GRID),
        dist.get_isht(NLAT, NLON, lmax=LMAX, mmax=MMAX, grid=GRID),
    )


def _build(config: LocalFilterConfig, embed_dim: int = 1) -> LocalFilter:
    return config.build(
        embed_dim,
        in_shape=(NLAT, NLON),
        out_shape=(NLAT, NLON),
        grid_in=GRID,
        grid_out=GRID,
        lmax=LMAX,
    )


def _basis_outputs(conv, fields: torch.Tensor) -> torch.Tensor:
    """Output of each basis function alone: (N, kernel_size, nlat, nlon)."""
    with torch.no_grad():
        return _disco_s2_contraction_fft(
            fields, conv.psi_fft_conj, conv.psi_gather_idx, conv.nlon_out
        ).squeeze(1)


def _transfer_matrix(conv, sht, isht) -> torch.Tensor:
    """The (lmax, kernel_size) Legendre transfer matrix of an isotropic basis.

    Entry ``[l, k]`` is the real factor by which basis function ``k`` scales the
    coefficient of ``Y_{l,0}``. An isotropic convolution acts as one real number
    per total wavenumber, so this matrix is what a learned weight combines to
    realize a per-``l`` profile.
    """
    coeffs = torch.zeros(LMAX, 1, LMAX, MMAX, dtype=torch.complex64)
    for ell in range(LMAX):
        coeffs[ell, 0, ell, 0] = 1.0
    fields = isht(coeffs).squeeze(1).unsqueeze(1)
    per_basis = _basis_outputs(conv, fields)
    with torch.no_grad():
        out = sht(per_basis.reshape(-1, NLAT, NLON).unsqueeze(1)).squeeze(1)
    out = out.reshape(LMAX, conv.kernel_size, LMAX, MMAX)
    return torch.stack(
        [out[ell, :, ell, 0].real for ell in range(LMAX)], dim=0
    ).double()


@pytest.mark.parametrize("wavenumber", [1, 2, 5])
def test_zonal_quarter_cycle_shift_advances_phase(wavenumber):
    """cos(m phi) becomes sin(m phi): a quarter cycle later in longitude."""
    lon = torch.arange(NLON) * 2 * math.pi / NLON
    got = zonal_quarter_cycle_shift(torch.cos(wavenumber * lon))
    torch.testing.assert_close(got, torch.sin(wavenumber * lon), atol=1e-6, rtol=1e-5)


def test_zonal_quarter_cycle_shift_annihilates_unshiftable_wavenumbers():
    """The zonal mean and the Nyquist wavenumber have no quadrature partner.

    The spectral filter behaves the same way: its m = 0 bin only ever sees the
    real part of the per-l weight, and an imaginary Nyquist coefficient is a
    null direction of the inverse real FFT.
    """
    lon = torch.arange(NLON) * 2 * math.pi / NLON
    zonal_mean = torch.ones(NLON)
    nyquist = torch.cos((NLON // 2) * lon)
    for field in (zonal_mean, nyquist):
        got = zonal_quarter_cycle_shift(field)
        torch.testing.assert_close(got, torch.zeros(NLON), atol=1e-6, rtol=0)


def test_zonal_quarter_cycle_shift_anticommutes_with_longitude_reflection():
    """H distinguishes eastward from westward, so reflecting longitude flips it.

    This is the mirror-symmetry breaking that no isotropic kernel can supply,
    and is why the second branch is needed at all.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 3, NLAT, NLON)
    # Reflect phi -> -phi on the periodic grid: index 0 is fixed, the rest flip.
    reflect = [0] + list(range(NLON - 1, 0, -1))
    reflected = x[..., reflect]
    torch.testing.assert_close(
        zonal_quarter_cycle_shift(reflected),
        -zonal_quarter_cycle_shift(x)[..., reflect],
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.medium_duration
def test_isotropic_branch_commutes_with_longitude_reflection():
    """An isotropic DISCO branch is mirror-symmetric, unlike the H branch.

    Together with the anticommuting test above, this is the symmetry half of the
    acceptance criteria: the a branch carries the isotropic part of the spectral
    filter and the H branch carries the chiral part, with no leakage between.
    """
    torch.manual_seed(0)
    single_branch = LocalFilterConfig(
        kernel_shape="lmax",
        basis_type="piecewise linear",
        theta_cutoff="global",
    )
    filter_ = _build(single_branch, embed_dim=2)
    x = torch.randn(1, 2, NLAT, NLON)
    reflect = [0] + list(range(NLON - 1, 0, -1))
    with torch.no_grad():
        from_reflected, _ = filter_(x[..., reflect])
        reflected_out = filter_(x)[0][..., reflect]
    # Longitude reflection is not an exact symmetry of the DISCO quadrature, so
    # this is a "much smaller than the signal" check rather than exact equality.
    error = (from_reflected - reflected_out).norm() / reflected_out.norm()
    assert error < 1e-2, f"isotropic branch is not mirror-symmetric: {error}"


@pytest.mark.medium_duration
def test_parity_basis_spans_arbitrary_per_l_profiles():
    """The radial basis must realize any real per-l profile with modest weights.

    This is the span-equality condition for reproducing the dhconv spectral
    filter, and the cheap kill-switch for the parity experiment: a family whose
    l-transfer matrix is rank-deficient or ill-conditioned at this size cannot
    express what the spectral filter expresses, so a measured difference in
    training would not be attributable to the receptive field.

    Sizing the basis to lmax is necessary but not sufficient -- an ill-conditioned
    family nominally spans the profiles while needing enormous cancelling
    weights, which training cannot find or hold. Hence the condition-number and
    solution-norm bounds, not just the residual.
    """
    sht, isht = _transforms()
    conv = _build(PARITY_FILTER).branches[0]
    transfer = _transfer_matrix(conv, sht, isht)
    assert transfer.shape == (LMAX, LMAX)

    singular_values = torch.linalg.svdvals(transfer)
    condition_number = float(singular_values[0] / singular_values[-1])
    assert condition_number < 1e4, (
        f"l-transfer matrix is ill-conditioned (cond {condition_number:.2e}); "
        "the basis cannot realize arbitrary per-l profiles in practice"
    )

    torch.manual_seed(0)
    target = torch.randn(LMAX, dtype=torch.float64)
    solution = torch.linalg.lstsq(transfer, target.unsqueeze(1)).solution
    residual = float((transfer @ solution - target.unsqueeze(1)).norm() / target.norm())
    assert residual < 1e-10, f"cannot fit a random per-l profile: residual {residual}"
    assert float(solution.norm()) < 1e3, "profile requires implausibly large weights"


def _fit_two_branch_to_dhconv(filter_, sht, isht):
    """Fit both branch weights to a random complex per-l weight.

    Returns the fitted weights and the per-l weight they target.
    """
    transfer = _transfer_matrix(filter_.branches[0], sht, isht)
    torch.manual_seed(0)
    real_part = torch.randn(LMAX, dtype=torch.float64)
    imag_part = torch.randn(LMAX, dtype=torch.float64)

    def fit(profile):
        return (
            torch.linalg.lstsq(transfer, profile.unsqueeze(1))
            .solution.squeeze(1)
            .float()
        )

    # H multiplies by -i sign(m), so the b branch carries the negated imaginary
    # profile; the design absorbs this sign convention into the learned kernel.
    return (
        fit(real_part),
        fit(-imag_part),
        torch.complex(real_part.float(), imag_part.float()),
    )


def _apply_fitted(filter_, fields, weight_a, weight_b):
    """Apply the fitted two-branch filter, and the a branch alone."""
    per_basis = _basis_outputs(filter_.branches[0], fields)
    branch_a = torch.einsum("bkxy,k->bxy", per_basis, weight_a).unsqueeze(1)
    branch_b = torch.einsum("bkxy,k->bxy", per_basis, weight_b).unsqueeze(1)
    return branch_a + zonal_quarter_cycle_shift(branch_b), branch_a


def _random_field(isht, max_l: int, max_m: int, seed: int = 1) -> torch.Tensor:
    coeffs = torch.zeros(4, 1, LMAX, MMAX, dtype=torch.complex64)
    torch.manual_seed(seed)
    for ell in range(min(max_l, LMAX)):
        n_m = min(ell + 1, max_m)
        coeffs[:, :, ell, :n_m] = torch.randn(4, 1, n_m, dtype=torch.complex64)
    return isht(coeffs)


@pytest.mark.medium_duration
def test_two_branch_reproduces_dhconv_on_resolved_scales():
    """The fitted two-branch filter approximates the dhconv spectral filter.

    The two-branch form is the exact decomposition of dhconv in the continuum,
    so this checks the implementation wires it up correctly. It is checked on the
    resolved part of the spectrum because DISCO's quadrature is only
    approximately isotropic on a lat-lon grid: the measured l-transfer drifts
    with zonal wavenumber (about 1% at m = 1, growing toward the Nyquist), so a
    fit performed at m = 0 cannot match the operator exactly at every m. That
    ceiling belongs to the DISCO discretization, not to this decomposition or to
    the choice of radial family, and it does not shrink with either.
    """
    sht, isht = _transforms()
    filter_ = _build(PARITY_FILTER)
    weight_a, weight_b, per_l_weight = _fit_two_branch_to_dhconv(filter_, sht, isht)

    fields = _random_field(isht, max_l=3 * LMAX // 4, max_m=MMAX)
    got, a_only = _apply_fitted(filter_, fields, weight_a, weight_b)
    with torch.no_grad():
        expected = isht(sht(fields) * per_l_weight.reshape(1, 1, LMAX, 1))

    error = float((got - expected).norm() / expected.norm())
    assert error < 0.15, f"two-branch filter does not reproduce dhconv: {error}"

    # Without H only the real part of the per-l weight is reproduced, which must
    # be markedly worse -- otherwise the test would pass with the chiral part
    # silently absent.
    error_without_h = float((a_only - expected).norm() / expected.norm())
    assert error_without_h > 3 * error, (
        "the H branch contributes little, so the imaginary part of the spectral "
        "filter is not being reproduced"
    )


@pytest.mark.medium_duration
def test_two_branch_leaves_the_zonal_mean_to_the_isotropic_branch():
    """On m = 0 fields the H branch must contribute exactly nothing.

    This is the structural counterpart of the reproduction test above: the
    spectral filter's m = 0 bin only ever sees the real part of its per-l weight,
    so a correct H annihilates the zonal mean rather than mixing the imaginary
    profile into it.
    """
    sht, isht = _transforms()
    filter_ = _build(PARITY_FILTER)
    weight_a, weight_b, per_l_weight = _fit_two_branch_to_dhconv(filter_, sht, isht)

    fields = _random_field(isht, max_l=LMAX, max_m=1)
    got, a_only = _apply_fitted(filter_, fields, weight_a, weight_b)
    torch.testing.assert_close(got, a_only, atol=1e-5, rtol=1e-4)

    # And what remains is the real part of the weight, to the DISCO ceiling.
    with torch.no_grad():
        expected = isht(sht(fields) * per_l_weight.real.reshape(1, 1, LMAX, 1))
    error = float((got - expected).norm() / expected.norm())
    assert error < 0.1, f"m=0 response does not match the real per-l profile: {error}"


@pytest.mark.medium_duration
def test_match_spectral_init_matches_spectral_filter_output_magnitude():
    """The DISCO filter must start at the spectral filter's output magnitude.

    DISCO's own initialization scales as 1 / sqrt(channels * kernel_size), which
    at kernel_size = lmax is far weaker than SpectralConvS2's 1 / sqrt(channels).
    Training the two arms from different effective output scales would confound
    the comparison, so the parity config rescales.
    """
    embed_dim = 16
    sht, isht = _transforms()
    torch.manual_seed(0)
    x = torch.randn(2, embed_dim, NLAT, NLON)

    spectral = SpectralConvS2(sht, isht, embed_dim, embed_dim, num_groups=1, bias=False)
    disco = _build(PARITY_FILTER, embed_dim=embed_dim)
    with torch.no_grad():
        spectral_out, _ = spectral(x)
        disco_out, _ = disco(x)

    ratio = float(disco_out.std() / spectral_out.std())
    assert 0.5 < ratio < 2.0, (
        f"DISCO output magnitude is {ratio:.3f}x the spectral filter's; the "
        "initializations are not comparable"
    )

    unscaled = _build(
        dataclasses.replace(PARITY_FILTER, match_spectral_init=False),
        embed_dim=embed_dim,
    )
    with torch.no_grad():
        unscaled_out, _ = unscaled(x)
    unscaled_ratio = float(unscaled_out.std() / spectral_out.std())
    assert (
        unscaled_ratio < 0.5 * ratio
    ), "match_spectral_init made no difference, so it is not being applied"


def test_default_config_reproduces_the_historical_local_filter():
    """The default must stay the 3x3 Morlet filter with the heuristic radius.

    local_blocks predates these options, so changing the default would silently
    redefine any existing use.
    """
    config = LocalFilterConfig()
    assert config.resolved_kernel_shape(LMAX) == (3, 3)
    assert config.basis_type == "morlet"
    assert not config.two_branch
    assert not config.match_spectral_init
    # The historical value: twice the mode-count heuristic.
    expected = 2 * (3 + 1) * 0.5 * math.pi / (NLAT - 1)
    assert config.resolved_theta_cutoff(NLAT, (3, 3)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"theta_cutoff": 4.0}, "theta_cutoff must be in"),
        ({"theta_cutoff": 0.0}, "theta_cutoff must be in"),
        ({"theta_cutoff": "everywhere"}, "theta_cutoff must be a number"),
        ({"kernel_shape": "all"}, "kernel_shape must be a list"),
    ],
)
def test_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LocalFilterConfig(**kwargs)


def test_lmax_kernel_shape_requires_a_radial_family():
    """A basis function count only determines kernel_shape for radial families."""
    config = LocalFilterConfig(kernel_shape="lmax", basis_type="morlet")
    with pytest.raises(ValueError, match="no purely radial form"):
        config.resolved_kernel_shape(LMAX)


def test_two_branch_requires_isotropic_branches():
    """The decomposition assumes real radius-only kernels.

    Plain Morlet is the trap this guards: it accepts a single azimuthal bin but
    keeps a Cartesian y-harmonic, so it is not isotropic.
    """
    config = LocalFilterConfig(
        kernel_shape=[3, 1], basis_type="morlet", two_branch=True
    )
    with pytest.raises(ValueError, match="not purely radial"):
        _build(config)
