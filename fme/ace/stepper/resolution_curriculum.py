import dataclasses

import torch

from fme.core.spherical_lowpass import SphericalLowPass
from fme.core.typing_ import TensorDict, TensorMapping


@dataclasses.dataclass
class TargetResolutionCurriculumConfig:
    """Spectral-truncation curriculum applied to the training loss target.

    Low-passes the loss target (via a spherical-harmonic round-trip) with a
    cutoff degree that increases linearly over epochs, so the model is asked to
    reproduce progressively higher-wavenumber structure as training proceeds.
    This gradually conditions an under-conditioned high-wavenumber spectral
    response instead of shocking it, which is the failure mode when fine-tuning
    a checkpoint pretrained on lower-effective-resolution data onto sharper data.

    Only the loss *target* is filtered; the model input is left at full
    resolution (filtering the input would zero the gradient to the high-degree
    spectral weights and freeze them). Filtering is applied during training
    only, and vanishes once ``ramp_epochs`` is reached, so the converged model
    is a full-resolution emulator.

    Parameters:
        start_fraction: Initial cutoff as a fraction of the grid's maximum
            representable spherical-harmonic degree, applied at epoch 0.
        ramp_epochs: Number of epochs over which the cutoff ramps linearly to
            the full resolution. At and after this epoch no filtering is applied.
    """

    start_fraction: float = 0.3
    ramp_epochs: int = 10

    def __post_init__(self):
        if not 0.0 < self.start_fraction <= 1.0:
            raise ValueError(
                f"start_fraction must be in (0, 1], got {self.start_fraction}"
            )
        if self.ramp_epochs < 1:
            raise ValueError(f"ramp_epochs must be >= 1, got {self.ramp_epochs}")

    def cutoff_fraction(self, epoch: int) -> float | None:
        """Fraction of the maximum degree to retain at ``epoch``.

        Returns ``None`` once the ramp is complete (no filtering).
        """
        if epoch >= self.ramp_epochs:
            return None
        frac = self.start_fraction + (1.0 - self.start_fraction) * (
            epoch / self.ramp_epochs
        )
        return min(frac, 1.0)


class ResolutionCurriculum:
    """Runtime that applies a ``TargetResolutionCurriculumConfig`` to targets.

    Holds the spherical low-pass and the current epoch's cutoff. Filtering is a
    no-op outside training, once the ramp is complete, or for any variable whose
    horizontal shape does not match the grid (e.g. scalars).
    """

    def __init__(
        self,
        config: TargetResolutionCurriculumConfig,
        nlat: int,
        nlon: int,
        grid: str,
    ):
        self._config = config
        self._nlat = nlat
        self._nlon = nlon
        self._lowpass = SphericalLowPass(nlat, nlon, grid)
        self._max_degree = self._lowpass.max_degree
        self._cutoff: int | None = None
        self._epoch: int | None = None
        self._is_training = True

    def init_for_epoch(self, epoch: int | None) -> None:
        if epoch == self._epoch:
            return
        if epoch is None:
            self._cutoff = None
        else:
            frac = self._config.cutoff_fraction(epoch)
            self._cutoff = (
                None if frac is None else max(1, round(frac * self._max_degree))
            )
        self._epoch = epoch

    def set_train(self) -> None:
        self._is_training = True

    def set_eval(self) -> None:
        self._is_training = False

    @property
    def active(self) -> bool:
        return (
            self._is_training
            and self._cutoff is not None
            and self._cutoff < self._max_degree
        )

    def filter_target(self, target: TensorMapping) -> TensorDict:
        """Low-pass each gridded variable in ``target`` at the current cutoff.

        Variables whose last two dims do not match the grid are passed through.
        The target carries no gradient, so the round-trip is done under no_grad.
        """
        if not self.active:
            return dict(target)
        assert self._cutoff is not None
        out: TensorDict = {}
        for name, value in target.items():
            if tuple(value.shape[-2:]) == (self._nlat, self._nlon):
                with torch.no_grad():
                    out[name] = self._lowpass(value, self._cutoff)
            else:
                out[name] = value
        return out
