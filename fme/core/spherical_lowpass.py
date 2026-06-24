import torch

from fme.core.distributed import Distributed


class SphericalLowPass:
    """Low-pass filter on the sphere via a spherical-harmonic round-trip.

    Band-limits a field (whose last two dimensions are latitude and longitude)
    to spherical-harmonic degrees below a cutoff, by a forward then inverse SHT
    truncated at that cutoff. The transform operators are built lazily and
    cached per (cutoff, device). The distributed SHT factory is used, so this is
    correct under spatial (model) parallelism.

    The degree-0 (global mean) coefficient is always retained, so the global
    mean of the field is preserved.
    """

    def __init__(self, nlat: int, nlon: int, grid: str):
        self._nlat = nlat
        self._nlon = nlon
        self._grid = grid
        # triangular truncation: the largest representable degree is bounded by
        # both the number of latitudes and the number of resolvable zonal modes
        self._max_degree = min(nlat, nlon // 2 + 1)
        self._cache: dict[tuple[int, torch.device], tuple[torch.nn.Module, ...]] = {}

    @property
    def max_degree(self) -> int:
        return self._max_degree

    def _ops(
        self, lmax: int, device: torch.device
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        key = (lmax, device)
        if key not in self._cache:
            dist = Distributed.get_instance()
            sht = dist.get_sht(
                self._nlat, self._nlon, lmax=lmax, mmax=lmax, grid=self._grid
            ).to(device=device)
            isht = dist.get_isht(
                self._nlat, self._nlon, lmax=lmax, mmax=lmax, grid=self._grid
            ).to(device=device)
            self._cache[key] = (sht, isht)
        return self._cache[key]  # type: ignore[return-value]

    def __call__(self, x: torch.Tensor, cutoff_degree: int) -> torch.Tensor:
        """Return ``x`` band-limited to spherical degrees below ``cutoff_degree``.

        ``cutoff_degree`` is clamped to ``[1, max_degree]``. The last two
        dimensions of ``x`` must be ``(nlat, nlon)``.
        """
        if tuple(x.shape[-2:]) != (self._nlat, self._nlon):
            raise ValueError(
                f"SphericalLowPass expects last two dims {(self._nlat, self._nlon)}, "
                f"got {tuple(x.shape[-2:])}."
            )
        lmax = max(1, min(int(cutoff_degree), self._max_degree))
        sht, isht = self._ops(lmax, x.device)
        # SHT operators are float32; cast in and back to preserve caller dtype.
        return isht(sht(x.float())).to(x.dtype)
