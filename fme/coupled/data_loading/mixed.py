"""Mixed training data for coupled fine-tuning with atmosphere-only replay
batches (prototype, Troy 2026-10-20).

Interleaves the coupled training loader (``CoupledBatchData``) with an uncoupled
atmosphere loader (``BatchData``, e.g. the random-CO2 ensemble that has no ocean)
so that a configured share of each epoch's batches is atmosphere-only. The kind
sequence of an epoch is drawn from a RandomState seeded by the distributed seed
and the epoch, so every data-parallel rank sees the same sequence and a resumed
segment can replay it to skip the right number of batches of each kind.
"""

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.gridded_data import GriddedData as AtmosphereGriddedData
from fme.core.distributed import Distributed
from fme.core.generics.data import GriddedDataABC
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.gridded_data import GriddedData as CoupledGriddedData

_KIND_RNG_OFFSET = (
    686  # distinct from the n_steps (684) and component-choice (685) RNGs
)


class _MixedLoader:
    """One epoch's interleaving of the two loaders, following ``kinds``
    (True = atmosphere-only batch, False = coupled batch).
    """

    def __init__(self, coupled_loader, atmosphere_loader, kinds: np.ndarray):
        self._coupled = coupled_loader
        self._atmosphere = atmosphere_loader
        self._kinds = kinds

    def __len__(self) -> int:
        return len(self._kinds)

    def __iter__(self) -> Iterator[CoupledBatchData | BatchData]:
        coupled = iter(self._coupled)
        atmosphere = iter(self._atmosphere)
        for is_atmosphere in self._kinds:
            yield next(atmosphere) if is_atmosphere else next(coupled)


class MixedGriddedData(GriddedDataABC[Any]):
    def __init__(
        self,
        coupled: CoupledGriddedData,
        atmosphere_only: AtmosphereGriddedData,
        fraction: float,
    ):
        """
        Args:
            coupled: The coupled training data.
            atmosphere_only: The uncoupled atmosphere training data used for the
                replay batches.
            fraction: Share of each epoch's batches that are atmosphere-only.
        """
        self._coupled = coupled
        self._atmosphere = atmosphere_only
        self._fraction = fraction
        self._epoch = 0
        n_coupled = coupled.n_batches
        n_atmos = int(round(n_coupled * fraction / (1.0 - fraction)))
        if n_atmos > atmosphere_only.n_batches:
            logging.warning(
                f"atmosphere-only loader has {atmosphere_only.n_batches} batches, "
                f"fewer than the {n_atmos} the replay fraction {fraction} asks for "
                "per epoch; using all of them"
            )
            n_atmos = atmosphere_only.n_batches
        self._n_coupled = n_coupled
        self._n_atmos = n_atmos

    # -- the epoch's kind sequence -------------------------------------------------
    def _kinds(self) -> np.ndarray:
        """True where the batch is atmosphere-only. Same on every rank for a given
        (seed, epoch); the first batch is always coupled so the trainer's
        first-batch logging sees a coupled batch.
        """
        rng = np.random.RandomState(
            Distributed.get_instance().get_seed()
            + _KIND_RNG_OFFSET
            + 1000003 * self._epoch
        )
        kinds = np.array([False] * self._n_coupled + [True] * self._n_atmos)
        rng.shuffle(kinds)
        if self._n_atmos > 0 and kinds[0]:
            first_coupled = int(np.argmin(kinds))
            kinds[0], kinds[first_coupled] = False, True
        return kinds

    def _loader_for(self, kinds: np.ndarray, start: int | None, stop: int | None):
        kinds = kinds[start:stop]
        skipped = self._kinds()[:start] if start else np.zeros(0, dtype=bool)
        n_atmos_skip = int(skipped.sum())
        n_coupled_skip = int(len(skipped) - n_atmos_skip)
        n_atmos = int(kinds.sum())
        n_coupled = int(len(kinds) - n_atmos)
        coupled_loader = self._coupled.subset_loader(
            start_batch=n_coupled_skip, stop_batch=n_coupled_skip + n_coupled
        )
        atmosphere_loader = self._atmosphere.subset_loader(
            start_batch=n_atmos_skip, stop_batch=n_atmos_skip + n_atmos
        )
        return _MixedLoader(coupled_loader, atmosphere_loader, kinds)

    # -- GriddedDataABC ----------------------------------------------------------------
    @property
    def loader(self):
        return self._loader_for(self._kinds(), None, None)

    def subset_loader(
        self, start_batch: int | None = None, stop_batch: int | None = None
    ):
        if start_batch is None and stop_batch is not None:
            # The trainer's post-epoch evaluation pass (``subset_loader(stop_batch=
            # train_evaluation_batches)`` after ``alternate_shuffle``) feeds the
            # coupled train aggregator, which needs both realms: serve coupled
            # batches only. A resume passes ``start_batch`` and gets the mixed
            # sequence from there.
            return self._coupled.subset_loader(stop_batch=stop_batch)
        return self._loader_for(self._kinds(), start_batch, stop_batch)

    @property
    def n_samples(self) -> int:
        return self._coupled.n_samples + self._n_atmos * self._atmosphere.batch_size

    @property
    def n_batches(self) -> int:
        return self._n_coupled + self._n_atmos

    @property
    def batch_size(self) -> int:
        return self._coupled.batch_size

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._coupled.set_epoch(epoch)
        self._atmosphere.set_epoch(epoch)

    def alternate_shuffle(self):
        self._coupled.alternate_shuffle()
        self._atmosphere.alternate_shuffle()

    def log_info(self, name: str):
        self._coupled.log_info(f"{name} (coupled part)")
        self._atmosphere.log_info(f"{name} (atmosphere-only part)")
        logging.info(
            f"Dataset {name} mixes {self._n_coupled} coupled and {self._n_atmos} "
            f"atmosphere-only batches per epoch (replay fraction {self._fraction})."
        )

    # -- pass-throughs the coupled trainer reads from its train data ------------------
    @property
    def dataset_info(self):
        return self._coupled.dataset_info

    @property
    def variable_metadata(self):
        return self._coupled.variable_metadata

    @property
    def timestep(self):
        return self._coupled.timestep

    @property
    def coords(self):
        return self._coupled.coords

    @property
    def horizontal_coordinates(self):
        return self._coupled.horizontal_coordinates

    @property
    def vertical_coordinate(self):
        return self._coupled.vertical_coordinate
