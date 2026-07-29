"""The per-pairing-group data loader.

One :class:`TranslateDataLoader` serves one pairing group: it batches that
group's :class:`~fme.translate.data.dataset.PairedStreamDataset` into a
:class:`~fme.translate.data.batch_data.TranslateBatchData` over the group's
streams. Everything it does beyond typing —
``subset``/``set_epoch``/``alternate_shuffle``/``batch_size``/``n_samples``/
``log_info`` — comes from :class:`fme.core.generics.dataloader.GenericDataLoader`,
exactly as :class:`fme.coupled.data_loading.dataloader.CoupledDataLoader` does.
It must not add required constructor arguments, because ``GenericDataLoader.subset``
re-instantiates it with only ``dataset``, ``sampler`` and ``collate_fn``.
"""

from fme.core.generics.dataloader import GenericDataLoader
from fme.translate.data.batch_data import TranslateBatchData

__all__ = ["TranslateDataLoader"]


class TranslateDataLoader(GenericDataLoader[TranslateBatchData]):
    pass
