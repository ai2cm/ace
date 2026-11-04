import pytest
import torch

from fme.core.corrector.registry import BatchCoinFlipCorrector, CorrectionSequence
from fme.core.distributed import Distributed


@pytest.mark.parallel
def test_batch_coin_flip_identical_across_ranks():
    dist = Distributed.get_instance()
    torch.manual_seed(dist.rank)  # distinct local draws per rank
    corrector = BatchCoinFlipCorrector(wrapped=CorrectionSequence([]))
    corrector.train(True)
    flags = []
    for _ in range(20):
        corrector.start_train_batch()
        flags.append(corrector._corrector_disabled)
    gathered = dist.gather_object(flags)
    if dist.is_root():
        assert gathered is not None
        assert all(rank_flags == flags for rank_flags in gathered)
        assert set(flags) == {True, False}
