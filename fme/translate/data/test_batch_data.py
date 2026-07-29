import pytest

from fme.ace.data_loading.batch_data import BatchData
from fme.translate.data.batch_data import TranslateBatchData, TranslateCollateFn


def _batch(epoch: int | None = 0, img_shape: tuple[int, int] = (4, 8)) -> BatchData:
    return BatchData.new_for_testing(
        names=["temp"], n_samples=2, n_timesteps=2, img_shape=img_shape, epoch=epoch
    )


def test_streams_are_accessible_by_name():
    batch = TranslateBatchData(streams={"era5": _batch(), "shield": _batch()})
    assert "era5" in batch
    assert len(batch) == 2
    assert sorted(batch) == ["era5", "shield"]
    assert batch["era5"] is batch.streams["era5"]


def test_epoch_is_the_shared_epoch():
    batch = TranslateBatchData(streams={"era5": _batch(epoch=7), "s": _batch(epoch=7)})
    assert batch.epoch == 7


def test_epoch_disagreement_raises():
    with pytest.raises(ValueError, match="same epoch"):
        TranslateBatchData(streams={"a": _batch(epoch=0), "b": _batch(epoch=1)})


def test_empty_batch_raises():
    with pytest.raises(ValueError, match="at least one stream"):
        TranslateBatchData(streams={})


def test_merge_combines_disjoint_stream_sets():
    merged = TranslateBatchData.merge(
        [
            TranslateBatchData(streams={"a": _batch(), "b": _batch()}),
            TranslateBatchData(streams={"c": _batch()}),
        ]
    )
    assert sorted(merged) == ["a", "b", "c"]


def test_merge_rejects_a_stream_in_two_groups():
    with pytest.raises(ValueError, match="sharing the streams"):
        TranslateBatchData.merge(
            [
                TranslateBatchData(streams={"a": _batch()}),
                TranslateBatchData(streams={"a": _batch()}),
            ]
        )


def test_collate_fn_requires_matching_stream_sets():
    with pytest.raises(ValueError, match="same streams"):
        TranslateCollateFn(
            horizontal_dims={"a": ["lat", "lon"]}, label_encodings={"b": None}
        )
