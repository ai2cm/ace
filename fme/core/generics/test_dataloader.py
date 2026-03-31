from fme.core.generics.dataloader import GenericDataLoader
from fme.core.generics.test_dataset import _Dataset


def identity_collate(batch: list[int]) -> list[int]:
    return batch


def test_generic_dataloader_subset():
    data = list(range(10))
    dataset = _Dataset(data)

    # Create initial loader with batch_size=2
    # Batches: [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]
    loader = GenericDataLoader(
        dataset, collate_fn=identity_collate, batch_size=2, shuffle=False
    )

    # Subset: skip first batch, stop before 3rd batch (so take batch 1 and 2, 0-indexed)
    # Expected items: [2, 3, 4, 5]
    subset_loader = loader.subset(start_batch=1, stop_batch=3)

    assert len(subset_loader) == 2
    assert subset_loader.batch_size == 2

    batches = list(subset_loader)
    assert len(batches) == 2
    assert batches[0] == [2, 3]
    assert batches[1] == [4, 5]


def test_generic_dataloader_subset_twice():
    data = list(range(12))
    dataset = _Dataset(data)

    # Loader batch_size=2
    # Batches: [0,1], [2,3], [4,5], [6,7], [8,9], [10,11]
    loader = GenericDataLoader(
        dataset, collate_fn=identity_collate, batch_size=2, shuffle=False
    )

    # First subset: start=1, stop=5
    # Skips [0,1]. Takes [2,3], [4,5], [6,7], [8,9]. Stops before [10,11]
    # Items: 2, 3, 4, 5, 6, 7, 8, 9
    # Length: 4 batches
    subset1 = loader.subset(start_batch=1, stop_batch=5)
    assert len(subset1) == 4

    # Second subset: start=1, stop=3 relative to subset1
    # subset1 batches: [2,3], [4,5], [6,7], [8,9]
    # skip 1 -> skip [2,3]
    # take until 3 -> take [4,5], [6,7]
    # Expected items: 4, 5, 6, 7
    subset2 = subset1.subset(start_batch=1, stop_batch=3)

    assert len(subset2) == 2

    batches = list(subset2)
    assert len(batches) == 2
    assert batches[0] == [4, 5]
    assert batches[1] == [6, 7]
