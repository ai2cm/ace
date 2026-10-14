# Share one `XarrayDataset` across concat members that differ only by `subset`

`get_xarray_datasets` builds one `XarrayDataset` per entry of a concat list. Entries
that name the same store and differ only in `XarrayDataConfig.subset` therefore each
hold their own copy of the store's time-invariant tensors, mask tensors and coordinate
tensors. Under the `forkserver` start method every such CPU tensor costs one file
descriptor per DataLoader worker launch, so the launch cost of a config scales with the
number of time segments a store is cut into rather than with the number of stores.

This change builds one `XarrayDataset` per distinct config-ignoring-`subset` within a
concat list and wraps it in one `XarraySubset` view per entry. No public API changes.

---

## `fme/core/dataset/xarray.py` (modified)

```python
class XarrayDataset(DatasetABC):
    def __init__(
        self,
        config: XarrayDataConfig,
        names: Sequence[str],
        n_timesteps: IntSchedule,
        allow_missing_variables: bool = False,
    ):  # CHANGED — stops recording config.subset
        ...

    # self.subset_config = config.subset  # REMOVED — written in __init__, read nowhere
```

```python
class XarraySubset(DatasetABC):
    @property
    def wrapped_dataset(self) -> XarrayDataset:  # NEW — the underlying dataset this view reads
        return self._wrapped_dataset
```

```python
def _equal_ignoring_subset(a: XarrayDataConfig, b: XarrayDataConfig) -> bool:  # NEW
    """Whether two configs select the same data apart from which samples they keep."""
    return all(
        getattr(a, field.name) == getattr(b, field.name)
        for field in dataclasses.fields(XarrayDataConfig)
        if field.name != "subset"
    )
```

```python
def get_xarray_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    names: Sequence[str],
    n_timesteps: IntSchedule,
    strict: bool = True,
    allow_missing_variables: bool = False,
) -> tuple[list[XarraySubset], DatasetProperties]:  # CHANGED — signature unchanged, body deduplicates
    # constructs XarrayDataset directly instead of calling get_xarray_dataset, so that
    # one dataset can back several XarraySubset views
    ...
```

```python
def get_xarray_dataset(  # unchanged — still the single-config path used by
    ...                  # XarrayDataConfig.build and by forcing-data callers
) -> tuple["XarraySubset", DatasetProperties]:
    ...
```

### Critical detail — the dedup key

Two entries of one concat list share an `XarrayDataset` **iff** their
`XarrayDataConfig` instances compare equal on every dataclass field except `subset`.
Field comparison is plain `==`, which is what `XarrayDataConfig.__eq__` already uses;
the exclusion is by field name, so a field added to the config later joins the key
automatically.

`XarrayDataConfig` is not hashable — it is an `eq=True` dataclass, and `isel` is a
`Mapping` — so the key cannot be a dict key. The lookup is a linear scan over the
configs already built in this call. Concat lists are short and each miss opens files,
so the scan is not the cost.

`names`, `n_timesteps` and `allow_missing_variables` are **not** part of the key.
`get_xarray_datasets` receives exactly one of each and passes them unchanged to every
member, so within a single call they cannot distinguish two entries. This is why the
deduplication is local to that function's loop and gets no reusable surface: callers
that build one config repeatedly under *different* names or horizons exist — the
coupled training path builds one atmosphere config twice, for the target horizon and
the forcing horizon — and a config-only key would be wrong for them.

Removing `XarrayDataset.subset_config` is what makes the exclusion of `subset` from the
key structural rather than documentary: after the removal, an `XarrayDataset` carries
no record of the subset it was built from, so there is nothing for a shared instance to
report inconsistently.

### Critical detail — consequences of sharing

- **Mask provider identity.** `XarrayDataset.properties` hands out
  `self._spatial_mask_provider` by reference, and `DatasetProperties.update_merged_dataset`
  mutates it in place through `SpatialMaskProvider.update`. Members that share a dataset
  now share that provider. Within `get_xarray_datasets` nothing mutates it, and merging
  happens after the concat is built, so this changes nothing for members that read the
  same store. Two concat members whose surrounding merge lists differ in more than
  `subset` are the case the key does not constrain, and the key is not widened to cover
  it.
- **Epochs.** Each `XarraySubset.set_epoch` forwards to the wrapped dataset, which fills
  one epoch tensor; several views over one dataset are idempotent, not additive. The same
  holds for `enable_shared_memory` and `set_global_epoch_tensor`.
- **Logging.** `_get_spatial_mask_provider` logs one mask inventory per `XarrayDataset`
  it builds, so a job log now carries one line per distinct store-and-config rather than
  one per concat member.

### Commits

1. Remove the unread `XarrayDataset.subset_config` attribute.
2. Deduplicate concat members in `get_xarray_datasets`, with the identity tests.

---

## Tests

## `fme/core/dataset/test_xarray.py` (modified)

```python
# Build on the existing mock_monthly_netcdfs fixture and the IntSchedule construction
# used by test_xarray_subset_has_correct_sample.

def test_get_xarray_datasets_shares_dataset_across_subsets():
    # GOAL: two configs on one path differing only by subset yield two XarraySubset
    # views whose wrapped_dataset is the same object.
    # PARAMETERIZE: subset type ∈ {Slice, TimeSlice, RepeatedInterval}.
    ...

def test_get_xarray_datasets_shared_dataset_preserves_per_member_subsets():
    # GOAL: sharing does not merge the views — each member's sample_start_times and
    # length match what it gets when built alone through get_xarray_dataset.
    ...

def test_get_xarray_datasets_distinct_configs_are_not_shared():
    # GOAL: configs differing in a field other than subset get distinct datasets.
    # PARAMETERIZE: differing field ∈ {data_path, n_repeats, dtype, isel, labels}.
    ...
```

---

## Open Questions

- `XarraySubset.wrapped_dataset` is added so the identity assertion does not reach into
  a private attribute. Reaching into `_wrapped_dataset` from the test instead would keep
  the surface unchanged — preference?
- The deduplication is a loop-local scan inside `get_xarray_datasets`. Shaping it as a
  small private builder object that takes `names`/`n_timesteps`/`allow_missing_variables`
  once would make the key's preconditions enforceable for a future second caller, at the
  cost of a type nothing else needs today.
