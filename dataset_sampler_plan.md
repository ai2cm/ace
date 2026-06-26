# Plan: Scheduled group-weighted dataset sampling

## Context

The training data loader concatenates many datasets via `concat`
(`configs/experiments/2026-06-26-fm/ace-train-config-4deg-AIMIP-nc-sfno-fm.yaml`,
`train_loader.dataset.concat`). Today there is **no way to control how often each
collection of datasets is sampled** — `concat` builds a
`torch.utils.data.ConcatDataset` (`fme/core/dataset/concat.py:17`) and the sampler
draws uniformly (`_get_sampler`, `fme/ace/data_loading/getters.py:50`). So each
group's effective weight is implicit, proportional to its number of valid
start-time samples.

Two needs:
1. Specify explicit per-group sampling weights (e.g. SHiELD-AMIP 30%,
   ramped random-CO2 50%, ERA5 20%).
2. Vary those weights across training **epochs** via a schedule — in particular,
   a cooldown phase that trains on 100% ERA5.

Outcome: a `group_weights` block on the train loader that assigns weights to
spans of `concat` members, with milestone-based epoch scheduling, implemented
with a weighted, epoch-aware, distributed-safe sampler.

## Key facts established during exploration

- Epoch propagation already exists: `train.py:157` calls `data.set_epoch(epoch)`
  → `GriddedData.set_epoch` (`gridded_data.py:127`) → `SlidingWindowDataLoader`/
  `TorchDataLoader.set_epoch` → `GenericDataLoader.set_epoch`
  (`fme/core/generics/dataloader.py:122`). The latter currently calls
  `set_epoch` only on `DistributedSampler`.
- The torch `DataLoader` re-calls `iter(sampler)` each epoch and **the sampler
  runs in the main process**, so a stateful sampler can change its weights per
  epoch with no worker restart, even with `persistent_workers=True`.
- `XarrayConcat._dataset` is a `torch.utils.data.ConcatDataset` (has
  `.cumulative_sizes`); members are in `_wrapped_datasets` (`concat.py:18`).
- `SubsetDataset` reindexes the dataset when `time_buffer > 0`
  (`getters.py:98-103`) — group-id arrays must be mapped through `indices`.
- Existing schedule machinery to reuse: `ValidatedMilestones` (generic over `T`),
  `IntSchedule` (`fme/core/dataset/schedule.py:18,53`). Tests in
  `fme/core/dataset/test_schedule.py`.
- Existing with-replacement / distributed precedent to mirror:
  `_get_sampler`'s `sample_with_replacement` branch (`getters.py:56-69`) — divides
  `num_samples` by `dist.total_data_parallel_ranks` and calls
  `dist.require_no_spatial_parallelism(...)`. Distributed exposes
  `data_parallel_rank`, `total_data_parallel_ranks`, `get_seed`,
  `require_no_spatial_parallelism`.
- Existing reproducibility plumbing must be preserved: `fme.core.rand.set_seed`
  stores a base seed on `Distributed`, and `Distributed.get_sampler` passes
  `dist.get_seed()` into `torch.utils.data.DistributedSampler`.
- `Trainer.train_one_epoch` calls `train_data.set_epoch(self._epochs_trained + 1)`.
  A pre-cooldown checkpoint saved at `pre_cooldown_checkpoint_epoch: 142` is
  written after 142 complete epochs; the next training epoch is passed to data as
  epoch 143.
- `Trainer.train_one_epoch` calls `train_data.alternate_shuffle()` before
  train-data evaluation, so the weighted sampler needs an equivalent hook to
  produce an independent sample order for that evaluation pass.
- `XarrayDataset` computes sample count from `IntSchedule.max_value`, so dataset
  length and concat member lengths remain stable across scheduled `n_timesteps`
  changes. Cached group ids are therefore valid for the lifetime of the loader.

## Design

### 1. Weight schedule type — `fme/core/dataset/schedule.py`

Add, mirroring `IntSchedule` and reusing `ValidatedMilestones[list[float]]`:

```python
@dataclasses.dataclass
class WeightMilestone:
    epoch: int
    value: list[float]            # one weight per group

@dataclasses.dataclass
class WeightSchedule:
    start_value: list[float]
    milestones: list[WeightMilestone]
    # __post_init__: every milestone len == len(start_value); weights >= 0;
    #   weights are finite; sum(weights) > 0 for each value.
    #   Build ValidatedMilestones.
    # get_value(epoch) -> list[float]
    # from_constant(value) classmethod
```

Validation in `__post_init__` (per AGENTS.md). Weights need NOT sum to 1 —
normalized at use. Cooldown is just a milestone whose value zeroes all groups
except ERA5.

### 2. Config — `fme/ace/data_loading/config.py`

Add a `GroupWeightsConfig` and an optional field on `DataLoaderConfig`:

```python
@dataclasses.dataclass
class GroupWeightsConfig:
    groups: list[int]             # # of consecutive concat members per group
    start_value: list[float]
    milestones: list[WeightMilestone] = field(default_factory=list)
    # __post_init__: len(groups) == len(start_value); all groups > 0;
    #   validate WeightSchedule; store self.schedule = WeightSchedule(...)

# DataLoaderConfig:
group_weights: GroupWeightsConfig | None = None
```

`DataLoaderConfig.__post_init__` cross-validation: when `group_weights` is set,
require `dataset` is a `ConcatDatasetConfig` and
`sum(group_weights.groups) == len(dataset.concat)`. Disallow combining with
`sample_with_replacement` (both are with-replacement sampler choices) — raise.

`DataLoaderConfig` is shared by training and validation, but this feature is
intended only for `train_loader`. `get_gridded_data` should raise if
`config.group_weights is not None and train is False`, so validation/inference
does not accidentally become a random with-replacement draw.

### 3. Group-id helper + expose concat boundaries

- Add a property to `XarrayConcat` (`concat.py`) exposing member sample counts or
  cumulative sizes (e.g. `member_lengths` from
  `[len(d) for d in self._wrapped_datasets]`), so callers don't reach into the
  private `_dataset`.
- Helper `build_group_ids(member_lengths, groups) -> np.ndarray` returning a
  per-sample group id (length = total concat samples). Validate `groups` are
  positive and `sum(groups) == len(member_lengths)`.
- If `time_buffer` subsetting removes every sample from a group, then any
  schedule value with positive weight for that now-empty group must raise a
  `ValueError` when the sampler computes weights for that epoch. Zero weight for
  an empty group is allowed.

### 4. Sampler — new `ScheduledWeightedSampler` (in `getters.py` or a new
`fme/ace/data_loading/sampler.py`)

```python
class ScheduledWeightedSampler(torch.utils.data.Sampler):
    # __init__(
    #   sample_group_ids, schedule, num_samples_per_rank, rank, base_seed, epoch=0
    # )
    # set_epoch(epoch): gw = schedule.get_value(epoch); compute per-group counts;
    #   per-sample weight = gw[g] / count[g]  (0 if count==0); store tensor.
    # alternate_shuffle(): change the draw seed for train-data evaluation without
    #   changing the schedule epoch.
    # __len__ -> num_samples_per_rank
    # __iter__: seed combines base_seed, epoch/draw seed, and rank;
    #   torch.multinomial(weights,
    #   num_samples_per_rank, replacement=True, generator=...) -> list[int]
```

With-replacement, per-rank seeding so data-parallel ranks draw different samples.
Include `dist.get_seed()` in the base seed so configured training seeds affect
weighted sample streams just as they affect `DistributedSampler`.

Epoch length policy: set `num_samples_per_rank =
ceil(len(dataset) / dist.total_data_parallel_ranks)`. This matches the
non-`drop_last` `DistributedSampler` convention of a roughly one-dataset-sized
epoch, avoids zero samples per rank for a non-empty dataset, and accepts the
small amount of oversampling that comes with with-replacement weighted sampling.
Do not reuse `sample_with_replacement` as an epoch-size override in this first
implementation; it remains mutually exclusive with `group_weights`.

### 5. Wiring — `get_gridded_data` (`getters.py:92-107`)

After the dataset is built and the `time_buffer` `SubsetDataset` applied:

```python
if config.group_weights is not None:
    if not train:
        raise ValueError("group_weights is only supported for training loaders")
    dist.require_no_spatial_parallelism("group_weights uses weighted sampling")
    gids = build_group_ids(concat.member_lengths, config.group_weights.groups)
    if config.time_buffer > 0:
        gids = gids[indices]               # map through SubsetDataset reindex
    sampler = ScheduledWeightedSampler(
        gids, config.group_weights.schedule,
        num_samples_per_rank=math.ceil(
            len(dataset) / dist.total_data_parallel_ranks
        ),
        rank=dist.data_parallel_rank,
        base_seed=dist.get_seed(),
    )
else:
    sampler = _get_sampler(dataset, config.sample_with_replacement, train)
```

Needs a handle to the underlying `XarrayConcat` to read `member_lengths` before
`SubsetDataset` wrapping — capture it from `config.get_dataset(...)` return
(`dataset` at `getters.py:92` is the `XarrayConcat` before subset wrap).

### 6. Propagate `set_epoch` and `alternate_shuffle` to custom samplers —
`fme/core/generics/dataloader.py:122`

Current code calls `set_epoch` only on `DistributedSampler`. Replace the concrete
`isinstance` with a `runtime_checkable` Protocol to avoid the isinstance smell
flagged in AGENTS.md and cover both sampler types with one check:

```python
@runtime_checkable
class _EpochAwareSampler(Protocol):
    def set_epoch(self, epoch: int): ...

@runtime_checkable
class _AlternateShuffleSampler(Protocol):
    def alternate_shuffle(self): ...

def set_epoch(self, epoch):
    self._dataset.set_epoch(epoch)
    if isinstance(self._sampler, _EpochAwareSampler):
        self._sampler.set_epoch(epoch)

def alternate_shuffle(self):
    if isinstance(self._sampler, _AlternateShuffleSampler):
        self._sampler.alternate_shuffle()
    elif isinstance(self._sampler, torch.utils.data.DistributedSampler):
        self._sampler.set_epoch(alternate_seed(self._sampler.epoch))
```

`DistributedSampler` already has `set_epoch`, so this is behavior-preserving for
the existing path and additionally drives `ScheduledWeightedSampler`. The
`alternate_shuffle` branch preserves existing behavior for `DistributedSampler`
and gives `ScheduledWeightedSampler` an independent draw for train-data
evaluation without advancing the group-weight schedule.

## Example config (cooldown to 100% ERA5)

```yaml
train_loader:
  group_weights:
    groups: [2, 9, 18, 2]              # AMIP=2, random-CO2=9, SOM=18, ERA5=2 members
    start_value: [0.30, 0.50, 0.00, 0.20]
    milestones:
      - epoch: 143                     # first epoch after checkpoint at epoch 142
        value: [0.0, 0.0, 0.0, 1.0]    # 100% ERA5
```

`sum(groups)` must equal `len(train_loader.dataset.concat)`.
Use `epoch: 142` only if the intended behavior is to switch during the epoch that
completes 142 epochs, before the pre-cooldown checkpoint is written. For the
existing `pre_cooldown_checkpoint_epoch: 142` workflow, use `epoch: 143`.

## Files to change

- `fme/core/dataset/schedule.py` — add `WeightMilestone`, `WeightSchedule`.
- `fme/core/dataset/concat.py` — expose `member_lengths` on `XarrayConcat`.
- `fme/ace/data_loading/config.py` — `GroupWeightsConfig`, `DataLoaderConfig.group_weights` + cross-validation.
- `fme/ace/data_loading/getters.py` (or new `sampler.py`) — `ScheduledWeightedSampler`, `build_group_ids`, wiring in `get_gridded_data`.
- `fme/core/generics/dataloader.py` — sampler protocols, broadened `set_epoch`
  and `alternate_shuffle`.

## Caveats / behavior notes

- Weighted sampling is **with replacement** → not compatible with spatial
  parallelism (same restriction as `sample_with_replacement`); enforced via
  `require_no_spatial_parallelism`.
- Weights are per-sample-drawn, not per-epoch fraction. Cooldown to 100% ERA5 is
  exact (other groups weight 0 → never drawn).
- Group fractions are achieved in expectation for non-degenerate weights, not as
  exact per-epoch counts.
- Schedule epochs are the data epochs passed by `Trainer`, currently
  `_epochs_trained + 1`.
- `subset()` for resume (`generics/dataloader.py:107`) freezes order via
  `list(self._sampler)`; weighted sampler materializes current-epoch indices —
  works correctly.

## Verification

- Unit: `fme/core/dataset/test_schedule.py` — `WeightSchedule.get_value` across
  milestones; validation errors (mismatched lengths, all-zero, negative,
  non-finite).
- Unit: new tests near `fme/ace/data_loading/test_data_loader.py` —
  `build_group_ids` maps members→groups correctly; `ScheduledWeightedSampler`
  draws in expected proportions for a fixed seed (statistical tolerance);
  `set_epoch` switches proportions at the milestone; a zeroed group is never
  drawn (cooldown); same base seed is reproducible; different base seeds change
  draws; data-parallel ranks draw different samples; `alternate_shuffle` changes
  the draw order without changing the scheduled weights; positive weight on an
  empty group raises.
- Config: `fme/ace/data_loading/test_data_loading_config.py` — `group_weights`
  round-trips; cross-validation raises when `sum(groups) != len(concat)` and when
  combined with `sample_with_replacement`; non-positive `groups` raise; using
  `group_weights` with `train=False` raises from `get_gridded_data`.
- Distributed: extend `fme/core/distributed/parallel_tests/test_get_sampler.py`
  to assert per-rank distinct draws and `require_no_spatial_parallelism` raising
  under spatial parallel. Also assert `num_samples_per_rank` follows the
  `ceil(len(dataset) / total_data_parallel_ranks)` policy.
- End-to-end smoke: short `make test_fast` run; a tiny training config with
  `group_weights` + a milestone, assert it trains an epoch across the boundary
  without error. Include a test or explicit config check that cooldown after
  `pre_cooldown_checkpoint_epoch: 142` uses schedule milestone epoch 143.

## Commit breakdown (atomic)

1. `feature/group-weight-schedule`: add `WeightSchedule`/`WeightMilestone` + tests.
2. `XarrayConcat.member_lengths` + sampler protocols in generics dataloader
   (`set_epoch` and `alternate_shuffle`, behavior-preserving) + tests.
3. `ScheduledWeightedSampler` + `build_group_ids` + tests (no wiring yet).
4. `GroupWeightsConfig`/`DataLoaderConfig.group_weights` + validation + config tests.
5. Wire into `get_gridded_data` + distributed/parallel test + e2e smoke.
