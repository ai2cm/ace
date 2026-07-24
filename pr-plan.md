# Make evaluating all forward steps configurable per inline-validation entry

Adds `evaluate_all_steps: bool = True` to both the ACE and coupled `InlineValidationConfig`
and plumbs it through the shared validation loop. The default keeps the current full-window
rollout; setting `false` per entry evaluates only the steps the train stepper itself would
sample for the batch, restoring training-commensurate validation cost under long-tailed step
schedules. `PerStepLossAggregator` is reworked to
reduce count-weighted sums with an agreed-on key universe, so ranks with mismatched
`loss_step_N` key sets neither hang nor corrupt means. Both steppers already implement both
behaviors (`train_on_batch(evaluate_all_steps=...)`); no stepper code changes.

---

## `fme/ace/aggregator/loss_metrics.py` (modified)

```python
class PerStepLossAggregator:
    def record(self, metrics: dict[str, torch.Tensor]) -> None:  # CHANGED — sums/counts kept as tensors for collective reduction
        ...

    def get_logs(self, label: str) -> dict[str, float]:  # CHANGED — distributed reduction reworked, see below
        ...
```

### Critical detail — deadlock-free, count-weighted reduction

Today `get_logs` calls `dist.reduce_mean` once per locally-observed key: ranks with different key
sets issue different collective sequences (hang), and the mean-of-rank-means ignores per-rank
counts. New scheme:

- Ranks first agree on the global key universe: each rank `gather_object`s its sorted local key
  list to global rank 0, which unions and sorts them and `scatter_object`s the result back. Even
  a rank with no recorded batches issues the same collective sequence, so nothing hangs. (This
  replaces a step-family/`reduce_max` scheme from an earlier draft of this plan: gathering the
  keys directly needs no contiguity assumption and handles arbitrary key shapes — plain
  `loss_step_N`, coupled `loss/{realm}_step_{k}`, and non-step keys — uniformly.)
- Over that universe, build dense sum and count vectors (zeros where the rank never saw the
  key), one `reduce_sum` each; mean = sum/count; keys with global count 0 are skipped. This
  robustness is live for both trainers, since coupled sampled evaluation also produces sparse
  per-step key sets. Fixing the gather/scatter path surfaced a latent bug:
  `ModelTorchDistributed.scatter_object` scattered from `_data_rank == 0` (true on every spatial
  rank) instead of global rank 0, deadlocking under spatial parallelism; it now matches
  `gather_object`.
- Count-weighting is unconditional (no legacy path). Reported values shift slightly vs. today's
  unweighted mean-of-rank-means whenever ranks have unequal batch counts — including existing
  deterministic runs. Worst case for existing (`evaluate_all_steps=True`) runs: the shift is
  bounded by the between-rank spread of per-key means times how far the count weights deviate
  from uniform — with the usual near-equal per-rank batch counts (equal or off by one) that
  deviation is ~R/(2B) (R ranks, B total batches). Concretely: R=8, B=100 (per-rank counts 12
  or 13) gives weight deviation 0.02; with per-batch loss noise of 10% of the loss value, rank
  means spread ≈ ±3%, so the shift is ≤ ~0.2% of the loss value vs. the metric's own
  batch-sampling noise of 10%/√100 = 1% — ~5× below the noise floor, as an upper bound. Even
  the degenerate extreme (2 ranks, 3 batches: weights 1/3, 2/3 vs 1/2, 1/2 → at most 1/6 of the
  between-rank difference) gives ≈ 2.3% shift against a 5.8% noise floor. (Under
  `evaluate_all_steps=false` per-key counts are arbitrarily uneven, but there is no old value to
  compare — the previous code would hang.) Chosen over preserving the old definition to
  avoid a validation-only fork: the same aggregator serves training-side `loss_step_N` logs, and
  `get_logs` calls are already rank-symmetric there.

## `fme/core/generics/validation.py` (modified)

```python
def run_validation_loop(
    ...,
    evaluate_all_steps: bool = True,  # NEW — passed to stepper.train_on_batch; all callers pass explicitly. Default is conservative: the generic layer never silently drops metrics; callers make a conscious choice to override
) -> None:
    ...

def run_validation(
    ...,
    evaluate_all_steps: bool = True,  # NEW — pass-through to run_validation_loop
) -> AggregatorSummary:
    ...
```

## `fme/core/generics/trainer.py` (modified)

```python
@dataclasses.dataclass
class ValidationTask(Generic[BD, TO]):
    evaluate_all_steps: bool = True  # NEW — both trainers pass explicitly; dense-by-default at the generic layer

def build_validation_callback(...) -> ValidationCallback:  # CHANGED — passes task.evaluate_all_steps to run_validation
    ...
```

The post-epoch train-data evaluation loop (`train_one_epoch`'s
`train_on_batch(..., evaluate_all_steps=True)`) is intentionally unchanged — this PR scopes the
flag to inline validation.

## `fme/ace/train/train_config.py` (modified)

```python
@dataclasses.dataclass
class InlineValidationConfig:
    evaluate_all_steps: bool = True  # NEW — True (default) rolls out the full data window with dense per-step metrics; False evaluates only the steps the train stepper would evaluate for the batch (the stochastically-sampled count under a schedule, the fixed count otherwise)

def _get_validation_callback(...) -> ValidationCallback:  # CHANGED — forwards each entry's evaluate_all_steps into its ValidationTask
    ...

def _get_validate_stepper_callback(...) -> ValidateStepper:  # CHANGED — LR tuning forwards each entry's flag to run_validation_loop
    ...
```

- Default `True` preserves current behavior: no run's metrics change by default, and the default
  is coherent with fixed-integer `n_forward_steps` (where `False` behaves identically). A `False`
  default (auto-restoring sampled rollout) was considered — it would spare stochastic-schedule
  users the surprise of slow validation — and rejected in review: it silently changes metrics and
  their uncertainty for existing configs. The cost win is opt-in: with a long-tailed step schedule
  (e.g. sampled from {1, 2, 4, 12, 20}) the validation window is sized to the schedule max, so
  evaluating every step makes validation ~×13–16 slower than training-commensurate cost; set
  `evaluate_all_steps: false` per entry to recover it.
- Naming: the bool encodes a policy, not an outcome — `False` = "evaluate what training would
  evaluate for this batch", `True` = "evaluate the full window regardless". An
  `evaluate_n_steps: "sampled" | "all" | int` spelling was rejected: the `int` handling is tricky
  for little added value, and `"sampled"` is a misnomer under fixed-integer `n_forward_steps`.
  The field docstring states this policy reading.
- Under `False`, `loss_step_N` at different leads aggregates over different numbers of batches:
  with `p = P(steps ≤ N)` under the schedule and `B` total validation batches, `loss_step_N`
  averages ~`B·(1−p)` batches — unbiased but noisier at long leads, and since `seed_eval(seed=0)`
  makes per-batch step draws consistent from epoch to epoch, a lead with `B·(1−p)` ≲ 1 may never
  be logged at all (larger validation batch sizes shrink `B`, making this more likely). This
  interpretation guidance goes in the field's docstring; there is no runtime logging or
  reweighting.
- Per-batch step sampling during validation is already driven by `stepper.seed_eval(seed=0)` in
  `run_validation_loop` with a rank-shared seed: all ranks draw the same step count per batch
  index, and each epoch evaluates the same set of step counts. Key-set mismatch across ranks
  arises only from unequal batch counts.
- No `__post_init__` validation added (nothing invalid to reject). A fixed-integer step-count
  mode was considered and dropped as speculative; a later widening to `bool | int` stays
  config-compatible.

## `fme/coupled/train/train_config.py` (modified)

```python
@dataclasses.dataclass
class InlineValidationConfig:
    evaluate_all_steps: bool = True  # NEW — mirrors the ACE field; True (default) rolls out the full window with dense per-realm per-step metrics, False evaluates only the steps the coupled loss samples for the batch

def _get_validation_callback(...) -> ValidationCallback:  # CHANGED — forwards each entry's evaluate_all_steps into its ValidationTask
    ...

def _get_validate_stepper_callback(...) -> ValidateStepper:  # CHANGED — LR tuning forwards each entry's flag to run_validation_loop
    ...
```

Same rationale and default as the ACE field. The coupled sampled-eval path already exists
(`CoupledTrainStepper.train_on_batch(evaluate_all_steps=False)` computes `loss/{realm}_step_{k}`
only for sampled steps, seeded via `seed_eval`), so this is config plumbing only.

## `fme/ace/stepper/loss_schedule.py` + `fme/ace/stepper/time_length_probabilities.py` (modified)

```python
@dataclasses.dataclass
class TimeLengthSchedule:
    @property
    def is_constant(self) -> bool:  # NEW — no milestones and a fixed (or single-outcome) step count
        ...

class LossSchedule:
    def init_for_epoch(self, epoch: int | None) -> None:  # CHANGED — raises EpochNotProvidedError for ANY non-constant schedule when epoch is None (previously only milestone schedules)
        ...
```

Defensive raise added in review: with a plain-probabilities schedule, `init_for_epoch(None)`
previously early-returned without building samplers, so a batch carrying `epoch=None` silently
evaluated every data step — under `evaluate_all_steps: false` that means no cost saving and
nothing raises. Real loaders stamp the epoch, so nothing legitimate hits this; the raise turns
the silent path into a hard failure. Constant schedules (fixed-int `n_forward_steps`, no
milestones) still accept `epoch=None`, where the fallback (evaluate the full window) is
identical to the sampled behavior anyway.

---

## Tests

## `fme/ace/aggregator/test_loss_metrics.py` (new)

```python
def test_per_step_loss_aggregator_single_process():
    # GOAL: sums/means over recorded batches unchanged by the reduction rework
    # (single-rank regression: record uneven key sets across records, assert means).
    ...

@pytest.mark.parallel
def test_per_step_loss_aggregator_mismatched_ranks():
    # GOAL: no hang and count-correct (count-weighted) means when ranks record
    # different loss_step_N key sets.
    # PARAMETERIZE: (a) ranks with different max steps; (b) one rank records no
    # batches at all; (c) unequal record counts for a shared step.
    ...
```

## `fme/ace/stepper/test_single_module.py` (schedule guard tests, modified)

```python
def test_stepper_step_probabilities_requires_epoch():
    # GOAL: a plain-probabilities schedule raises EpochNotProvidedError on
    # init_for_epoch(None) instead of silently evaluating densely.
    ...

def test_stepper_step_int_does_not_require_epoch():
    # GOAL: a constant (fixed-int) schedule still accepts epoch=None.
    ...
```

## `fme/core/distributed/parallel_tests/test_scatter_object.py` (new)

```python
@pytest.mark.parallel
def test_scatter_object_scatters_from_global_root():
    # GOAL: pin the global-root contract of scatter_object that
    # PerStepLossAggregator.get_logs depends on (mirrors test_gather_object).
    ...
```

## `fme/ace/stepper/test_single_module.py` (modified)

```python
def test_train_on_batch_evaluate_all_steps_with_schedule():
    # GOAL: with a probabilistic n_forward_steps_schedule, evaluate_all_steps=False
    # yields loss_step_0..loss_step_{n_sampled-1} only, while True yields metrics for
    # every data step; both count only sampled steps toward the accumulated loss.
    # PARAMETERIZE: evaluate_all_steps ∈ {False, True}.
    ...
```

## `fme/core/generics/test_validation.py` (modified)

```python
def test_run_validation_loop_evaluate_all_steps_passthrough():
    # GOAL: the flag given to run_validation_loop reaches stepper.train_on_batch,
    # and the default remains True.
    # PARAMETERIZE: evaluate_all_steps ∈ {False, True, unset}.
    ...
```

## `fme/ace/train/test_train_config.py` (modified)

```python
def test_inline_validation_config_evaluate_all_steps_default():
    # GOAL: config round-trip — YAML omitting the field parses with
    # evaluate_all_steps=True; explicit true/false parse through.
    ...

def test_validation_callback_per_entry_evaluate_all_steps():
    # GOAL: highest-seam test — run the validation callback end-to-end with a sampled
    # schedule and two entries (one sampled, one all-steps); assert sparse vs. dense
    # per-step metrics per entry and an unchanged weighted validation loss.
    ...
```

## `fme/coupled/train/test_train_config.py` (modified)

```python
def test_inline_validation_config_evaluate_all_steps_default():
    # GOAL: config round-trip — YAML omitting the field parses with
    # evaluate_all_steps=True; explicit true/false parse through.
    ...

def test_validation_callback_per_entry_evaluate_all_steps():
    # GOAL: entry flag reaches the coupled validation loop — with stochastic
    # per-component n_steps, a sampled entry yields sparse loss/{realm}_step_{k}
    # metrics and an all-steps entry yields dense ones.
    ...
```
