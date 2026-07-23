# Make evaluating all forward steps configurable per inline-validation entry

Adds `evaluate_all_steps: bool = False` to both the ACE and coupled `InlineValidationConfig`
and plumbs it through the shared validation loop, so inline validation defaults to rolling out
only the stochastically-sampled step count per batch. `PerStepLossAggregator` is reworked to
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

- Keys are grouped into step families by the suffix pattern `^(?P<family>.*)_step_(?P<step>\d+)$`
  (plain ACE keys `loss_step_N` → family `loss`; coupled keys `loss/{realm}_step_{k}` → family
  `loss/{realm}`), plus non-step keys (e.g. coupled `loss/ocean`).
- Within a family, a rank's observed steps are contiguous from 0 (a batch evaluated for k steps
  always yields steps 0..k−1) — stated in a comment and enforced cheaply at `get_logs` time. One
  `reduce_max` on the rank's max observed step (−1 if none) yields the family's global step
  universe `range(global_max + 1)`.
- Per family: build dense sum and count vectors of length `global_max + 1` (zeros where the rank
  never saw the step), one `reduce_sum` each; mean = sum/count; steps with global count 0 are
  skipped. Non-step keys get the same sum/count treatment keyed by name.
- Family names and non-step key names are assumed rank-symmetric (as today — every rank that
  records batches produces the same families); the delta this PR must handle is per-step key sets
  differing across ranks within a family. This robustness is live for both trainers, since the
  coupled sampled default also produces sparse per-step key sets.
- Count-weighting is unconditional (no legacy path). Reported values shift slightly vs. today's
  unweighted mean-of-rank-means whenever ranks have unequal batch counts — including existing
  deterministic runs; negligible vs. batch noise. Chosen over preserving the old definition to
  avoid a validation-only fork: the same aggregator serves training-side `loss_step_N` logs, and
  `get_logs` calls are already rank-symmetric there.

## `fme/core/generics/validation.py` (modified)

```python
def run_validation_loop(
    ...,
    evaluate_all_steps: bool = True,  # NEW — passed to stepper.train_on_batch; default keeps coupled trainer and LR tuning byte-identical
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
    evaluate_all_steps: bool = False  # NEW — False evaluates only the steps the train stepper samples for the batch (the stochastically-sampled count under a schedule, the fixed count otherwise); True rolls out the full data window with dense per-step metrics

def _get_validation_callback(...) -> ValidationCallback:  # CHANGED — forwards each entry's evaluate_all_steps into its ValidationTask
    ...

def _get_validate_stepper_callback(...) -> ValidateStepper:  # CHANGED — LR tuning forwards each entry's flag to run_validation_loop
    ...
```

- Default `False` restores the pre-existing sampled-rollout behavior: with a long-tailed step
  schedule (e.g. sampled from {1, 2, 4, 12, 20}) the validation window is sized to the schedule
  max, so evaluating every step makes validation ~×13–16 slower than training-commensurate cost.
  Users who monitor long-lead metrics opt in per entry. The default flip and the count-weighting
  change are called out in the PR description/changelog, not in docstrings or runtime logging.
- Under the default, `loss_step_N` at different leads aggregates over different numbers of
  batches (unbiased but noisier at long leads); interpreting them in light of the step-sampling
  probabilities is documented as the user's responsibility.
- Per-batch step sampling during validation is already driven by `stepper.seed_eval(seed=0)` in
  `run_validation_loop` with a rank-shared seed: all ranks draw the same step count per batch
  index, and each epoch evaluates a comparable set of step counts. Key-set mismatch across ranks
  arises only from unequal batch counts.
- No `__post_init__` validation added (nothing invalid to reject). A fixed-integer step-count
  mode was considered and dropped as speculative; a later widening to `bool | int` stays
  config-compatible.

## `fme/coupled/train/train_config.py` (modified)

```python
@dataclasses.dataclass
class InlineValidationConfig:
    evaluate_all_steps: bool = False  # NEW — mirrors the ACE field; False evaluates only the steps the coupled loss samples for the batch, True rolls out the full window with dense per-realm per-step metrics

def _get_validation_callback(...) -> ValidationCallback:  # CHANGED — forwards each entry's evaluate_all_steps into its ValidationTask
    ...

def _get_validate_stepper_callback(...) -> ValidateStepper:  # CHANGED — LR tuning forwards each entry's flag to run_validation_loop
    ...
```

Same rationale and default as the ACE field. The coupled sampled-eval path already exists
(`CoupledTrainStepper.train_on_batch(evaluate_all_steps=False)` computes `loss/{realm}_step_{k}`
only for sampled steps, seeded via `seed_eval`), so this is config plumbing only.

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
    # evaluate_all_steps=False; explicit true/false parse through.
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
    # evaluate_all_steps=False; explicit true/false parse through.
    ...

def test_validation_callback_per_entry_evaluate_all_steps():
    # GOAL: entry flag reaches the coupled validation loop — with stochastic
    # per-component n_steps, a sampled entry yields sparse loss/{realm}_step_{k}
    # metrics and an all-steps entry yields dense ones.
    ...
```
