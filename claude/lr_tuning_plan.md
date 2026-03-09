# LR Tuning Feature — Implementation Plan

## Overview

Add an experimental feature to the generic `Trainer` that periodically tests whether
a lower learning rate would improve validation loss more than the current one. The trial
is a fully isolated subroutine: deep-copy the stepper twice, train both forks, validate
both, then discard them. The only mutation if the candidate wins is setting the LR on the
real `Optimization`. If interrupted mid-trial, `_current_epoch_num_batches_seen` is still
0 so the trial re-runs on resume.

## Key Design Decisions (from discussion with user)

- **No model weight replacement**: if candidate wins, only the LR changes on `self.optimization`.
- **Isolated subroutine**: both forks are deep-copied from the original stepper. Original is untouched.
- **No wandb logging** during the trial.
- **No explicit checkpoint** of tuning state — rely on `_current_epoch_num_batches_seen > 0` after the first real training batch to avoid re-running.
- **Pre-trial validation loss** comes from the end of the previous epoch (already computed). Only run validation if there's no prior result (e.g., start of training with `evaluate_before_training=False`).
- **Same data ordering**: trial uses `train_data.subset_loader(stop_batch=N)` from the epoch start. The real epoch then proceeds normally from batch 0 with unchanged shuffle.
- **Optimization state must be deep-copied** for the trial forks so optimizer momentum is preserved.

## Configuration

```python
# fme/core/generics/lr_tuning.py
@dataclasses.dataclass
class LRTuningConfig:
    epoch_frequency: int          # run trial every N epochs
    lr_factor: float              # candidate LR = current_lr * lr_factor
    num_batches: int              # batches to train each fork
    improvement_threshold: float  # e.g. 0.1 means candidate must improve >=10% more
```

## Commits

### Commit 1: `Optimization` changes

**Goal**: Add `set_learning_rate` to `Optimization` and add `OptimizationABC.set_learning_rate`
so both real and null implementations have it. Also ensure optimization state can be loaded
into a freshly-built `Optimization` for different parameter objects (needed for the trial
forks to preserve momentum).

**Files changed**:
- `fme/core/optimization.py` — add `set_learning_rate()` method to `Optimization`
- `fme/core/generics/optimization.py` — add `set_learning_rate()` to `OptimizationABC`
- Tests for LR setting and state loading across different parameter objects

**`set_learning_rate`**:
```python
def set_learning_rate(self, lr: float):
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
```

**Optimization state copying for trial forks**: When we `build_optimization(fork.modules)`,
we get a fresh optimizer with no momentum. For a fair trial, the forked optimizers should
start with the same momentum state as the original. The approach:
- After building, call `fork_opt.load_state(original_opt.get_state())`.
- Then call `fork_opt.set_learning_rate(desired_lr)` for the candidate.
- PyTorch's `optimizer.load_state_dict` re-maps state by param-group index, so this
  works when parameter ordering is identical (guaranteed by deepcopy).
- Test that loading state_dict from one optimizer into another (with different parameter
  objects but same structure) correctly preserves momentum.

### Commit 2: Extract `run_validation` helper

**Goal**: Pull the validation loop out of `Trainer.validate_one_epoch` into a standalone
function. The Trainer method delegates to it (keeping its `_current_epoch_num_batches_seen`
assertion). Add isolated unit tests for the new function.

**Files changed**:
- `fme/core/generics/trainer.py` — add module-level `run_validation()`, refactor `validate_one_epoch` to delegate
- `fme/core/generics/test_trainer.py` — add tests for `run_validation` directly

**`run_validation` signature**:
```python
def run_validation(
    stepper: TrainStepperABC,
    valid_data: GriddedDataABC,
    get_aggregator: Callable[[], AggregatorABC[TO]],
    ema: EMATracker | None,
    validate_using_ema: bool,
) -> dict[str, Any]:
```

Takes `get_aggregator` (i.e. `aggregator_builder.get_validation_aggregator`) rather than
the full `AggregatorBuilderABC` — minimize the API surface.

The function:
1. Calls `stepper.set_eval()`
2. Creates aggregator via `get_aggregator()`
3. Optionally enters `ema.applied_params(stepper.modules)` context
4. Iterates `valid_data.loader` with `torch.no_grad()`, calling `stepper.train_on_batch(..., NullOptimization(), compute_derived_variables=True)`
5. Returns `aggregator.get_logs(label="val")`
6. Does NOT call `aggregator.flush_diagnostics` (caller decides)

**Trainer.validate_one_epoch** keeps its assertion and delegates:
```python
def validate_one_epoch(self):
    if self._current_epoch_num_batches_seen > 0:
        raise RuntimeError(...)
    ...
    logs = run_validation(
        stepper=self.stepper,
        valid_data=self.valid_data,
        get_aggregator=self._aggregator_builder.get_validation_aggregator,
        ema=self._ema,
        validate_using_ema=self.config.validate_using_ema,
    )
    aggregator.flush_diagnostics(...)  # flush still handled by Trainer
    return logs
```

Actually, since `run_validation` creates the aggregator internally and the Trainer needs
to call `flush_diagnostics` on it, the function should return both the logs and the
aggregator, or the Trainer should pass in the aggregator. Simplest: have `run_validation`
accept the aggregator directly (already built by the caller) instead of a factory.
Then the Trainer builds the aggregator, passes it in, and flushes after. The trial
builds its own aggregator, passes it in, and skips flushing.

**Revised signature**:
```python
def run_validation(
    stepper: TrainStepperABC,
    valid_data: GriddedDataABC,
    aggregator: AggregatorABC[TO],
    ema: EMATracker | None,
    validate_using_ema: bool,
) -> dict[str, Any]:
```

**Test approach**: Use the existing mock classes in `test_trainer.py` (`TrainStepper`,
`TrainData`, `ValidationAggregator`). Test that:
- The returned logs contain `val/mean/loss`
- The stepper is set to eval mode
- With `validate_using_ema=True`, EMA params are applied during validation
- With `validate_using_ema=False`, EMA is not applied

### Commit 3: `LRTuningConfig` and `run_lr_tuning_trial`

**Goal**: Implement the trial function and its config. Unit-tested independently —
not yet wired into Trainer.

**Files created/changed**:
- `fme/core/generics/lr_tuning.py` — new file with `LRTuningConfig` and `run_lr_tuning_trial()`
- `fme/core/generics/test_lr_tuning.py` — new file with unit tests

**`run_lr_tuning_trial` signature**:
```python
def run_lr_tuning_trial(
    stepper: TrainStepperABC,
    train_data: GriddedDataABC,
    valid_data: GriddedDataABC,
    optimization: Optimization,
    build_optimization: Callable[[torch.nn.ModuleList], Optimization],
    build_ema: Callable[[torch.nn.ModuleList], EMATracker],
    config: LRTuningConfig,
    current_lr: float,
    pre_trial_val_loss: float,
    get_validation_aggregator: Callable[[], AggregatorABC],
    validate_using_ema: bool,
) -> float | None:
```

Takes `optimization` (the real one) so it can copy momentum state into the forks.
Takes `get_validation_aggregator` rather than the full builder — minimize API.

**Internal flow**:
1. `candidate_lr = current_lr * config.lr_factor`
2. `baseline_stepper = copy.deepcopy(stepper)`
3. `candidate_stepper = copy.deepcopy(stepper)`
4. Build optimization for each via `build_optimization(fork.modules)`, then
   `fork_opt.load_state(optimization.get_state())`. Set candidate's LR via
   `candidate_opt.set_learning_rate(candidate_lr)`.
5. Build EMA for each via `build_ema(fork.modules)`
6. Iterate `train_data.subset_loader(stop_batch=config.num_batches)`:
   - `fork_stepper.set_train()`
   - `baseline_stepper.train_on_batch(batch, baseline_opt)` + `baseline_ema(baseline_stepper.modules)`
   - `candidate_stepper.train_on_batch(batch, candidate_opt)` + `candidate_ema(candidate_stepper.modules)`
7. Validate both via `run_validation()` (passing `get_validation_aggregator()` as the aggregator)
8. `baseline_improvement = pre_trial_val_loss - baseline_val_loss`
9. `candidate_improvement = pre_trial_val_loss - candidate_val_loss`
10. If both > 0 and `candidate_improvement > baseline_improvement * (1 + config.improvement_threshold)`: return `candidate_lr`
11. Else return `None`

**Note on train_data epoch**: The caller (`_maybe_tune_lr`) should call
`train_data.set_epoch` for the trial before invoking. `train_one_epoch` already
calls `set_epoch(self._epochs_trained + 1)` so the ordering is restored
automatically. The trial should use the same epoch value so it sees the same
first N batches the real epoch will.

**Test cases**:
- Candidate wins (improvement exceeds threshold) → returns new LR
- Candidate improves but below threshold → returns None
- Candidate worsens → returns None
- Both worsen → returns None
- Baseline worsens, candidate improves → returns None (requirement: both must improve)

### Commit 4: Wire into Trainer + ACE TrainConfig

**Goal**: Connect everything into the Trainer and expose it in the ACE training config.

**Files changed**:
- `fme/core/generics/trainer.py`:
  - `TrainConfigProtocol`: add `lr_tuning: LRTuningConfig | None` property
  - `Trainer.__init__`: store `_build_optimization`, `_build_ema`, init `_last_val_loss = None`
  - `Trainer.train()`: store `valid_loss` into `_last_val_loss` after each validation; call `_maybe_tune_lr()` before `train_one_epoch()`
  - Add `Trainer._maybe_tune_lr()` method
- `fme/ace/train.py` (or wherever ACE's TrainConfig lives):
  - Add `lr_tuning: LRTuningConfig | None = None` field
- `fme/core/generics/test_trainer.py` — update mock `Config` to include `lr_tuning`, add integration-style tests for the full Trainer loop with LR tuning enabled

**`_maybe_tune_lr` logic**:
```python
def _maybe_tune_lr(self):
    cfg = self.config.lr_tuning
    if cfg is None:
        return
    if self._current_epoch_num_batches_seen > 0:
        return  # resumed mid-epoch, skip
    if self._epochs_trained % cfg.epoch_frequency != 0:
        return
    if self._last_val_loss is None:
        val_logs = self.validate_one_epoch()
        self._last_val_loss = val_logs["val/mean/loss"]
    new_lr = run_lr_tuning_trial(
        stepper=self.stepper,
        train_data=self.train_data,
        valid_data=self.valid_data,
        optimization=self.optimization,
        build_optimization=self._build_optimization,
        build_ema=self._build_ema,
        config=cfg,
        current_lr=self.optimization.learning_rate,
        pre_trial_val_loss=self._last_val_loss,
        get_validation_aggregator=self._aggregator_builder.get_validation_aggregator,
        validate_using_ema=self.config.validate_using_ema,
    )
    if new_lr is not None:
        logging.info(f"LR tuning: adopting candidate LR {new_lr}")
        self.optimization.set_learning_rate(new_lr)
```

## Key file locations

- Trainer: `fme/core/generics/trainer.py`
- Trainer tests: `fme/core/generics/test_trainer.py`
- Optimization: `fme/core/optimization.py`
- Optimization ABC: `fme/core/generics/optimization.py`
- EMA: `fme/core/ema.py`
- TrainStepper ABC: `fme/core/generics/train_stepper.py`
- GriddedData ABC: `fme/core/generics/data.py`
- Aggregator ABCs: `fme/core/generics/aggregator.py`
- ACE train config: `fme/ace/train.py` (to be confirmed at implementation time)

## Test infrastructure

Tests in `fme/core/generics/test_trainer.py` use custom mock implementations:
- `TrainStepper(TrainStepperABC)` — wraps a `torch.nn.Linear(1,1)`, tracks batches
- `TrainData(GriddedDataABC)` — controllable shuffle and batch iteration
- `TrainAggregator` / `ValidationAggregator` — return configurable loss values
- `AggregatorBuilder` — factory for the above
- `Config` dataclass — protocol-conforming mock config
- `get_trainer()` factory function — builds a fully configured Trainer with mocks
- `fail_after_calls_patch()` / `preempt_after_calls_patch()` — for interruption testing
- `mock_wandb()` context manager — captures wandb logs without real W&B calls

Reuse these for new tests wherever possible.
