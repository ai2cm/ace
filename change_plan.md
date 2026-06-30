# Plan: Minimal `input_dropout` variable masking

## Context

The branch `feature/var_masking_simple` adds synthetic input-variable dropout (training-time
random masking of input channels) but does so with more machinery than needed: a *union* of two
mutually-exclusive masking configs (`UniformMaskingConfig | PerVariableMaskingConfig`), **per-sample**
masks of shape `[batch, n_channels]`, and `n_ensemble` threading via `_repeat_interleaved_tensor_mapping`
so each base sample's mask repeats across its ensemble members (20 files, ~1016 insertions).

We want a **simpler reimplementation against `main`** with these properties:

1. Add `input_dropout` variable masking.
2. Masks are **broadcast across the whole batch** → `n_ensemble` is never threaded through.
3. Every sample and every ensemble member therefore gets the **same mask** (free consequence of #2).
4. `min_vars` is **hard-coded to 0**, not a config option.
5. Uniform masking with `max_vars` an **integer** (uniform count of dropped vars in `[0, max_vars]`).
6. A mapping of **variable name → bernoulli rate** can also be provided (per-variable masking).
7. The bernoulli mask does **not** OR/AND with the uniform mask. Instead, each bernoulli var that fires
   **replaces** a randomly-selected var already in the uniform drop-set with itself. This keeps the total
   dropped count equal to the uniform count `k` and guarantees fired bernoulli vars are dropped, without
   any logical AND/OR. Bernoulli vars cannot replace other fired bernoulli vars.

Intended outcome: same user-facing capability as the branch (training-only input corruption, correct under
spatial-parallel and multi_call, blocked in coupled training) with a single combined config and no
ensemble threading.

The reference implementation on `feature/var_masking_simple` should be consulted for exact wording of
docstrings/tests, but **do not** copy the union-type config or the `n_ensemble`/`_repeat_interleaved_tensor_mapping`
machinery.

## Design summary

- **One** config class `VariableMaskingConfig` holds *both* the uniform count and the per-variable rates;
  there is no union, no `kind` discriminator.
- The sampled mask has **no batch dimension** — shape `[1, n_channels]` so it broadcasts over the batch
  (and hence over ensemble members) for free. This is what removes all `n_ensemble` threading.
- Mask is sampled **once per rollout** up in `TrainStepper` and threaded down through the existing
  `StepArgs.input_dropout_mask` field → constant across forward steps and across all multi_call sub-calls.
- Decisions confirmed with user: **include** spatial-broadcast scaffolding, coupled-training guard, and
  multi_call delegation.

## Changes

### 1. `fme/core/var_masking.py` (NEW — replaces branch's union design)

Single dataclass:

```python
@dataclasses.dataclass
class VariableMaskingConfig:
    max_masked_vars: int = 0                      # uniform component; min is hard-coded 0
    variable_masking_rates: dict[str, float] | None = None   # per-variable bernoulli

    def __post_init__(self):
        # max_masked_vars must be a non-negative int
        # each rate in variable_masking_rates must be in [0, 1]

    def sample_mask(self, names: list[str], device: torch.device) -> torch.Tensor:
        # returns a PRESENCE mask of shape [1, n_channels] (True = keep, False = dropped)
```

`sample_mask` logic (n = `len(names)`):
- **uniform drop**: draw count `k` uniformly from `[0, min(max_masked_vars, n)]`; drop `k` random channels.
  Reuse the branch's argsort trick (`noise.argsort.argsort >= k`) with `batch_size=1`, or a `randperm`.
- **bernoulli fire**: for each `name` present in `variable_masking_rates`, draw `Bernoulli(rate)`; collect the
  set of *fired* channels.
- **combine by replacement (req 7)**: start from the uniform `drop` set (size `k`). For each fired bernoulli
  channel **not already in `drop`**, evict one randomly-selected channel that is in `drop` **and not itself a
  fired bernoulli channel**, then add the fired channel. Net: `|drop|` stays `k`, every evictable slot can be
  taken by a fired bernoulli var. Return `present = ~drop`, shape `[1, n_channels]`.
  - Edge cases: if `k == 0` there are no slots, so no bernoulli var can be dropped (replacement is a no-op).
    If more bernoulli vars fire than there are evictable (non-fired) uniform slots, only `k`-worth are dropped;
    which fired vars win is random. Document both in the docstring.

Note the present-mask convention (True = keep) is what the downstream helpers already expect, so building the
*drop* set then negating keeps `_apply_input_mask` / `_build_channel_mask_dict` unchanged.

Export `VariableMaskingConfig` from `fme/ace/__init__.py` (drop the `UniformMaskingConfig` /
`PerVariableMaskingConfig` exports the branch added).

### 2. `fme/core/step/args.py`

Add the `input_dropout_mask: TensorMapping | None = None` field to `StepArgs.__init__`, preserve it in
`apply_input_process_func`, and document it. **Identical to the branch** except the docstring says values are
`[1]`-shaped (broadcast over batch) rather than `[n_batch]`. (Branch ref: `fme/core/step/args.py` diff.)

### 3. `fme/core/step/step.py` (ABC)

Add default `make_input_dropout_mask(self, device) -> TensorMapping | None` returning `None`, and
`has_input_dropout(self) -> bool` returning `False`. **Drop the `batch_size` parameter** the branch had —
the batch-broadcast mask does not depend on batch size. (Branch ref: `step.py:359-384`.)

### 4. `fme/core/step/single_module.py`

- Add `input_dropout: VariableMaskingConfig | None = None` to `SingleModuleStepConfig`.
- In `SingleModuleStep.step`, apply the mask after normalization, before packing — **unchanged** from branch
  (`single_module.py:387-391`): `if args.input_dropout_mask is not None: input_norm = _apply_input_mask(...)`.
- Implement `make_input_dropout_mask(self, device)` (no `batch_size`):
  - return `None` if `self._config.input_dropout is None` or module not in `training` mode;
  - `mask = self._config.input_dropout.sample_mask(self.in_packer.names, device)` → `[1, n_channels]`;
  - `mask = Distributed.get_instance().broadcast_spatial(mask)`;
  - return `{name: mask[:, i] for i, name in enumerate(names)}` (each value `[1]`-shaped).
- `has_input_dropout` returns `self._config.input_dropout is not None`.
- **Reuse as-is**: `_apply_input_mask` (`:513`) and `_build_channel_mask_dict` (`:529`) already broadcast
  `[1]`/`[batch]` masks correctly via `torch.where` / `&`.

### 5. `fme/core/step/multi_call.py`

`MultiCallStep` delegates both hooks to the wrapped step (drop `batch_size`):
```python
def make_input_dropout_mask(self, device): return self._wrapped_step.make_input_dropout_mask(device)
def has_input_dropout(self): return self._wrapped_step.has_input_dropout()
```
The same mask reaches every sub-call automatically: `MultiCallStep.step` passes the same `args` to base and
multi-call, and `MultiCall.step` propagates it via `args.apply_input_process_func` (`_multi_call.py:179`).

### 6. `fme/ace/stepper/single_module.py`

- `Stepper.make_input_dropout_mask(self, device)` delegates to `self._step_obj.make_input_dropout_mask(device)`
  (drop `batch_size`); `has_input_dropout` delegates likewise. (Branch ref: `:936-942`.)
- `predict_generator` keeps the `input_dropout_mask: TensorMapping | None = None` param and passes it into
  each step's `StepArgs` (constant across the rollout). **Unchanged** from branch (`:1141,:1170`).
- In `_accumulate_loss`, sample **once**, no ensemble repeat:
  ```python
  input_dropout_mask = self._stepper.make_input_dropout_mask(device=sample_tensor.device)
  ```
  passed straight to `predict_generator`.
- **DELETE** `_repeat_interleaved_tensor_mapping` (`:485`) and all `n_ensemble` references in the mask path.

### 7. Distributed: spatial broadcast (port branch verbatim)

Add `broadcast_spatial(self, tensor) -> torch.Tensor` to:
- `fme/core/distributed/base.py` — abstract method + docstring.
- `fme/core/distributed/distributed.py` — `Distributed` wrapper delegates to backend.
- `fme/core/distributed/model_torch_distributed.py` — broadcast from spatial-group root when `h_size>1 or w_size>1`.
- `fme/core/distributed/non_distributed.py` — identity.
- `fme/core/distributed/torch_distributed.py` — identity (data-parallel only).

These diffs are small and self-contained; copy them from the branch.

### 8. `fme/coupled/stepper.py` — coupled guard

At stepper init, raise if any component step `has_input_dropout()` (coupled training never calls the hook,
so a configured dropout would be a silent no-op). Port the branch's check (`stepper.py:1770-1780`).

## Tests

Mirror the branch's tests, dropping anything ensemble-threading-specific and adding OR-combination coverage:

- `fme/core/test_var_masking.py`: validation rejects negative `max_masked_vars` and rates outside `[0,1]`;
  uniform count lands in `[0, max]` with 0 possible (min hard-coded 0); bernoulli rate 0 → keep all;
  **replacement combination**: with `max_masked_vars=0` a rate-1 var is **not** dropped (no slots);
  with `max_masked_vars>=1` a rate-1 var is always dropped and total dropped count stays `k`; a fired
  bernoulli var already in the uniform set leaves the set unchanged; `sample_mask` returns `[1, n_channels]` bool.
- `fme/core/step/test_step.py`: masked channels zeroed in network input; indicator channel reflects
  `data_mask & input_dropout_mask`; `make_input_dropout_mask` is `None` when unset and in eval mode;
  `has_input_dropout` stays `True` in eval.
- `fme/ace/stepper/test_single_module.py`: mask constant across rollout steps; **every batch/ensemble member
  shares the same mask** (replaces the branch's "ensemble members share mask" test); eval-mode training batch
  applies no dropout; `predict()` applies no dropout.
- `fme/core/distributed/parallel_tests/test_backend.py` + `test_step.py`: `broadcast_spatial` backend test;
  mask identical across spatial tiles of a sample.
- `fme/coupled/test_stepper.py`: coupled training with `input_dropout` raises.

## Verification

- `pre-commit run --all-files` (ruff, ruff-format, mypy).
- `python -m pytest fme/core/test_var_masking.py fme/core/step/test_step.py fme/ace/stepper/test_single_module.py fme/coupled/test_stepper.py`
- Spatial-parallel smoke test (mask identity across tiles):
  ```
  FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=1 \
    torchrun --nproc-per-node 2 -m pytest -m parallel fme/core/distributed/parallel_tests/test_step.py
  ```
- Manual config check: a `SingleModuleStepConfig` with
  `input_dropout: {max_masked_vars: 3, variable_masking_rates: {air_temperature: 0.1}}`
  loads, trains (masks vary per rollout, identical across batch), and `predict()` is unaffected.

## Out of scope

- The branch's union-type configs (`UniformMaskingConfig`, `PerVariableMaskingConfig`).
- `n_ensemble` threading / `_repeat_interleaved_tensor_mapping`.
- Any `min_vars` config (hard-coded 0).
