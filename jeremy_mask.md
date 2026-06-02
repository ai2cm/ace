# Plan: Add `UniformVariableMaskingConfig` alongside existing `VariableMaskingConfig`

## Context

The current `input_dropout` config (`VariableMaskingConfig`) uses per-variable Bernoulli probabilities (`default_rate`, `rates`). The goal is to add a second, parallel option: a "uniform" scheme where a random integer `n` is drawn from `[min_vars, max_vars]` and `n` variables are uniformly sampled (from those not in `ignore_vars`) and masked. Both options remain available; users choose which to use in their YAML config. Both expose the same `sample_masks()` signature so the calling code in `single_module.py` is unchanged.

---

## Files to change

### 1. `fme/core/var_masking.py` — Add `UniformVariableMaskingConfig`

Keep `VariableMaskingConfig` entirely unchanged. Add a new dataclass below it:

```python
@dataclasses.dataclass
class UniformVariableMaskingConfig:
    min_vars: int | Literal["min"] = "min"
    max_vars: int | Literal["max"] = "max"
    ignore_vars: list[str] = dataclasses.field(default_factory=list)
```

**`__post_init__` validations (config-time):**
- `min_vars` must be `"min"` or a non-negative `int`; else raise `ValueError`
- `max_vars` must be `"max"` or a non-negative `int`; else raise `ValueError`
- If both are `int`, `min_vars <= max_vars`; else raise `ValueError`

**`sample_masks(variable_names, batch_size, device=None)` logic:**
1. `eligible = [v for v in variable_names if v not in ignore_vars]`
2. `lo = 0 if min_vars == "min" else min_vars`
3. `hi = len(eligible) if max_vars == "max" else max_vars`
4. At-sample-time validation: raise `ValueError` if `lo > len(eligible)` (integer `min_vars` too large) or `hi > len(eligible)` (integer `max_vars` too large), or `lo > hi`
5. Return `{}` immediately when `hi == 0`
6. Build a `[batch_size, n_eligible]` bool tensor (initially all True). For each row `i`, draw `n = randint(lo, hi)` inclusive via `torch.randint`, then use `torch.randperm(n_eligible)[:n]` to select which positions to set False
7. Return `dict[name, bool_tensor]` (True = present, False = masked), only for variables masked in at least one sample. Move to `device` if provided.

**Add needed import:** `from typing import Literal`

**Export `UniformVariableMaskingConfig`** from `fme/ace/__init__.py` alongside the existing export.

---

### 2. `fme/core/step/single_module.py` — Widen the `input_dropout` type

Line 90, change:
```python
input_dropout: VariableMaskingConfig | None = None
```
to:
```python
input_dropout: VariableMaskingConfig | UniformVariableMaskingConfig | None = None
```

Add the import of `UniformVariableMaskingConfig` from `fme.core.var_masking`. No other changes — `sample_masks()` has the same signature on both classes.

---

### 3. `fme/core/test_var_masking.py` — Add tests for `UniformVariableMaskingConfig`

Keep all existing tests for `VariableMaskingConfig`. Append new tests:

| Test | What it checks |
|---|---|
| `test_uniform_invalid_min_vars_negative` | `min_vars=-1` raises `ValueError` |
| `test_uniform_invalid_max_vars_negative` | `max_vars=-1` raises `ValueError` |
| `test_uniform_invalid_min_gt_max` | `min_vars=3, max_vars=1` raises `ValueError` |
| `test_uniform_min_max_strings_accepted` | `min_vars="min", max_vars="max"` constructs without error |
| `test_uniform_min_vars_exceeds_eligible_raises` | `min_vars=5` with 2 eligible vars raises at `sample_masks` time |
| `test_uniform_max_vars_exceeds_eligible_raises` | `max_vars=5` with 2 eligible vars raises at `sample_masks` time |
| `test_uniform_ignore_vars_excluded` | variables in `ignore_vars` never appear in returned masks |
| `test_uniform_mask_count_in_range` | masked-variable count per sample is always in `[lo, hi]` |
| `test_uniform_min_max_zero_returns_empty` | `min_vars=0, max_vars=0` returns `{}` |
| `test_uniform_max_vars_max_string` | `max_vars="max"` can mask all eligible variables |
| `test_uniform_device_placement` | returned tensors are on the requested device |

---

### 4. No changes needed

- `fme/core/step/test_step.py` — existing tests use `VariableMaskingConfig`, which is unchanged
- `fme/ace/stepper/test_single_module.py` — same; existing dropout tests stay valid
- All other calling code — `sample_masks()` signature is identical on both classes

---

## Verification

```bash
# Unit tests for both masking config classes
python -m pytest fme/core/test_var_masking.py -v

# Step integration test (uses VariableMaskingConfig, should still pass)
python -m pytest fme/core/step/test_step.py::test_input_dropout_applied_in_train_mode_not_eval -v

# Stepper tests
python -m pytest fme/ace/stepper/test_single_module.py::test_train_on_batch_input_dropout_zeroed_during_training fme/ace/stepper/test_single_module.py::test_train_on_batch_input_dropout_merged_with_data_mask -v

# Type checking and linting
pre-commit run --all-files
```
