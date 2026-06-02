# Plan: Why Masking Hurts and How to Fix It

## Context

Variable masking (input dropout) was tested in `configs/experiments/2026-05-27-var-masking/` across mask rates (0%, 20%, 80%), masking strategies (uniform / forcing), global-mean-removal, and residual-prediction flags. The result: **no masking wins on almost every metric** (val loss, 10yr RMSE, 46yr RMSE), with 20% masking slightly beating no-masking only on 5-day RMSE.

The question is why, and whether Jeremy's proposed `UniformVariableMaskingConfig` (see `jeremy_mask.md`) would help.

---

## Root-cause diagnosis: why no masking wins

### 1. Train-test mismatch (the main problem)
Masking is **training-only** (`single_module.py:397`: `and self.module.torch_module.training`). At inference the model always sees full inputs. So masking is pure training-time noise. The model learns a corrupted-data distribution but is evaluated on the clean distribution.

### 2. No mask indicator → ambiguous signal
`include_channel_mask_inputs` is **False** in all experiments. When a variable is masked, its value is zeroed (= climatological mean in normalized space). The model **cannot distinguish** "this variable is at its climatological mean" from "this variable is missing." It must silently handle both cases with the same weights, which degrades both.

### 3. Prefix-grouped Bernoulli draws
All levels of `air_temperature_` (8 levels) share one Bernoulli draw. A 20% mask rate means entire vertical profiles disappear ~20% of the time — a very large chunk of information to lose in one draw.

### 4. Why 5-day RMSE improves slightly at 20%
Random input dropout acts as a mild regularizer against short-term pattern overfitting. The benefit is small and only visible at short horizons. Long-run climate skill is hurt more than it is helped.

---

## Assessment of Jeremy's `UniformVariableMaskingConfig`

Jeremy's plan (`jeremy_mask.md`) proposes sampling N variables uniformly instead of independent Bernoulli draws. This:
- ✅ Allows precise count control when `min_vars` and `max_vars` are set as integers (e.g. always mask exactly 1–5 vars)
- ❌ With defaults (`min_vars="min"`, `max_vars="max"`), N ~ Uniform[0, n_eligible] — actually *more* count variance than Bernoulli, not less
- ❌ Does **not** fix train-test mismatch
- ❌ Does **not** fix the ambiguous-signal problem (no mask indicators)

It's useful infrastructure for controlling masking cardinality, but the count is only constrained when integer bounds are specified. Alone it won't improve climate skill over no-masking.

---

## Recommended approach: informed masking

The highest-leverage change is already in the codebase, just turned off.

### Step 1: Enable mask indicators (no new code needed)

Set `include_channel_mask_inputs: true` in `step.config`. This appends a binary channel (1 = present, 0 = masked) for every input variable. The network can now explicitly learn "when variable X is missing, infer it from Y and Z." This directly resolves the ambiguous-signal problem.

**Config change** (in `generate_masking_configs.py` or new experiment configs):
```yaml
stepper:
  step:
    config:
      include_channel_mask_inputs: true
      input_dropout:
        default_rate: 0.2
```

Note: enabling `include_channel_mask_inputs` doubles the input channel count, so the SFNO builder must be initialized for that larger input size. Check whether the `builder` config needs updating (e.g. `in_channels` or similar field).

### Step 2: Implement `UniformVariableMaskingConfig` (Jeremy's plan)

Per `jeremy_mask.md`, add this alongside `VariableMaskingConfig`. It gives better control over masking cardinality and is a clean building block for future experiments.

**`fme/core/var_masking.py`** — add:
```python
@dataclasses.dataclass
class UniformVariableMaskingConfig:
    min_vars: int | Literal["min"] = "min"   # resolves to 0
    max_vars: int | Literal["max"] = "max"   # resolves to len(eligible)
    ignore_vars: list[str] = dataclasses.field(default_factory=list)
```
`sample_masks()`: draw `n ~ randint(lo, hi)` inclusive, then `torch.randperm(n_eligible)[:n]` to select which variables to mask. Same signature as `VariableMaskingConfig.sample_masks()`.

**`fme/core/step/single_module.py` line 90** — widen type:
```python
input_dropout: VariableMaskingConfig | UniformVariableMaskingConfig | None = None
```

Add `from typing import Literal` to `var_masking.py` and export `UniformVariableMaskingConfig` from `fme/ace/__init__.py`.

### Step 3: New experiment configs with mask indicators

Generate a new experiment set (e.g., `configs/experiments/2026-06-XX-var-masking-v2/`) that:
- Uses `include_channel_mask_inputs: true`
- Tries `UniformVariableMaskingConfig` with `min_vars=min, max_vars=max` (light, controlled masking)
- Keeps forcing variables in `ignore_vars`
- Uses 20% Bernoulli + mask indicators as a direct comparison to the v1 runs

---

## Files to modify

| File | Change |
|------|--------|
| `fme/core/var_masking.py` | Add `UniformVariableMaskingConfig` |
| `fme/core/step/single_module.py` | Widen `input_dropout` type (line 90) |
| `fme/ace/__init__.py` | Export `UniformVariableMaskingConfig` |
| `fme/core/test_var_masking.py` | Add tests per Jeremy's table in `jeremy_mask.md` |
| `configs/experiments/2026-05-27-var-masking/generate_masking_configs.py` | Add `include_channel_mask_inputs: true` variant |

---

## Verification

```bash
# Unit tests for new masking class
python -m pytest fme/core/test_var_masking.py -v

# Existing step integration tests (must still pass)
python -m pytest fme/core/step/test_step.py::test_input_dropout_applied_in_train_mode_not_eval -v
python -m pytest fme/ace/stepper/test_single_module.py::test_train_on_batch_input_dropout_zeroed_during_training -v

# Type check and lint
pre-commit run --all-files
```

After generating new configs, run a short training job to verify SFNO initializes correctly with doubled input channel count.
