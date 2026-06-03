# Implementation Plan: Combined Infill-Prediction Training

## Context

ACE currently trains models for a single task: forward prediction. The design in `infill-prediction-design.md` specifies a new step type (`InfillPredictionStep`) that trains on multiple tasks per batch (auto-encoding, infill, prediction, infill-prediction, combined-all) by randomly masking variables. At inference time, it behaves like the existing `SingleModuleStep`.

**Multi-task per update**: Rather than replacing the final step's prediction loss with a task loss, we compute the normal prediction loss for ALL N forward steps, then run one additional forward pass with task masking and add its loss. Every gradient update includes both the prediction objective and a task objective.

**Per-sample task sampling**: Task type and variable assignments are sampled independently for each sample in the batch, since `data_mask` (variable availability) varies per sample.

This plan breaks the implementation into 3 independently reviewable, tested commits.

---

## Commit 1: Config dataclasses + TaskSampler with tests

**New file:** `fme/core/step/infill_prediction.py`

### Types to add

- **`TaskWeights`** — dataclass with sampling weights and loss scales per task type (10 float fields)
- **`TaskSamplingConfig`** — wraps `TaskWeights` + `min_input_variables: int` + `min_output_variables: int`
- **`InferenceSchemeConfig`** — dataclass: `in_names`, `out_names`, `next_step_forcing_names`, `prescribed_prognostic_names`

### SampledTasks — per-sample task assignments

Instead of a single `TrainingTask` with lists of variable names (which can't represent per-sample variation), the sampler returns per-sample masks:

```python
@dataclasses.dataclass
class SampledTasks:
    """Per-sample task assignments for one training batch.

    Attributes:
        previous_step_input_mask: {var_name: [batch] bool} — True where the
            variable should be taken from the previous-step model state.
        current_step_input_mask: {var_name: [batch] bool} — True where the
            variable should be taken from current-step ground truth.
        output_data_mask: {var_name: [batch] float} — loss_scale where the
            variable is a prediction target, 0.0 otherwise. Float values
            (rather than bool) naturally weight different task types in the
            loss computation.
    """
    previous_step_input_mask: dict[str, torch.Tensor]
    current_step_input_mask: dict[str, torch.Tensor]
    output_data_mask: dict[str, torch.Tensor]
```

The combined `input_data_mask` (for passing to the step) is `prev | curr` per variable.

The `output_data_mask` uses float values equal to the task's `loss_scale` (instead of bool) so that the existing `WeightedMappingLoss` reduction (`(bc * mask).sum(dim=0) / mask.sum(dim=0)`) naturally produces a properly weighted average across samples with different task types.

### TaskSampler class

```python
class TaskSampler:
    def __init__(self, config: TaskSamplingConfig, all_names: list[str], forcing_names: list[str]): ...
    def sample(self, data_mask: TensorMapping | None, batch_size: int) -> SampledTasks: ...
```

Key implementation details:
- Pre-compute which tasks are feasible given variable counts and constraints; raise `ValueError` in `__init__` if a task with non-zero weight is infeasible
- `sample()` loops over batch samples. For each sample:
  1. Filter available variables using that sample's `data_mask`
  2. Draw a task type from the weighted distribution
  3. Sample variable assignments per the task's constraints (see below)
  4. Set the corresponding entries in the mask tensors

**Task constraints:**

| Task | Previous-step inputs | Current-step inputs | Outputs | Key constraint |
|------|---------------------|--------------------|---------|----|
| auto_encode | (none) | subset of non-forcing | exactly the current inputs | outputs = inputs |
| infill | (none) | from all_names | from non-forcing; disjoint from current inputs | inputs ∩ outputs = ∅ |
| prediction | from all_names | (none) | from non-forcing | standard forward prediction |
| infill_prediction | ≥1 from all_names | ≥1 from all_names | from non-forcing; disjoint from current inputs | mixed timestep |
| combined_all | from all_names | from all_names | from non-forcing | at least 1 input, 1 output |

All tasks require at least one output variable (`min_output_variables ≥ 1` enforced in config validation).

### Variable selection algorithm

The core challenge is sampling input and output variable sets without biasing toward more outputs than inputs (or vice versa). For tasks where current-step inputs and outputs must be disjoint, they compete over the same pool of non-forcing variables. Forcing variables are input-only and don't compete.

**Disjoint tasks (infill, infill_prediction current-step portion):**

Non-forcing variables are the "contested pool" — each can be either a current-step input or an output, but not both. Forcing variables are sampled independently as additional inputs.

```
# 1. Sample forcing inputs (independent of the contested split)
n_additional_in ~ Uniform(0, n_forcing_available)

# 2. Determine how many contested inputs are needed given forcing coverage
min_in_contested = max(0, min_input_variables - n_additional_in)

# 3. Sample total contested participants
n_total ~ Uniform(min_output_variables + min_in_contested, n_non_forcing_available)

# 4. Split contested pool symmetrically
n_out ~ Uniform(min_output_variables, n_total - min_in_contested)
n_in_contested = n_total - n_out

# 5. Assign specific variables to roles (uniform random subsets)
Shuffle available non-forcing variables, take first n_total as participants
Shuffle participants, take first n_out as outputs, rest as current-step inputs
Shuffle available forcing variables, take first n_additional_in as additional inputs
```

Variable assignment in step 5 uses **uniform random subsets** — every variable in a pool has equal probability of being selected for each role. Implemented via `random.sample(pool, k)` (equivalent to shuffling and taking the first k). No variable is privileged by its position in `all_names`.

When `min_input_variables = min_output_variables` and `n_additional_in = 0`, the split in step 4 is perfectly symmetric: `E[n_out] = E[n_in_contested]`. When forcing inputs are sampled, they add to the input side without distorting the contested-pool symmetry. When `n_additional_in >= min_input_variables`, the contested pool can be entirely outputs (all inputs are forcing).

**Non-disjoint tasks (prediction, combined_all):**

Inputs and outputs don't compete (inputs come from t-1, or overlap is allowed), so counts and assignments are independent:

```
n_out ~ Uniform(min_output_variables, n_non_forcing_available)
n_in ~ Uniform(min_input_variables, n_all_available)
outputs = random.sample(available_non_forcing, n_out)
inputs = random.sample(available_all, n_in)
```

For **prediction**, all inputs are previous-step. For **combined_all**, inputs can be previous-step, current-step, or both — each selected input variable is independently assigned to previous-step, current-step, or both with equal probability.

**Auto-encode:**

```
n ~ Uniform(max(min_input_variables, min_output_variables), n_non_forcing_available)
selected = random.sample(available_non_forcing, n)
# selected are both current-step inputs and outputs
```

**Infill_prediction specifics:**

Uses the disjoint algorithm above for the current-step input/output split, then additionally samples previous-step inputs (at least 1) from all available variables independently:

```
# Disjoint split for current-step (as above, steps 1-5)
...
# Additionally, sample previous-step inputs
n_prev ~ Uniform(1, n_all_available)
prev_inputs = random.sample(available_all, n_prev)
```

### Tests (`fme/core/step/test_infill_prediction.py`)

**Constraint validation tests** (deterministic, per task type):
- TaskSampler produces valid SampledTasks for each task type
- auto_encode: output variables are exactly the current-step input variables
- infill/infill_prediction: current-step inputs ∩ outputs = ∅ for every sample
- prediction: no current-step inputs, all inputs are previous-step
- All tasks: outputs only contain non-forcing variables
- All tasks: at least `min_output_variables` outputs per sample
- All tasks: at least `min_input_variables` total inputs per sample
- Per-sample data_mask: absent variables never appear as inputs or outputs for that sample
- output_data_mask values are the task's loss_scale (float), not bool

**Config/error tests:**
- Infeasible task (non-zero weight but impossible given variable counts) raises ValueError
- Zero-weight tasks never sampled (run many samples, verify)
- `min_output_variables < 1` raises ValueError

**Variable selection algorithm tests** (statistical, run many samples):
- Symmetry: for disjoint tasks with `min_in = min_out`, `E[n_out] ≈ E[n_in_contested]` (within statistical tolerance)
- Uniformity: each non-forcing variable appears as output with approximately equal frequency
- Uniformity: each variable appears as input with approximately equal frequency
- Forcing variables appear as inputs but never as outputs
- Full coverage: with enough samples, at least one sample has all non-forcing as outputs
- Full coverage: with enough samples, at least one sample has all non-forcing as inputs
- Forcing coverage: some samples have all-forcing inputs (no contested inputs)

**Per-sample independence tests:**
- Different samples in the same batch can get different task types
- Samples with different data_masks get different variable pools

---

## Commit 2: InfillPredictionStepConfig + InfillPredictionStep with tests

**Modified files:**
- `fme/core/step/step.py` — add `all_training_names` property to `StepConfigABC` and `StepSelector`
- `fme/ace/stepper/single_module.py` — update `StepperConfig.all_names` to use `all_training_names`
- `fme/core/step/__init__.py` — import new types

**New code in:** `fme/core/step/infill_prediction.py`

### StepConfigABC change (small, prerequisite)

Add to `StepConfigABC`:
```python
@property
def all_training_names(self) -> list[str] | None:
    return None
```

Add delegation in `StepSelector` (same pattern as other properties).

Update `StepperConfig.all_names` (line 678 of `fme/ace/stepper/single_module.py`):
```python
@property
def all_names(self) -> list[str]:
    training_names = self.step.all_training_names
    if training_names is not None:
        return training_names
    return list(set(self.input_names + self.output_names))
```

**Why this is needed:** `StepperConfig.all_names` drives data loading requirements. For InfillPredictionStep, `input_names`/`output_names` derive from the inference scheme (a subset), but training needs ALL variables loaded.

### InfillPredictionStepConfig

Registered as `@StepSelector.register("infill_prediction")`. Implements `StepConfigABC`.

Fields: `builder`, `all_names`, `forcing_names`, `normalization`, `inference_scheme`, `include_channel_mask_inputs=True`, plus optional `ocean`, `corrector`, `secondary_decoder`, `global_mean_removal`.

Key properties (all derive from inference_scheme for the external interface):
- `input_names` → `inference_scheme.in_names` (+ ocean forcing if configured)
- `output_names` → `inference_scheme.out_names` (+ secondary decoder names)
- `loss_names` → all non-forcing variables in `all_names` (any can be a target during training)
- `all_training_names` → `all_names` (overrides the default None)
- `allow_missing_variables` → `True` (inference datasets may lack some training-only variables)
- `n_ic_timesteps` → 1
- `_normalize_names` → `all_names` (normalizer needs stats for all variables)
- No residual prediction support

Validation in `__post_init__`:
- All inference_scheme.in_names and out_names must be in all_names
- All forcing_names must be in all_names
- forcing_names must not be in inference_scheme.out_names
- include_channel_mask_inputs must be True

### InfillPredictionStep

Implements `StepABC`. Internal packers use `all_names` for input and non-forcing names for output:
- `n_in_channels = len(all_names) * 2` (variables + channel mask indicators)
- `n_out_channels = len(non_forcing_names)` (forcing variables are input-only, never predicted)
- `in_packer = Packer(all_names)`
- `out_packer = Packer(non_forcing_names)` where `non_forcing_names = [n for n in all_names if n not in forcing_names]`

The `step()` method is nearly identical to `SingleModuleStep.step()`, reusing `_apply_input_mask`, `_build_channel_mask_dict`, and `step_with_adjustments` from `fme/core/step/single_module.py`.

**Critical detail — handling partial inputs:** During inference (and training Phase 1 via predict_generator), `args.input` may only contain inference scheme variables, not all `all_names`. The step must:
1. Fill missing `all_names` variables with zeros (in denormalized space)
2. Construct/augment `data_mask`: for variables not in `args.data_mask`, default to `True` if in `inference_scheme.in_names` and present in `args.input`, `False` otherwise

This makes the step self-contained — no changes needed to `Stepper` or `predict_generator`.

### Tests
- Config validation (valid construction, invalid configs raise ValueError)
- Config properties derive correctly from inference_scheme
- Step forward pass with full data_mask (all variables provided)
- Step forward pass with partial input (inference mode — only inference scheme vars)
- StepSelector registry: `StepSelector(type="infill_prediction", config=...)` works
- `all_training_names` returns the right value
- `StepperConfig.all_names` returns `all_training_names` when available, falls back for other step types
- Output only contains non-forcing variables

---

## Commit 3: TrainStepper integration with tests

**Modified file:** `fme/ace/stepper/single_module.py`

### TrainStepperConfig change

Add field:
```python
task_sampling: TaskSamplingConfig | None = None
```

### TrainStepper.__init__ change

When `task_sampling` is not None:
- Access the underlying step config via `stepper._step_obj.config`
- Validate it's an `InfillPredictionStepConfig` (runtime check)
- Build a `TaskSampler` from `task_sampling`, `step_config.all_names`, `step_config.forcing_names`
- Build a separate task loss object using `loss_names` from the step config (all non-forcing variables)

### Multi-task loss accumulation

The existing `_accumulate_loss` loop runs ALL N forward prediction steps unchanged. When `task_sampling` is active, additional logic runs after the prediction loop:

**After the prediction loop:**
1. Retrieve the model's state from step N-2 (saved during the loop), or IC if `n_forward_steps=1`
2. Call `self._task_sampler.sample(data_mask, batch_size)` → `SampledTasks` with per-sample masks
3. Construct task step input for all `all_names`:
   - For each variable, combine per-sample from previous-step state, current-step ground truth, or zeros using `SampledTasks.previous_step_input_mask` and `current_step_input_mask`
   - Forcing variables at the previous timestep come from `forcing_dict`, not model state (model doesn't predict forcing)
4. Compute `input_data_mask = prev_mask | curr_mask` per variable
5. Construct forcing/next_step_input at the correct timestep
6. Run `self._stepper.step(StepArgs(input=..., data_mask=input_data_mask, ...))` — one additional forward pass
7. Compute task loss using the task loss object with `output_data_mask` (float-valued, incorporates per-sample loss_scale)
8. `optimization.accumulate_loss(task_loss.total())` — adds to the prediction loss already accumulated

**When `n_forward_steps = 1`:** The prediction loop runs 1 step normally. The task step uses the IC as "previous step" data.

**Forcing routing for the task step** (same indexing as predict_generator for step N-1):
```python
step = n_forward_steps - 1
input_forcing = {
    k: (forcing_dict[k][:, step] if k not in next_step_forcing_names
         else forcing_dict[k][:, step + 1])
    for k in input_only_names
}
next_step_input_dict = {
    k: forcing_dict[k][:, step + 1]
    for k in step_obj.next_step_input_names
}
```

### Tests
- `task_sampling=None` preserves existing behavior exactly
- Training with `n_forward_steps=1` (prediction + task on same step)
- Training with `n_forward_steps=2` (prediction for both steps + task on final)
- Validate that `task_sampling` requires an `InfillPredictionStep`
- Loss includes both prediction and task components
- Per-sample task masks are correctly applied
- `output_data_mask` float values correctly weight the task loss

---

## Verification

After all commits:
1. `pre-commit run --all-files` passes (ruff, ruff-format, mypy)
2. `python -m pytest fme/core/step/test_infill_prediction.py -v` — all unit tests pass
3. `python -m pytest fme/ace/stepper/test_single_module.py -v` — existing tests unchanged
4. `make test_very_fast` — no regressions
