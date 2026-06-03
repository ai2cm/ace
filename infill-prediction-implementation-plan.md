# Implementation Plan: Combined Infill-Prediction Training

## Context

ACE currently trains models for a single task: forward prediction. The design in `infill-prediction-design.md` specifies a new step type (`InfillPredictionStep`) that trains on multiple tasks per batch (auto-encoding, infill, prediction, infill-prediction, combined-all) by randomly masking variables. At inference time, it behaves like the existing `SingleModuleStep`.

This plan breaks the implementation into 3 independently reviewable, tested commits.

---

## Commit 1: Config dataclasses + TaskSampler with tests

**New file:** `fme/core/step/infill_prediction.py`

### Types to add

- **`TaskWeights`** — dataclass with sampling weights and loss scales per task type (10 float fields)
- **`TaskSamplingConfig`** — wraps `TaskWeights` + `min_input_variables: int` + `min_output_variables: int`
- **`TrainingTask`** — runtime dataclass: `previous_step_input_names`, `current_step_input_names`, `output_names`, `loss_scale`
- **`InferenceSchemeConfig`** — dataclass: `in_names`, `out_names`, `next_step_forcing_names`, `prescribed_prognostic_names`

### TaskSampler class

```python
class TaskSampler:
    def __init__(self, config, all_names, forcing_names): ...
    def sample(self, data_mask=None) -> TrainingTask: ...
```

Key implementation details:
- Pre-compute which tasks are feasible given variable counts and constraints; raise `ValueError` in `__init__` if a task with non-zero weight is infeasible
- `sample()` filters out variables absent in data_mask (all-False), draws a task type from the weighted distribution, then samples variable assignments per the task's constraints
- Variable count drawn uniformly between min and max allowed
- Task constraint enforcement per the design doc table (see below)

**Task constraints to enforce:**

| Task | Previous-step inputs | Current-step inputs | Outputs | Key constraint |
|------|---------------------|--------------------|---------|----|
| auto_encode | (none) | subset of all_names | subset of current inputs; excludes forcing | inputs ⊇ outputs |
| infill | (none) | from all_names | from non-forcing; disjoint from current inputs | inputs ∩ outputs = ∅ |
| prediction | from all_names | (none) | from non-forcing | standard forward prediction |
| infill_prediction | ≥1 from all_names | ≥1 from all_names | from non-forcing; disjoint from current inputs | mixed timestep |
| combined_all | from all_names | from all_names | from non-forcing | at least 1 input, 1 output |

### Tests (`fme/core/step/test_infill_prediction.py`)
- TaskSampler produces valid TrainingTask for each task type
- Task constraints are satisfied (e.g., auto_encode inputs ⊇ outputs)
- data_mask filtering excludes absent variables
- Zero-weight tasks never sampled
- min_input/output_variables respected
- Infeasible tasks raise ValueError in constructor
- InferenceSchemeConfig and TrainingTask construction

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

Implements `StepABC`. Internal packers use `all_names`:
- `n_in_channels = len(all_names) * 2` (variables + channel mask indicators)
- `n_out_channels = len(all_names)`

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

Also build a separate loss object using all training names (loss_names from the step config, which includes all non-forcing variables).

### _accumulate_loss branching

In `_accumulate_loss`, when `self._task_sampler` is set, delegate to `_accumulate_loss_with_task`.

### _accumulate_loss_with_task — two-phase training loop

**Phase 1 — Normal forward prediction (steps 0 to N-2):**
- Use existing `predict_generator` for `n_forward_steps - 1` steps
- Compute loss at each step on inference scheme output variables (same as current behavior)
- Keep the last yielded state (contains all `all_names` from step output)

**Phase 2 — Task step (step N-1):**
1. Sample a `TrainingTask` from the `TaskSampler`, passing the batch's `data_mask`
2. Get "previous step" data from Phase 1's last output state (or IC if `n_forward_steps=1`)
3. Construct input dict with all `all_names`:
   - `task.previous_step_input_names`: from model state
   - `task.current_step_input_names`: from ground truth at output timestep
   - Others: fill with zeros (will be masked)
4. Construct `input_data_mask`: True for task inputs, False for others
5. Construct forcing/next_step_input at the correct timestep (replicate predict_generator's forcing routing for step N-1)
6. Call `self._stepper.step(StepArgs(input=..., next_step_input_data=..., labels=..., data_mask=input_data_mask))`
7. Construct `loss_data_mask`: True only for `task.output_names`
8. Compute loss with `loss_data_mask`, scale by `task.loss_scale`

**When `n_forward_steps = 1`:** Phase 1 is skipped. The single step IS the task step. "Previous step" data comes from the initial condition.

**Forcing routing for the task step** (replicates predict_generator logic for step index N-1):
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
- `task_sampling=None` preserves existing behavior
- Training with `n_forward_steps=1` (task step only)
- Training with `n_forward_steps=2` (Phase 1 + Phase 2)
- Validate that `task_sampling` requires an `InfillPredictionStep`
- Loss is computed and backpropagated correctly
- `loss_data_mask` correctly restricts loss to task output variables

---

## Verification

After all commits:
1. `pre-commit run --all-files` passes (ruff, ruff-format, mypy)
2. `python -m pytest fme/core/step/test_infill_prediction.py -v` — all unit tests pass
3. `python -m pytest fme/ace/stepper/test_single_module.py -v` — existing tests unchanged
4. `make test_very_fast` — no regressions
