# Combined Infill-Prediction Training

## Motivation

ACE currently trains models for a single task: forward prediction (given state at time *t*, predict state at time *t+1*). We want to train a single model that can perform multiple tasks:

- **Auto-encoding**: reconstruct current-step variables from themselves (identity-like task).
- **Infill**: predict missing current-step variables from other current-step variables (cross-variable reconstruction).
- **Forward prediction**: predict next-step variables from previous-step variables (the existing task).
- **Combined infill-prediction**: predict some current-step variables from some previous-step and some current-step variables.
- **Combined-all**: mix of all the above — some previous-step and some current-step inputs, predicting some current-step outputs.

The model is given 2 (or more) timesteps of data during training (as today). Each batch, the TrainStepper runs normal forward prediction for all steps up to the last, then randomly selects a task for the final step — randomly choosing which variables serve as inputs vs. outputs (subject to task constraints) — and optimizes over the full trajectory. This teaches the model a flexible understanding of inter-variable and temporal relationships, enabling richer inference strategies.

At inference time, the model performs forward prediction using a configurable `InferenceSchemeConfig`, behaving similarly to today's ACE step. Future work could extend this to include post-hoc auto-encoding or iterative infill refinement.

## User Stories

1. **As a researcher**, I want to train a single model that learns both infill and prediction tasks, so that the same weights can be used for multiple downstream applications.

2. **As a researcher**, I want inference to work like the current ACE step (specified input variables, specified output variables, forward stepping), so that existing evaluation infrastructure works without changes.

3. **As a researcher**, I want the ability to later add new inference strategies (e.g., auto-encode after prediction to filter outputs) without rewriting the training or model code.

4. **As a researcher**, I want to control the relative weighting and loss scaling of tasks during training, so that I can tune the balance between prediction accuracy and infill capability.

## Design

### Overview

```
TrainConfig
├── stepper: StepperConfig
│   └── step: StepSelector → InfillPredictionStepConfig (new, registered as "infill_prediction")
│       ├── builder: ModuleSelector (e.g. NoiseConditionedSFNO)
│       ├── all_names: list[str]  — all variables the model could use
│       ├── forcing_names: list[str]  — input-only variables (e.g. solar radiation)
│       ├── normalization: NetworkAndLossNormalizationConfig
│       ├── inference_scheme: InferenceSchemeConfig (new)
│       │   ├── in_names: list[str]
│       │   ├── out_names: list[str]
│       │   ├── next_step_forcing_names: list[str]
│       │   └── prescribed_prognostic_names: list[str]
│       ├── include_channel_mask_inputs: bool = True  (required for this step type)
│       ├── corrector, ocean, global_mean_removal, secondary_decoder  (same as SingleModuleStep)
│       └── ...
└── stepper_training: TrainStepperConfig
    ├── loss, n_ensemble, n_forward_steps, parameter_init  (existing fields)
    └── task_sampling: TaskSamplingConfig | None = None  (new, optional)
```

### New Types

#### `TaskWeights`

Location: `fme/core/step/infill_prediction.py` (new file)

```python
@dataclasses.dataclass
class TaskWeights:
    """Relative sampling weights and loss scaling for each task type.

    Sampling probabilities are derived from the weight values
    (normalized to sum to 1). Loss scaling is applied as a multiplier
    on the task step's loss.

    A weight of 0 disables a task entirely.
    """
    auto_encode: float = 1.0
    infill: float = 1.0
    prediction: float = 1.0
    infill_prediction: float = 1.0
    combined_all: float = 1.0

    auto_encode_loss_scale: float = 1.0
    infill_loss_scale: float = 1.0
    prediction_loss_scale: float = 1.0
    infill_prediction_loss_scale: float = 1.0
    combined_all_loss_scale: float = 1.0
```

#### `TaskSamplingConfig`

```python
@dataclasses.dataclass
class TaskSamplingConfig:
    """Configuration for random task selection during training.

    Attributes:
        task_weights: Sampling weights and loss scaling per task type.
        min_input_variables: Minimum number of input variables to select.
        min_output_variables: Minimum number of output variables to select.
    """
    task_weights: TaskWeights = dataclasses.field(default_factory=TaskWeights)
    min_input_variables: int = 1
    min_output_variables: int = 1
```

Each task type imposes constraints on which variables can be inputs vs. outputs and which timesteps they come from. Variables in `forcing_names` can only appear as inputs (never as prediction targets).

| Task | Input timestep(s) | Output timestep | Constraint |
|------|-------------------|-----------------|------------|
| auto_encode | t | t | inputs ⊇ outputs; outputs are a subset of inputs; outputs exclude forcing_names |
| infill | t | t | inputs ∩ outputs = ∅; outputs exclude forcing_names |
| prediction | t-1 | t | outputs exclude forcing_names |
| infill_prediction | t-1 and t | t | at least one input from each of t-1 and t; outputs only from t; current-step inputs ∩ outputs = ∅; outputs exclude forcing_names |
| combined_all | t-1 and/or t | t | at least one input, at least one output; outputs exclude forcing_names |

#### `InferenceSchemeConfig`

```python
@dataclasses.dataclass
class InferenceSchemeConfig:
    """Defines how the model behaves at inference time.

    This mirrors the variable routing of SingleModuleStepConfig for
    standard forward prediction. Future schemes could add post-hoc
    auto-encoding or iterative infill.

    Attributes:
        in_names: Input variable names for inference.
        out_names: Output variable names for inference.
        next_step_forcing_names: Input variables that come from the output timestep.
        prescribed_prognostic_names: Prognostic variables overwritten from forcing.
    """
    in_names: list[str]
    out_names: list[str]
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    prescribed_prognostic_names: list[str] = dataclasses.field(default_factory=list)
```

#### `TrainingTask`

A runtime object (not a config) representing a sampled task for a single batch:

```python
@dataclasses.dataclass
class TrainingTask:
    """A concrete task assignment for one training batch.

    Attributes:
        previous_step_input_names: Variables from the previous timestep to
            provide as (unmasked) inputs. Empty for auto_encode and infill tasks.
        current_step_input_names: Variables from the current timestep to
            provide as (unmasked) inputs.
        output_names: Variables to predict (always at the current timestep).
        loss_scale: Multiplicative scaling factor for this task's loss.
    """
    previous_step_input_names: list[str]
    current_step_input_names: list[str]
    output_names: list[str]
    loss_scale: float
```

For example, in an `infill_prediction` task:
- `previous_step_input_names`: `["air_temperature", "surface_pressure"]`
- `current_step_input_names`: `["specific_humidity"]`
- `output_names`: `["air_temperature"]`
- `loss_scale`: 1.0

This means the model receives air_temperature(t-1), surface_pressure(t-1), and specific_humidity(t) as unmasked inputs, and must predict air_temperature(t).

#### `TaskSampler`

Built from `TaskSamplingConfig` + `all_names` + `forcing_names`. Implements the random selection logic:

```python
class TaskSampler:
    def __init__(
        self,
        config: TaskSamplingConfig,
        all_names: list[str],
        forcing_names: list[str],
    ): ...

    def sample(self) -> TrainingTask:
        """Sample a task type, then sample variable assignments for that task."""
        ...
```

The sampler first draws a task type from the weighted distribution, then randomly selects which variables serve as inputs and outputs subject to the task's constraints. The number of input and output variables is drawn uniformly between the configured minimum and the maximum allowed by the constraint (all non-forcing variables for outputs, all variables for inputs).

### `InfillPredictionStepConfig` and `InfillPredictionStep`

Location: `fme/core/step/infill_prediction.py`

Registered as `@StepSelector.register("infill_prediction")`.

**Config** (`InfillPredictionStepConfig`):
- Inherits from `StepConfigABC`.
- Contains `all_names`, `forcing_names`, `inference_scheme`, plus the same builder/normalization/corrector/ocean fields as `SingleModuleStepConfig`.
- Contains `task_sampling: TaskSamplingConfig` for building a `TaskSampler` (used by the TrainStepper, not by the step itself).
- The `StepConfigABC` interface properties (`input_names`, `output_names`, `prognostic_names`, `next_step_input_names`, etc.) are derived from `inference_scheme` — they describe the inference-time behavior, not the training-time behavior. This is critical because the `Stepper` and inference pipeline use these properties to route variables.
- `n_ic_timesteps` returns 1 (same as SingleModuleStep).
- Residual prediction is not supported (omitted from config).
- Validation in `__post_init__`:
  - All `inference_scheme.in_names` and `inference_scheme.out_names` must be in `all_names`.
  - All `forcing_names` must be in `all_names`.
  - `forcing_names` must not appear in `inference_scheme.out_names`.
  - `include_channel_mask_inputs` must be True.

**Step** (`InfillPredictionStep`):
- Implements `StepABC`.
- Internally, packers use `all_names` for both input and output channels:
  - `n_in_channels = len(all_names) * 2` (variables + mask indicators, since `include_channel_mask_inputs` is always True).
  - `n_out_channels = len(all_names)`.
- The `step()` method:
  1. Starts with a zero tensor dict for all `all_names` (in normalized space).
  2. Overwrites entries for variables present in `args.input`.
  3. Applies `data_mask` (zeros masked variables, builds channel mask indicators).
  4. Packs, runs network, unpacks all `all_names`.
  5. Applies post-processing (corrector, ocean, prescribed prognostics) — these use the inference_scheme's variable routing and are applied if configured.
  6. Returns the full output dict.
- Has a `build_task_sampler()` method that returns a `TaskSampler` for the TrainStepper to use.

Key insight: The `step()` method doesn't need to know about tasks. The task determines (1) which variables to put in `StepArgs.input` vs. mask via `data_mask`, and (2) which variables to include in the loss. The `step()` method just does the forward pass with whatever masking is provided. This means `StepABC` doesn't need changes.

### Training: Extending the Existing `TrainStepper`

Location: modifications to `fme/ace/stepper/single_module.py`

Rather than creating a new `TrainStepperABC` implementation, we extend the existing `TrainStepperConfig` and `TrainStepper` with an optional `task_sampling` field. When `task_sampling` is `None`, behavior is identical to today. When set, the training loop is modified for the final step.

**`TrainStepperConfig` changes:**
```python
@dataclasses.dataclass
class TrainStepperConfig:
    loss: StepLossConfig = ...
    optimize_last_step_only: bool = False
    n_ensemble: int = -1
    n_forward_steps: TimeLength | TimeLengthSchedule | None = None
    parameter_init: ParameterInitializationConfig = ...
    task_sampling: TaskSamplingConfig | None = None  # NEW
```

**`TrainStepper` changes:**
- In `__init__`, if `task_sampling` is configured, build a `TaskSampler` from the underlying step's config. The step must be an `InfillPredictionStep` (validated at init time).
- In `_accumulate_loss`, when task sampling is active, the method branches to `_accumulate_loss_with_task`.

**`_accumulate_loss_with_task` — two-phase training loop:**

Phase 1 — Normal forward prediction (steps 0 to N-2):
- Uses the existing `predict_generator` for `N-1` forward steps with normal data_mask.
- Computes loss at each step on all output variables (same as current behavior).
- The model learns standard forward prediction for intermediate steps.

Phase 2 — Task step (step N-1):
1. Sample a `TrainingTask` from the `TaskSampler`.
2. Extract state from the last predict_generator output (the model's prediction at step N-2). This state contains prognostic variables that serve as "previous step" data.
3. Construct the task step input:
   - For variables in `task.previous_step_input_names`: take from the model's state.
   - For variables in `task.current_step_input_names`: take from ground truth data at the output timestep.
   - For all other `all_names`: provide as zeros (masked).
4. Construct `input_data_mask`: True for variables in `task.previous_step_input_names ∪ task.current_step_input_names`, False otherwise.
5. Gather `next_step_input_data` from forcing data at the output timestep (same as predict_generator would for this step).
6. Call `self._stepper.step(StepArgs(input=input_dict, next_step_input_data=..., labels=..., data_mask=input_data_mask))`.
7. Construct `loss_data_mask`: True for variables in `task.output_names`, False otherwise. This ensures the loss function only penalizes the task's target variables.
8. Compute loss with `loss_data_mask` and scale by `task.loss_scale`.

When `n_forward_steps = 1`, Phase 1 is skipped entirely — the single step is the task step.

**Forcing routing for the task step**: The task step is at step index `N-1` in the forward loop. We need to construct `input_forcing` and `next_step_input_dict` the same way predict_generator would. Since predict_generator's logic is simple (index into forcing_dict by step), we replicate it for this single step:

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

### Inference: Unchanged Pipeline

At inference time, `InfillPredictionStep` exposes the same `StepABC` interface as `SingleModuleStep`, with variable names derived from `InferenceSchemeConfig`. The `Stepper` and inference pipeline see no difference — they call `step()` the same way, route variables the same way, and get denormalized outputs the same way.

The only difference from the model's perspective is that at inference time, `data_mask` will indicate all inference input variables are present (no masking), and the channel mask indicators will all be 1.0 for inference inputs and 0.0 for variables not in the inference input set. The model was trained to handle partial-input scenarios, so receiving a subset of variables is just one such scenario.

### How the Existing `data_mask` / `include_channel_mask_inputs` Mechanism Fits

The existing masking infrastructure was designed for datasets with missing variables. It does exactly what we need:

1. `_apply_input_mask`: Zeros masked variables in normalized space (equivalent to replacing with climatological mean).
2. `_build_channel_mask_dict` + `include_channel_mask_inputs`: Appends per-variable presence indicators (0.0/1.0) so the network knows which inputs are real vs. masked.
3. `WeightedMappingLoss` with `data_mask`: Excludes masked variables from loss computation. The `total()` method averages only over active channels, so masked-out variables never affect the optimization target.

For infill training, we construct `data_mask` from the sampled `TrainingTask` rather than from dataset availability. The mechanism is identical.

`include_channel_mask_inputs` is required (always True) for `InfillPredictionStepConfig`, since the network must know which inputs are present to perform infill correctly.

### Network Architecture

The `NoiseConditionedSFNO` (or any `ModuleConfig`) doesn't need changes. It receives:
- `n_in_channels = len(all_names) * 2` (variables + mask indicators)
- `n_out_channels = len(all_names)` (all variables predicted, loss applied selectively)

The model always predicts all variables, but loss is only computed on the task's output variables. This is simpler than dynamically changing the output size per task and allows the model to learn cross-variable relationships even when a variable isn't in the task's explicit output set.

### File Organization

```
fme/core/step/
├── infill_prediction.py  (new)
│   ├── TaskWeights
│   ├── TaskSamplingConfig
│   ├── TaskSampler
│   ├── TrainingTask
│   ├── InferenceSchemeConfig
│   ├── InfillPredictionStepConfig  (implements StepConfigABC)
│   └── InfillPredictionStep  (implements StepABC)
└── ...

fme/ace/stepper/
├── single_module.py  (modified — TrainStepperConfig and TrainStepper extended)
└── ...
```

### Config Example (YAML)

```yaml
stepper:
  step:
    type: infill_prediction
    config:
      builder:
        type: NoiseConditionedSFNO
        config:
          embed_dim: 256
          num_layers: 12
          noise_type: gaussian
      all_names:
        - air_temperature
        - specific_humidity
        - surface_pressure
        - TOA_incident_shortwave_radiation
        # ... all variables
      forcing_names:
        - TOA_incident_shortwave_radiation
      normalization:
        network: ...
        loss: ...
      task_sampling:
        task_weights:
          auto_encode: 1.0
          infill: 1.0
          prediction: 1.0
          infill_prediction: 1.0
          combined_all: 1.0
          auto_encode_loss_scale: 0.5
          infill_loss_scale: 1.0
          prediction_loss_scale: 1.0
          infill_prediction_loss_scale: 1.0
          combined_all_loss_scale: 1.0
        min_input_variables: 1
        min_output_variables: 1
      inference_scheme:
        in_names:
          - air_temperature
          - specific_humidity
          - surface_pressure
          - TOA_incident_shortwave_radiation
        out_names:
          - air_temperature
          - specific_humidity
          - surface_pressure
        next_step_forcing_names:
          - TOA_incident_shortwave_radiation
      include_channel_mask_inputs: true
  input_masking: null
  derived_forcings: {}
stepper_training:
  loss:
    type: MSE
  n_forward_steps: 2
  task_sampling:
    task_weights:
      auto_encode: 1.0
      infill: 1.0
      prediction: 1.0
      infill_prediction: 1.0
      combined_all: 1.0
    min_input_variables: 1
    min_output_variables: 1
  parameter_init: {}
```

Note: `task_sampling` appears in `stepper_training` (where it controls the training loop behavior), while `task_sampling` in the step config provides the default config that gets passed up. The `TrainStepperConfig.task_sampling` field is what actually drives the behavior — if it's `None`, no task sampling occurs regardless of what's in the step config.

## Open Questions

1. **`InfillPredictionStep.step()` handling of missing input variables**: The step receives only a subset of `all_names` in `args.input` (either inference inputs, or task-selected inputs). It needs to fill in zeros for the rest before packing. Two approaches: (a) fill in zeros in normalized space for missing variables inside `step()`, or (b) require the caller to always provide all `all_names` with masked variables already zeroed. Option (a) is more encapsulated; option (b) is more explicit. Leaning toward (a) since it makes `step()` robust to partial inputs from any caller.

2. **Where `TaskSamplingConfig` lives**: The config is needed by both the step (to validate variable names, build the sampler) and the `TrainStepperConfig` (to control training behavior). Should it be defined once in the step config and referenced by `TrainStepperConfig`? Or duplicated? The YAML example above shows it in `stepper_training`, which is the cleanest since task sampling is a training concern, not an inference concern. The step config can omit it and just provide `all_names` and `forcing_names` for the sampler to use.

## Follow-on Work

- **Extended inference schemes**: Implement `InferenceSchemeConfig` variants that call the model multiple times per step (e.g., predict then auto-encode to filter, or iterative infill refinement).
- **Task curricula**: Schedule task weights over training epochs (e.g., start with auto-encoding, gradually increase prediction weight).
- **Variable group constraints**: Allow specifying groups of variables that should be masked/unmasked together (e.g., all pressure levels of temperature).
- **Evaluation per task**: Add per-task evaluation metrics during training to monitor how well the model learns each task independently.
