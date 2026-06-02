# Combined Infill-Prediction Training

## Motivation

ACE currently trains models for a single task: forward prediction (given state at time *t*, predict state at time *t+1*). We want to train a single model that can perform multiple tasks:

- **Auto-encoding**: reconstruct current-step variables from themselves (identity-like task).
- **Infill**: predict missing current-step variables from other current-step variables (cross-variable reconstruction).
- **Forward prediction**: predict next-step variables from previous-step variables (the existing task).
- **Combined infill-prediction**: predict some current-step variables from some previous-step and some current-step variables.
- **Combined-all**: mix of all the above — some previous-step and some current-step inputs, predicting some current-step outputs.

The model is given 2 timesteps of data during training (as today). Each batch, the TrainStepper randomly selects a task, randomly selects which variables serve as inputs vs. outputs (subject to task constraints), and optimizes over that batch. This teaches the model a flexible understanding of inter-variable and temporal relationships, enabling richer inference strategies.

At inference time, the model performs forward prediction using a configurable `StepScheme`, behaving similarly to today's ACE step. Future work could extend `StepScheme` to include post-hoc auto-encoding or iterative infill refinement.

## User Stories

1. **As a researcher**, I want to train a single model that learns both infill and prediction tasks, so that the same weights can be used for multiple downstream applications.

2. **As a researcher**, I want inference to work like the current ACE step (specified input variables, specified output variables, forward stepping), so that existing evaluation infrastructure works without changes.

3. **As a researcher**, I want the ability to later add new inference strategies (e.g., auto-encode after prediction to filter outputs) without rewriting the training or model code.

4. **As a researcher**, I want to control the relative weighting of tasks during training, so that I can tune the balance between prediction accuracy and infill capability.

## Design

### Overview

```
TrainConfig
├── stepper: StepperConfig
│   └── step: StepSelector → InfillPredictionStepConfig (new, registered as "infill_prediction")
│       ├── builder: ModuleSelector (e.g. NoiseConditionedSFNO)
│       ├── all_names: list[str]  — all variables the model could use
│       ├── normalization: NetworkAndLossNormalizationConfig
│       ├── task_config: TaskSamplingConfig (new)
│       ├── inference_scheme: InferenceSchemeConfig (new)
│       │   ├── in_names: list[str]
│       │   ├── out_names: list[str]
│       │   ├── next_step_forcing_names: list[str]
│       │   └── prescribed_prognostic_names: list[str]
│       ├── include_channel_mask_inputs: bool = True  (required for this step type)
│       ├── residual_prediction: bool = False
│       ├── corrector, ocean, global_mean_removal, secondary_decoder  (same as SingleModuleStep)
│       └── ...
└── stepper_training: TrainStepperConfig
    └── (unchanged — same loss, optimizer, n_forward_steps config)
```

### New Types

#### `TaskSamplingConfig`

Location: `fme/core/step/infill_prediction.py` (new file)

```python
@dataclasses.dataclass
class TaskSamplingConfig:
    """Configuration for random task selection during training.

    Attributes:
        task_weights: Relative weights for each task type.
            Keys: "auto_encode", "infill", "prediction", "infill_prediction", "combined_all"
            Default: equal weights for all five tasks.
        min_input_variables: Minimum number of input variables to select (default: 1).
        min_output_variables: Minimum number of output variables to select (default: 1).
    """
    task_weights: dict[str, float]
    min_input_variables: int = 1
    min_output_variables: int = 1
```

Each task type imposes constraints on which variables can be inputs vs. outputs and which timesteps they come from:

| Task | Input timestep(s) | Output timestep | Constraint |
|------|-------------------|-----------------|------------|
| auto_encode | t | t | inputs ⊇ outputs (outputs are a subset of inputs) |
| infill | t | t | inputs ∩ outputs = ∅ |
| prediction | t-1 | t | no same-timestep overlap required |
| infill_prediction | t-1 and t | t | at least one input from each of t-1 and t; outputs only from t; current-step inputs ∩ outputs = ∅ |
| combined_all | t-1 and/or t | t | union of the above; at least one input, at least one output |

#### `InferenceSchemeConfig`

Location: `fme/core/step/infill_prediction.py`

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
        input_names: Variables to provide as (unmasked) inputs.
        output_names: Variables to predict.
        use_previous_timestep: Whether the input includes previous-timestep data
            (True for prediction/infill_prediction/combined_all tasks).
    """
    input_names: list[str]
    output_names: list[str]
    use_previous_timestep: bool
```

#### `TaskSampler`

Built from `TaskSamplingConfig` + the list of `all_names`. Implements the random selection logic:

```python
class TaskSampler:
    def __init__(self, config: TaskSamplingConfig, all_names: list[str]): ...

    def sample(self) -> TrainingTask:
        """Sample a task, then sample variable assignments for that task."""
        ...
```

### `InfillPredictionStepConfig` and `InfillPredictionStep`

Location: `fme/core/step/infill_prediction.py`

Registered as `@StepSelector.register("infill_prediction")`.

**Config** (`InfillPredictionStepConfig`):
- Inherits from `StepConfigABC`.
- Contains `all_names`, `task_config`, `inference_scheme`, plus the same builder/normalization/corrector/ocean fields as `SingleModuleStepConfig`.
- The `StepConfigABC` interface properties (`input_names`, `output_names`, `prognostic_names`, `next_step_input_names`, etc.) are derived from `inference_scheme` — they describe the inference-time behavior, not the training-time behavior. This is critical because the `Stepper` and inference pipeline use these properties to route variables.
- `n_ic_timesteps` returns 1 (same as SingleModuleStep).

**Step** (`InfillPredictionStep`):
- Implements `StepABC`.
- The `step()` method works identically to `SingleModuleStep.step()` — it takes `StepArgs` with `data_mask` and produces denormalized outputs. The masking mechanism (`_apply_input_mask` + `_build_channel_mask_dict`) is exactly what we need: masked input variables are zeroed in normalized space, and channel mask indicators tell the network which inputs are present.
- Has a `sample_task()` method (or exposes the `TaskSampler`) for training to call. This is **not** part of the `StepABC` interface.

Key insight: The `step()` method itself doesn't need to know about tasks. The task determines (1) which variables to put in `StepArgs.input` vs. mask via `data_mask`, and (2) which variables to include in the loss. The `step()` method just does the forward pass with whatever masking is provided. This means `StepABC` doesn't need changes.

### Training: New `InfillPredictionTrainStepper`

Location: `fme/ace/stepper/infill_prediction.py` (new file)

This is a new `TrainStepperABC` implementation, paired with a new `InfillPredictionTrainStepperConfig`:

```python
@dataclasses.dataclass
class InfillPredictionTrainStepperConfig:
    loss: StepLossConfig = ...
    n_ensemble: int = ...
    parameter_init: ParameterInitializationConfig = ...

    def get_train_stepper(
        self, stepper_config: StepperConfig, dataset_info: DatasetInfo, ...
    ) -> "InfillPredictionTrainStepper": ...
```

**Why a new TrainStepper?** The existing `TrainStepper` always runs `predict_generator` (multi-step forward prediction) and computes loss on all output variables. For infill training, we need fundamentally different behavior per batch:

1. Sample a `TrainingTask` from the step's `TaskSampler`.
2. Construct `data_mask` based on the task (mask out variables not in `task.input_names`).
3. Run the model for either 0 or 1 forward steps depending on `task.use_previous_timestep`.
4. Compute loss only on `task.output_names` (not all outputs).

The `_accumulate_loss` method is the core difference. For tasks that don't use the previous timestep (auto-encode, infill), the "input" and "target" come from the same timestep. For prediction tasks, they come from consecutive timesteps as today.

**Loss masking**: The existing `data_mask` mechanism in `WeightedMappingLoss` already handles per-variable, per-sample loss masking. We construct a `data_mask` where only `task.output_names` are `True`, and the loss computation automatically ignores the other variables. This works because the loss function's mask means "only compute loss for these variables."

### Inference: Unchanged Pipeline

At inference time, `InfillPredictionStep` exposes the same `StepABC` interface as `SingleModuleStep`, with variable names derived from `InferenceSchemeConfig`. The `Stepper` and inference pipeline see no difference — they call `step()` the same way, route variables the same way, and get denormalized outputs the same way.

The only difference from the model's perspective is that at inference time, `data_mask` will indicate all inference input variables are present (no masking), and the channel mask indicators will all be 1.0. The model was trained to handle this case (it's one possible outcome of the random task sampling).

### How the Existing `data_mask` / `include_channel_mask_inputs` Mechanism Fits

The existing masking infrastructure was designed for datasets with missing variables. It does exactly what we need:

1. `_apply_input_mask`: Zeros masked variables in normalized space (equivalent to replacing with climatological mean).
2. `_build_channel_mask_dict` + `include_channel_mask_inputs`: Appends per-variable presence indicators (0.0/1.0) so the network knows which inputs are real vs. masked.
3. `WeightedMappingLoss` with `data_mask`: Excludes masked variables from loss computation.

For infill training, we construct `data_mask` from the sampled `TrainingTask` rather than from dataset availability. The mechanism is identical.

`include_channel_mask_inputs` should default to `True` (or be required) for `InfillPredictionStepConfig`, since the network must know which variables are present to perform infill correctly.

### Network Architecture

The `NoiseConditionedSFNO` (or any `ModuleConfig`) doesn't need changes. It receives:
- `n_in_channels = len(all_names) * 2` (variables + mask indicators)
- `n_out_channels = len(all_names)` (all variables predicted, loss applied selectively)

The model always predicts all variables, but loss is only computed on the task's output variables. This is simpler than dynamically changing the output size per task and allows the model to learn cross-variable relationships even when a variable isn't in the task's explicit output set.

### File Organization

```
fme/core/step/
├── infill_prediction.py  (new)
│   ├── TaskSamplingConfig
│   ├── TaskSampler
│   ├── TrainingTask
│   ├── InferenceSchemeConfig
│   ├── InfillPredictionStepConfig  (implements StepConfigABC)
│   └── InfillPredictionStep  (implements StepABC)
└── ...

fme/ace/stepper/
├── infill_prediction.py  (new)
│   ├── InfillPredictionTrainStepperConfig
│   └── InfillPredictionTrainStepper  (implements TrainStepperABC)
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
      normalization:
        network: ...
        loss: ...
      task_config:
        task_weights:
          auto_encode: 1.0
          infill: 1.0
          prediction: 1.0
          infill_prediction: 1.0
          combined_all: 1.0
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
      residual_prediction: false
  input_masking: null
  derived_forcings: {}
stepper_training:
  loss:
    type: MSE
  parameter_init: {}
```

## Open Questions

1. **Variable subsets for tasks**: Should we distinguish "prognostic-only" variables (those that can serve as both input and output in prediction) from "forcing-only" variables (input-only, like solar radiation) when sampling tasks? Forcing variables arguably shouldn't be prediction targets. One option: add `forcing_names` to the config so the sampler knows which variables are input-only.

2. **Residual prediction with infill**: Residual prediction (network predicts tendency, added to input) makes sense for forward prediction but is ill-defined for infill when the output variable isn't in the input. Should we disable residual connections for infill tasks, or only apply them to variables that are in both input and output?

3. **Loss weighting across tasks**: Should different tasks use different loss weights? For example, auto-encoding is an "easier" task — should it have lower weight? The current design applies the same `StepLossConfig` weights to all tasks.

4. **Number of forward steps**: For prediction tasks, should we support multi-step rollouts like the existing TrainStepper, or limit to single-step? Multi-step adds complexity since each step would need its own task sample (or the same task applied repeatedly). Starting with single-step seems safest.

5. **`n_forward_steps` for data requirements**: The existing infrastructure uses `n_forward_steps` to determine how much data to load per sample. For infill-only tasks, we only need 1 timestep. For prediction, we need 2. We should always request 2 timesteps (since the task is sampled at runtime), but this is worth documenting clearly.

6. **Corrector/ocean at training time for infill**: The corrector and ocean model are post-processing steps designed for forward prediction. Should they be applied during infill training? Probably not — they enforce physical constraints that may not apply to same-timestep reconstruction. We may want to skip them for non-prediction tasks.

7. **How to handle the `TrainConfig.stepper_training` type**: Currently `TrainConfig` has a single `stepper_training: TrainStepperConfig` field. We need `InfillPredictionTrainStepperConfig` instead. Options: (a) make `stepper_training` a union type, (b) add a registry/selector pattern like `StepSelector`, (c) embed the task sampling config in the step config and keep using the existing `TrainStepperConfig` with a modified `_accumulate_loss`. Option (c) is simplest if we can make it work.

## Follow-on Work

- **Extended inference schemes**: Implement `StepScheme` variants that call the model multiple times per step (e.g., predict then auto-encode to filter, or iterative infill refinement).
- **Task curricula**: Schedule task weights over training epochs (e.g., start with auto-encoding, gradually increase prediction weight).
- **Variable group constraints**: Allow specifying groups of variables that should be masked/unmasked together (e.g., all pressure levels of temperature).
- **Evaluation per task**: Add per-task evaluation metrics during training to monitor how well the model learns each task independently.
