<!-- Planning artifact for early review. Deleted before merge. -->

# Add normalized correction metrics to the inference aggregators

Corrector-equipped runs now log how much the model leans on its corrector: normalized-space
correction-magnitude metrics, computed from the `StepDiagnostics.delta` carried on prediction
data (#1335), merged into the existing `time_mean_norm` / `mean_norm` metric groups in both the
evaluator and no-target inference aggregators. Metrics are uniform over all corrected variables
(no per-variable category filtering).

The wiring mirrors the `StepDiagnostics` container rather than its current contents: each host
aggregator carries a **single optional `StepDiagnosticsAggregator`** — a concrete wrapper
that owns the dict of per-concern sub-aggregators (one entry today — the correction-delta
aggregator) and fans record/log/flush calls out over it. Future `StepDiagnostics` fields get
sibling entries inside the wrapper, not more host surgery, and the hosts never do
combine-the-aggregators bookkeeping. Granular sub-aggregators for specific container fields
(just `delta`, for now) live another level down and are equally invisible to the hosts.

---

## `fme/ace/aggregator/inference/step_diagnostics.py` (new)

```python
class StepDiagnosticsSubAggregator(Protocol):
    """A per-concern aggregator of metrics from the StepDiagnostics carried
    on prediction data. Entries in the StepDiagnosticsAggregator's dict
    satisfy this; the wrapper delegates record/log/flush calls without
    knowing what any entry measures."""

    def record_batch(
        self,
        prediction: TensorMapping,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None: ...
    def summary_logs(self) -> dict[str, Any]: ...
    def time_series_logs(self, step_slice: slice) -> dict[str, Any]: ...
    def diagnostics(self) -> dict[str, xr.Dataset]: ...


@dataclasses.dataclass
class StepDiagnosticsMetricConfig:
    """Granularity of StepDiagnostics-derived metrics (correction deltas
    only, for now). Flat booleans; nest per-field configs if StepDiagnostics
    grows more fields."""

    correction_scalars: bool = True   # scalar metrics on by default
    correction_maps: bool = False     # map images opt-in

    def build(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None,
        enable_time_series: bool,
        normalize: NormalizeFn | None,
    ) -> StepDiagnosticsAggregator | None:
        # StepDiagnosticsAggregator(
        #     {"correction": CorrectionDeltaAggregator(...)}) — the key names
        # the flush_diagnostics files. None when normalize is None (the
        # correction metrics are normalized-space; callers that supply no
        # normalizer keep current behavior) or when both flags are off.
        ...


class StepDiagnosticsAggregator:
    """The one aggregator the hosts see: wraps the dict of per-concern
    sub-aggregators (one entry today) and fans record_batch / summary_logs /
    time_series_logs / diagnostics out over it, merging the results. Owns
    the combine-the-sub-aggregators logic so the hosts don't."""

    def __init__(self, aggregators: dict[str, StepDiagnosticsSubAggregator]): ...


class CorrectionDeltaAggregator:
    """StepDiagnosticsSubAggregator for the correction delta: computes
    the normalized correction once per batch as
    normalize(delta, apply_mean=False) — i.e. delta / std (#1353) — and
    dispatches to the granular sub-aggregators below, which the hosts never
    see. The mean subtraction must be skipped because the delta is a
    difference quantity: centering would wrongly subtract the per-variable
    mean offset. Off-mask cells stay NaN, consistent with the masked
    prediction. Raises ValueError if a delta key is dropped by the
    normalizer (the StandardNormalizer silently filters to known names; a
    silently vanishing metric would be worse than a loud failure).

    Not a registered sub-aggregator of the hosts' one-name-per-aggregator
    registry because (a) the metrics consume data.step_diagnostics, which
    InferenceBatchData does not carry, and (b) the logs merge into the
    existing time_mean_norm / mean_norm label groups, which the registry
    cannot express."""

    TIME_MEAN_LABEL = "time_mean_norm"
    TIME_SERIES_LABEL = "mean_norm"

    def __init__(
        self,
        normalize: NormalizeFn,
        time_mean: CorrectionDeltaTimeMeanAggregator | None,
        time_series: CorrectionDeltaMeanAggregator | None,
    ): ...

    def record_batch(
        self,
        prediction: TensorMapping,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None:
        # Silent no-op when step_diagnostics is None (no corrector ran, or
        # coupled inference, which never attaches diagnostics).
        # Raises ValueError if the delta's time dim does not match the
        # prediction window's — the alignment is guaranteed by construction
        # in Stepper.predict, but asserted here rather than trusted.
        ...

    def summary_logs(self) -> dict[str, Any]: ...          # time_mean under "time_mean_norm"
    def time_series_logs(self, step_slice: slice) -> dict[str, Any]: ...  # under "mean_norm"
    def diagnostics(self) -> dict[str, xr.Dataset]:
        # {"time_mean_norm_correction": ..., "mean_norm_correction": ...} —
        # extra entries for flush_diagnostics; suffixed names because the
        # plain "time_mean_norm"/"mean_norm" diagnostics files belong to the
        # existing aggregators.
        ...


class CorrectionDeltaTimeMeanAggregator:
    """Granular sub-aggregator: time-mean maps and scalars of the normalized
    correction. Silent (empty logs/dataset) until non-empty data is recorded.
    Map images are built only when enabled (correction_maps flag)."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        record_maps: bool = False,
    ): ...

    def record_batch(self, correction_norm: TensorMapping) -> None: ...

    def get_logs(self, label: str) -> dict[str, float | Image]:
        # correction_magnitude/{name}: area-weighted global mean of the
        #   time-mean |normalized correction| (mask-aware: reduction goes
        #   through ops.area_weighted_mean(..., name=name), so NaN off-mask
        #   cells drop out of the weights). No channel_mean — its
        #   interpretation shifts with whichever correctors are enabled.
        # correction_map/{name}: signed time-mean map image (diverging
        #   palette), normalized space; only when record_maps. Denormalized
        #   per-cell deltas are the step-diagnostics writer's job.
        ...

    def get_dataset(self) -> xr.Dataset:
        # correction_map-{name}, dims ("lat", "lon") — signed time-mean map
        # (only when record_maps)
        ...


class CorrectionDeltaMeanAggregator:
    """Granular sub-aggregator: per-forecast-step area-weighted series of the
    normalized correction, mirroring the mean_norm MeanAggregator. Built only
    when time series are enabled (inline training drops it via
    enable_time_series)."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        # Two AreaWeightedSingleTargetReducedMetric instances (reuse from .reduced):
        #   weighted_correction_magnitude -> ops.area_weighted_mean_dict(|corr|)
        #   weighted_correction_std      -> ops.area_weighted_std_dict(corr)
        ...

    def record_batch(self, correction_norm: TensorMapping, i_time_start: int) -> None: ...

    def get_logs(self, label: str, step_slice: slice | None = None) -> dict[str, Any]:
        # One wandb Table under "{label}/correction_series" — a key distinct
        # from the MeanAggregator's "{label}/series" because to_inference_logs
        # strips after the final "/" and the two tables would collide.
        ...

    def get_dataset(self) -> xr.Dataset:
        # weighted_correction_{magnitude,std}-{name} over forecast_step
        ...
```

## `fme/ace/aggregator/inference/main.py` (modified)

```python
def build_inference_evaluator_aggregator(
    ...,
    step_diagnostics: StepDiagnosticsMetricConfig | None = None,  # NEW
) -> "InferenceEvaluatorAggregator":
    # NEW: builds the single step_diagnostics_aggregator via
    # (step_diagnostics or StepDiagnosticsMetricConfig()).build(...) from
    # dataset_info.gridded_operations / variable_metadata, n_timesteps,
    # enable_time_series, and the existing normalize argument.
    ...


class InferenceEvaluatorAggregatorConfig:
    step_diagnostics: StepDiagnosticsMetricConfig = ...  # NEW — default-constructed;
    # also on the Legacy flag config

    def build(self, ...) -> "InferenceEvaluatorAggregator":  # CHANGED — forwards the config
        ...


class InferenceEvaluatorAggregator:
    def __init__(
        self,
        ...,
        step_diagnostics_aggregator: StepDiagnosticsAggregator | None = None,  # NEW
    ): ...

    def record_batch(self, data: PairedData) -> InferenceLogs:
        # CHANGED — if the aggregator is set:
        #   agg.record_batch(prediction=data.prediction,
        #     step_diagnostics=data.step_diagnostics,
        #     i_time_start=self._n_timesteps_seen) before slicing logs
        ...

    def _get_logs(self): ...                    # CHANGED — merge agg.summary_logs()
    def get_summary(self): ...                  # CHANGED — same merge (loss key untouched)
    def _get_inference_logs_slice(self, ...): ...  # CHANGED — merge agg.time_series_logs(step_slice)
    def flush_diagnostics(self, subdir=None): ...  # CHANGED — write agg.diagnostics()


class InferenceAggregatorConfig:
    step_diagnostics: StepDiagnosticsMetricConfig = ...  # NEW — default-constructed

    def build(
        self,
        dataset_info: DatasetInfo,
        n_timesteps: int,
        output_dir: str | None = None,
        save_diagnostics: bool = True,
        normalize: NormalizeFn | None = None,  # NEW
    ) -> "InferenceAggregator":
        # The no-target aggregator has no normalizer today; the config's
        # build returns None when normalize is None — callers that do not
        # pass one keep current behavior (metrics silently skipped).
        ...


class InferenceAggregator:
    def __init__(
        self,
        ...,
        step_diagnostics_aggregator: StepDiagnosticsAggregator | None = None,  # NEW
    ):
        # CHANGED — _log_time_series also true when the step-diagnostics
        # aggregator carries a time-series member (mean_norm has no other
        # member here)
        ...

    # record_batch / _get_logs / get_summary / _get_inference_logs_slice /
    # flush_diagnostics: CHANGED — same fan-out/merges as the evaluator
```

## `fme/ace/inference/inference.py` (modified)

```python
# CHANGED — the config.aggregator.build(...) call gains
#   normalize=stepper.normalizer.normalize
# (evaluator.py already passes it; no change there)
```

---

## Tests

## `fme/ace/aggregator/inference/test_step_diagnostics.py` (new)

```python
# Identity normalizer (a NormalizeFn ignoring apply_mean) where
# hand-computable exactness matters.

def test_correction_delta_aggregator_normalizes_without_mean():
    # GOAL: with a StandardNormalizer (nonzero means), recorded values equal
    # delta / std — the mean offset does not leak in.

def test_correction_delta_aggregator_raises_on_unknown_delta_key():
    # GOAL: a delta key the normalizer drops raises ValueError, not a silent skip.

def test_time_mean_aggregator_constant_offset():
    # GOAL: exact values — constant delta per variable yields
    # correction_magnitude/{var} == |delta| and (with record_maps) a constant
    # signed correction_map.

def test_time_mean_aggregator_masked_cells_do_not_poison_scalars():
    # GOAL: NaN off-mask delta cells + a mask-aware GriddedOperations yield
    # finite scalars matching a hand computation over on-mask cells only.

def test_time_mean_aggregator_null_masking_matches_unmasked_hand_computation():
    # GOAL: with NullSpatialMasking (no NaNs), values are identical to the
    # unmasked hand-computed case.

def test_time_mean_aggregator_maps_off_by_default():
    # GOAL: without record_maps, no correction_map logs and no map dataset
    # variables; scalars unaffected.

def test_mean_aggregator_constant_offset_series():
    # GOAL: per-step weighted_correction_magnitude == |delta| and
    # weighted_correction_std == 0 for a spatially constant correction;
    # a spatially varying case matches ops.area_weighted_std_dict.

def test_aggregators_silent_without_data():
    # GOAL: empty logs and empty dataset before any recorded batch.
    # PARAMETERIZE: both granular aggregator classes.

def test_correction_delta_aggregator_raises_on_time_dim_mismatch():
    # GOAL: a delta whose time dim differs from the prediction window's raises.

def test_metric_config_build_granularity():
    # GOAL: StepDiagnosticsMetricConfig.build honors the flags — default
    # (scalars on, maps off); both off or normalize=None -> None.

def test_evaluator_aggregator_logs_correction_metrics():
    # GOAL: integration — record_batch with a PairedData carrying a real
    # StepDiagnostics (constant-offset delta) logs
    # time_mean_norm/correction_magnitude/{var},
    # mean_norm/weighted_correction_{magnitude,std}/{var}; only corrected
    # variables appear; existing time_mean_norm/rmse metrics are unaffected;
    # flush_diagnostics writes the correction datasets.

def test_evaluator_aggregator_silent_paths():
    # GOAL: no correction logs and no diagnostics files.
    # PARAMETERIZE: config with both flags off; step_diagnostics=None.

def test_inline_training_drops_correction_series():
    # GOAL: enable_time_series=False keeps time_mean_norm correction metrics
    # but drops the mean_norm correction series.

def test_no_target_aggregator_logs_correction_metrics():
    # GOAL: InferenceAggregatorConfig.build(normalize=...) logs the same
    # metric keys from prediction-only PairedData.

def test_no_target_aggregator_skips_without_normalizer():
    # GOAL: build without normalize logs nothing correction-related
    # (backward compatibility for callers that do not pass one).

def test_config_defaults():
    # GOAL: both configs default to scalars on, maps off.
```
