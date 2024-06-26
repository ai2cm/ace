import sys

from fme.ace.inference.evaluator import (
    DataWriterConfig,
    InferenceDataLoaderConfig,
    InferenceEvaluatorAggregatorConfig,
    InferenceEvaluatorConfig,
    OceanConfig,
    run_evaluator_from_config,
)
from fme.ace.inference.inference import InferenceConfig, run_inference_from_config
from fme.ace.registry.sfno import SFNO_V0_1_0, SphericalFourierNeuralOperatorBuilder
from fme.core.corrector import CorrectorConfig
from fme.core.data_loading.config import TimeSlice, XarrayDataConfig
from fme.core.data_loading.inference import InferenceInitialConditionIndices
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import FromStateNormalizer, NormalizationConfig
from fme.core.parameter_init import ParameterInitializationConfig
from fme.core.registry import ModuleSelector, get_available_module_types, register

from .train import run_train_from_config
from .train_config import (
    CopyWeightsConfig,
    DataLoaderConfig,
    EMAConfig,
    ExistingStepperConfig,
    InlineInferenceConfig,
    LoggingConfig,
    OptimizationConfig,
    SingleModuleStepperConfig,
    Slice,
    TrainConfig,
)

# Get all the names defined in the current module
module = sys.modules[__name__]
__all__ = [name for name in dir(module) if not name.startswith("_")]
del sys, module
