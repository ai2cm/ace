import sys

from fme.ace.data_loading.inference import (
    ExplicitIndices,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.data_loading.perturbation import (
    ConstantConfig,
    GreensFunctionConfig,
    PerturbationSelector,
    SSTPerturbation,
)
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.evaluator import (
    DataWriterConfig,
    InferenceDataLoaderConfig,
    InferenceEvaluatorAggregatorConfig,
    InferenceEvaluatorConfig,
    OceanConfig,
    run_evaluator_from_config,
)
from fme.ace.inference.inference import (
    ForcingDataLoaderConfig,
    InferenceAggregatorConfig,
    InferenceConfig,
    InitialConditionConfig,
    run_inference_from_config,
)
from fme.ace.models.healpix.healpix_activations import (
    CappedGELUConfig,
    DownsamplingBlockConfig,
)
from fme.ace.models.healpix.healpix_blocks import ConvBlockConfig, RecurrentBlockConfig
from fme.ace.registry.hpx import (
    HEALPixRecUNetBuilder,
    UNetDecoderConfig,
    UNetEncoderConfig,
)
from fme.ace.registry.sfno import SFNO_V0_1_0, SphericalFourierNeuralOperatorBuilder
from fme.core.corrector.corrector import CorrectorConfig
from fme.core.corrector.ocean import OceanCorrectorConfig
from fme.core.dataset.config import TimeSlice, XarrayDataConfig
from fme.core.gridded_ops import GriddedOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.ocean import SlabOceanConfig
from fme.core.optimization import SchedulerConfig
from fme.core.parameter_init import FrozenParameterConfig, ParameterInitializationConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.typing_ import Slice

from .train.train import run_train
from .train.train_config import (
    CopyWeightsConfig,
    DataLoaderConfig,
    EMAConfig,
    ExistingStepperConfig,
    InlineInferenceConfig,
    LoggingConfig,
    OptimizationConfig,
    SingleModuleStepperConfig,
    TrainConfig,
)

# Get all the names defined in the current module
module = sys.modules[__name__]
__all__ = [name for name in dir(module) if not name.startswith("_")]
del sys, module
