from fme import ace
from fme.ace import InferenceConfig, InferenceEvaluatorConfig, TrainConfig

IMPORTED_SYMBOLS = ace.__all__

CHECKED_SYMBOLS = set[str]()
MISSING_SYMBOLS = set[type]()


def populate_missing_symbols(config: type):
    if not hasattr(config, "__dataclass_fields__") or not hasattr(config, "__name__"):
        return
    config_name = config.__name__
    if config_name in CHECKED_SYMBOLS:
        return
    CHECKED_SYMBOLS.add(config_name)
    if config_name not in ace.__all__:
        MISSING_SYMBOLS.add(config)
    elif getattr(ace, config_name) is not config:
        raise AssertionError(
            f"Symbol {config_name} is not the same type as the imported symbol"
        )
    for field in config.__dataclass_fields__.values():
        if hasattr(field.type, "__args__"):
            for type in field.type.__args__:
                if hasattr(type, "__dataclass_fields__"):
                    populate_missing_symbols(type)
        if hasattr(field.type, "__dataclass_fields__"):
            populate_missing_symbols(field.type)


def test_train_config_nested_dataclass_symbols_are_imported():
    try:
        populate_missing_symbols(TrainConfig)
        assert len(MISSING_SYMBOLS) == 0, f"Missing symbols: {MISSING_SYMBOLS}"
        assert ace.DataLoaderConfig.__name__ in CHECKED_SYMBOLS  # sanity checks
        assert ace.StepperConfig.__name__ in CHECKED_SYMBOLS
        assert ace.OptimizationConfig.__name__ in CHECKED_SYMBOLS
        assert ace.LoggingConfig.__name__ in CHECKED_SYMBOLS
        assert ace.InlineInferenceConfig.__name__ in CHECKED_SYMBOLS
        assert ace.EMAConfig.__name__ in CHECKED_SYMBOLS
        assert ace.Slice.__name__ in CHECKED_SYMBOLS
        assert ace.CopyWeightsConfig.__name__ in CHECKED_SYMBOLS
        assert ace.TimeLengthProbability.__name__ in CHECKED_SYMBOLS
    finally:
        CHECKED_SYMBOLS.clear()
        MISSING_SYMBOLS.clear()


def test_inference_evaluator_config_nested_dataclass_symbols_are_imported():
    try:
        populate_missing_symbols(InferenceEvaluatorConfig)
        assert len(MISSING_SYMBOLS) == 0, f"Missing symbols: {MISSING_SYMBOLS}"
        assert ace.InferenceDataLoaderConfig.__name__ in CHECKED_SYMBOLS
        assert ace.InferenceEvaluatorAggregatorConfig.__name__ in CHECKED_SYMBOLS
    finally:
        CHECKED_SYMBOLS.clear()
        MISSING_SYMBOLS.clear()


def test_inference_config_nested_dataclass_symbols_are_imported():
    try:
        populate_missing_symbols(InferenceConfig)
        assert len(MISSING_SYMBOLS) == 0, f"Missing symbols: {MISSING_SYMBOLS}"
        assert ace.ForcingDataLoaderConfig.__name__ in CHECKED_SYMBOLS
        assert ace.InitialConditionConfig.__name__ in CHECKED_SYMBOLS
        assert ace.InferenceAggregatorConfig.__name__ in CHECKED_SYMBOLS
        assert ace.StepperOverrideConfig.__name__ in CHECKED_SYMBOLS
    finally:
        CHECKED_SYMBOLS.clear()
        MISSING_SYMBOLS.clear()
