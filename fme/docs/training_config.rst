===============
Training Config
===============

.. literalinclude:: train-config.yaml
   :language: yaml
   :caption: Example YAML Configuration

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure.
The configuration is divided into several sub-configurations, each with its own dataclass.
The top-level configuration is the :class:`fme.ace.TrainConfig` class.

.. autoclass:: fme.ace.TrainConfig
   :show-inheritance:
   :noindex:

The sub-configurations are:

.. autoclass:: fme.ace.DataLoaderConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.XarrayDataConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.SingleModuleStepperConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.ModuleSelector
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.NormalizationConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.FromStateNormalizer
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.ParameterInitializationConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.OceanConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.WeightedMappingLossConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.CorrectorConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.ExistingStepperConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.OptimizationConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.LoggingConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InlineInferenceConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.CopyWeightsConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.Slice
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.TimeSlice
   :show-inheritance:
   :noindex:
