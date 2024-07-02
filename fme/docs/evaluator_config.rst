.. _evaluator-config:

================
Evaluator Config
================

.. literalinclude:: evaluator-config.yaml
   :language: yaml
   :caption: Example YAML Configuration

.. testcode::
   :hide:

   from fme.ace import InferenceEvaluatorConfig
   import yaml
   import dacite

   with open('evaluator-config.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InferenceEvaluatorConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure.
The configuration is divided into several sub-configurations, each with its own dataclass.
The top-level configuration is the :class:`fme.ace.InferenceEvaluatorConfig` class.

.. autoclass:: fme.ace.InferenceEvaluatorConfig
   :show-inheritance:
   :noindex:

The sub-configurations are:

.. autoclass:: fme.ace.LoggingConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InferenceDataLoaderConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InferenceInitialConditionIndices
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.ExplicitIndices
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.TimestampList
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.XarrayDataConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.DataWriterConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InferenceEvaluatorAggregatorConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.OceanConfig
   :show-inheritance:
   :noindex:
