.. _inference-config:

================
Inference Config
================

.. literalinclude:: inference-config.yaml
   :language: yaml
   :caption: Example YAML Configuration


.. testcode::
   :hide:

   from fme.ace import InferenceConfig
   import yaml
   import dacite

   with open('inference-config.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InferenceConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure.
The configuration is divided into several sub-configurations, each with its own dataclass.
The top-level configuration is the :class:`fme.ace.InferenceConfig` class.

.. autoclass:: fme.ace.InferenceConfig
   :show-inheritance:
   :noindex:

The sub-configurations are:

.. autoclass:: fme.ace.LoggingConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InitialConditionConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.ForcingDataLoaderConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.XarrayDataConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.DataWriterConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.InferenceAggregatorConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.OceanConfig
   :show-inheritance:
   :noindex:
