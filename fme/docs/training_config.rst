.. _train-config:

===============
Training Config
===============

.. literalinclude:: train-config.yaml
   :language: yaml
   :caption: Example YAML Configuration

.. testcode::
   :hide:

   from fme.ace import TrainConfig
   import yaml
   import dacite

   with open('train-config.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      TrainConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure.
The configuration is divided into several sub-configurations, each with its own dataclass.
The top-level configuration is the :class:`fme.ace.TrainConfig` class.

.. autoclass:: fme.ace.TrainConfig
   :show-inheritance:
   :noindex:

The top-level sub-configurations are:

.. autoclass:: fme.ace.DataLoaderConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.SingleModuleStepperConfig
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

.. autoclass:: fme.ace.EMAConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.Slice
   :show-inheritance:
   :noindex: