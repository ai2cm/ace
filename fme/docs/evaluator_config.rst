.. _evaluator-config:

================
Evaluator Config
================

The following is an example configuration for running inference while evaluating against target data.
While you can use absolute paths in the config yamls (we encourage it!), the example uses paths relative to the directory you run the command.
The example assumes you are running in a directory structure like:

::

   .
   ├── ckpt.tar
   └── validation
       ├── data1.nc  # files can have any name, but must sort into time-sequential order
       ├── data2.nc  # can have any number of netCDF files
       └── ...

The ``.nc`` files correspond to data files like ``2021010100.nc`` in the `Zenodo repository`_, while ``ckpt.tar`` corresponds to a file like ``ace_ckpt.tar`` in that repository.

.. _Zenodo repository: https://zenodo.org/doi/10.5281/zenodo.10791086

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
   # these paths are used in the documentation on this page
   # if they change then update the docs!
   assert config.checkpoint_path == "ckpt.tar"
   assert config.loader.dataset.data_path == "validation"
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
