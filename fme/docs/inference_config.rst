.. _inference-config:

================
Inference Config
================

The following is an example configuration for running inference.
While you can use absolute paths in the config yamls (we encourage it!), the example uses paths relative to the directory you run the command.
The example assumes you are running in a directory structure like:

::

   .
   ├── ace_ckpt.tar
   ├── climSST
   │   ├── forcing_2021.zarr
   │   ├── ic_2021-01-01.zarr
   │   └── ic_2021.zarr
   └── inference-config.yaml

that includes a model checkpoint (``ace_ckpt.tar``), forcing data (``forcing_2021.zarr``), and an initial condition (e.g., ``ic_2021-01-01.zarr``).

The specified initial condition file should contain a time dimension of at least length 1, but can also
contain multiple times. If multiple times are present and ``start_indices`` is not specified in the
:class:`fme.ace.InitialConditionConfig` configuration, the inference will run an ensemble using all times
in the initial condition file.  The ``ic_2021.zarr`` file is an example of a file with multiple times, containing
initial conditions for each month of 2021.  For examples of selecting specific initial
conditions, see :ref:`initial-condition-examples`.

While Zarr files are specified in the example, netCDFs are also compatible. E.g.,
specifying the parent folder with the netCDF files as the ``path`` setting ``engine`` to ``netcdf4``
in the dataset configuration.  See :class:`fme.ace.XarrayDataConfig` for an example.

Example YAML Configuration
---------------------------

.. TODO: Add updated Zenodo repository

.. _Zenodo repository: https://zenodo.org/doi/10.5281/zenodo.10791086

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
   # these paths are used in the documentation on this page
   # if they change then update the docs!
   assert config.checkpoint_path == "ace_ckpt.tar"
   assert config.initial_condition.path == "climSST/ic_2021.zarr"
   assert config.forcing_loader.dataset.data_path == "climSST"
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully

Configuration structure
-----------------------

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


   .. _initial-condition-examples:
:class:`fme.ace.InitialConditionConfig` Examples
-------------------------------------------------

The ``start_indices`` attribute can be used to specify which initial conditions
to use when multiple are present in the dataset (instead of using all available).
The following examples show example selections using the yaml builder pattern for
an ``InitialConditionConfig``.


:class:`fme.ace.InferenceInitialConditionIndices`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Select a number of regularly spaced initial conditions.

.. literalinclude:: configs/inference-ic-indices.yaml
   :language: yaml

.. testcode::
   :hide:

   from fme.ace import InitialConditionConfig
   import dacite
   import yaml

   with open('configs/inference-ic-indices.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InitialConditionConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   assert config.start_indices.n_initial_conditions == 3

:class:`fme.ace.TimestampList`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Selecting two timestamps from the initial conditions.

.. literalinclude:: configs/timestamp-list.yaml
   :language: yaml

.. testcode::
   :hide:

   from fme.ace import InitialConditionConfig
   import dacite
   import yaml

   with open('configs/timestamp-list.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InitialConditionConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   assert config.start_indices.times[0] == '2021-01-01T00:00:00'

:class:`fme.ace.ExplicitIndices`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Selecting specific indices from the initial conditions.

.. literalinclude:: configs/explicit-indices.yaml
   :language: yaml

.. testcode::
   :hide:

   from fme.ace import InitialConditionConfig
   import dacite
   import yaml

   with open('configs/explicit-indices.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InitialConditionConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   assert config.start_indices.list[1] == 3


