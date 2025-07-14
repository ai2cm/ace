.. _inference-config:

================
Inference Config
================

The following is an example configuration for running inference.
While you can use absolute paths in the config yamls (we encourage it!), the example uses paths relative to the directory you run the command.
The example assumes you are running in a directory structure like:

::

   .
   ├── ace2_era5_ckpt.tar
   ├── initial_conditions
   │   ├── ic_1940.nc
   │   ├── ic_1950.nc
   │   ├── ...
   │   └── ic_2020.nc
   ├── forcing_data
   │   ├── forcing_1940.nc
   │   ├── forcing_1941.nc
   │   ├── ...
   │   └── forcing_1989.nc
   └── inference-config.yaml

that includes a model checkpoint (``ace2_era5_ckpt.tar``), forcing data (e.g., ``forcing_1940.nc``), and initial conditions (e.g., ``ic_1940.nc``).
You can find the checkpoint and forcing and initial condition data in the `ACE2-ERA5 Hugging Face page`_.

The specified initial condition file should contain a time dimension of at least length 1, but can also
contain multiple times. If multiple times are present and ``start_indices`` is not specified in the
:class:`fme.ace.InitialConditionConfig` configuration, the inference will run an ensemble using all times
in the initial condition file.  The ``ic_1940.nc`` file is an example of a file with multiple times, containing
initial conditions for each month of 1940.  For examples of selecting specific initial
conditions, see :ref:`initial-condition-examples`.

While netCDFs files are specified in the example, zarr stores are also compatible, e.g.,
specifying the parent folder containing the zarr store directory as the ``path``, setting ``engine`` to "zarr", and setting ``file_pattern`` to "<zarr_store_name>.zarr"
in the dataset configuration.  See :class:`fme.ace.XarrayDataConfig` for more information.

Example YAML Configuration
---------------------------

.. _ACE2-ERA5 Hugging Face page: https://huggingface.co/allenai/ACE2-ERA5

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
   assert config.checkpoint_path == "ace2_era5_ckpt.tar"
   assert config.initial_condition.path == "initial_conditions/ic_1940.nc"
   assert config.forcing_loader.dataset.data_path == "forcing_data"
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

.. autoclass:: fme.ace.StepperOverrideConfig
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


