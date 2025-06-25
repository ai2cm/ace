.. _train-config:

===============
Training Config
===============

The following is an example configuration for running training while evaluating against target data.
While you can use absolute paths in the config yamls (we encourage it!), the example uses paths relative to the directory you run the command.
The example is based on training with our full dataset (containing data from 10 ensemble runs) and assumes you are running in a directory structure like:

::

   .
   ├── ckpt.tar
   └── validation
       ├── data1.nc  # files can have any name, but must sort into time-sequential order
       ├── data2.nc  # can have any number of netCDF files
       └── ...
   └── traindata
         ├── ic_0001
         │   ├── data1.nc  # files can have any name, but must sort into time-sequential order
         │   ├── data2.nc  # can have any number of netCDF files
         │   └── ...
         ├── ic_0002
         │   └── ...
         ├── ...
         └── ic_0010
             └── ...

You can modify the example to run on fewer ensemble members by removing entries, or change the data paths as you wish.
The ``.nc`` files correspond to data files like ``2021010100.nc`` in the `Zenodo repository`_, while ``ckpt.tar`` corresponds to a file like ``ace_ckpt.tar`` in that repository.

.. _Zenodo repository: https://zenodo.org/doi/10.5281/zenodo.10791086


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
   # these paths are used in the documentation on this page
   # if they change then update the docs!
   assert config.validation_loader.dataset.data_path == "validation"
   assert config.train_loader.dataset.concat[0].data_path == "traindata/ic_0001"
   assert config.train_loader.dataset.concat[1].data_path == "traindata/ic_0002"
   assert config.train_loader.dataset.concat[9].data_path == "traindata/ic_0010"
   assert len(config.train_loader.dataset.concat) == 10
   assert config.inference.loader.dataset.data_path == "validation"
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

.. autoclass:: fme.ace.StepperConfig
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