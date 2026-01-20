.. _downscaling-inference:

=====================
Downscaling Inference
=====================


Overview
--------
The downscaling inference entrypoint generates high-resolution downscaled outputs from trained diffusion models using coarse-resolution input data. Unlike training or evaluation, this entrypoint does not require fine-resolution target data, making it suitable for generating downscaled predictions for any region and time period where coarse data is available.
Multiple outputs can be specified in a single configuration file, each with different spatial regions, time ranges, ensemble sizes, and output variables. Outputs are processed sequentially, with generation parallelized across GPUs using distributed data loading.

A separate zarr file is generated for each output at ``{experiment_dir}/{output_name}.zarr`` with dimensions ``(time, ensemble, latitude, longitude)``.

Launching Inference
-------------------

To run inference on GPUs:

.. code-block:: bash

   torchrun --nproc_per_node=<num_gpus> -m fme.downscaling.inference config.yaml

Replace ``<num_gpus>`` with the number of GPUs you want to use.


Example YAML Configuration
--------------------------

The following example shows a configuration which generates two outputs: one using ``EventConfig`` for a single time snapshot, and one using ``TimeRangeConfig`` for a time range.

.. literalinclude:: downscaling-inference-config.yaml
   :language: yaml



.. testcode::
   :hide:

   from fme.downscaling.inference import InferenceConfig
   import yaml
   import dacite

   with open('downscaling-inference-config.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      InferenceConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   # these paths are used in the documentation on this page
   # if they change then update the docs!
   assert config.model.checkpoint_path == "/HiRO.ckpt"
   assert config.data.coarse[0].data_path == "/output_directory"
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully



Configuration Structure
-----------------------

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure. The top-level configuration is the :class:`fme.downscaling.inference.inference.InferenceConfig` class.

.. autoclass:: fme.downscaling.inference.inference.InferenceConfig
   :show-inheritance:
   :noindex:


Output Configuration Types
----------------------------

The ``outputs`` list can contain two types of configurations: ``EventConfig`` for single time snapshots and ``TimeRangeConfig`` for time ranges.


EventConfig
^^^^^^^^^^^

:class:`fme.downscaling.inference.output.EventConfig` is used for generating a single time snapshot over a spatial region. This is useful for capturing specific events like hurricane landfall, extreme weather events, or any single-timestep high-resolution snapshot of a region.

.. autoclass:: fme.downscaling.inference.output.EventConfig
   :show-inheritance:
   :noindex:

Example EventConfig:

.. code-block:: yaml

   name: "hurricane_landfall_2023"
   save_vars: ["PRATEsfc"]
   n_ens: 64
   max_samples_per_gpu: 8
   event_time: "2023-09-15T12:00:00"
   time_format: "%Y-%m-%dT%H:%M:%S"
   lat_extent:
      start: 25.0
      stop: 35.0
   lon_extent:
      start: 260.0
      stop: 275.0

You can also use integer indices for ``event_time``:

.. code-block:: yaml

   name: "event_at_index_100"
   save_vars: ["PRATEsfc"]
   n_ens: 32
   max_samples_per_gpu: 4
   event_time: 100
   lat_extent:
      start: 30.0
      stop: 40.0


TimeRangeConfig
^^^^^^^^^^^^^^^

:class:`fme.downscaling.inference.output.TimeRangeConfig` is used for generating a time segment over a spatial region. This is the most common and flexible configuration, suitable for generating downscaled data over regions like CONUS, continental areas, or custom domains over extended time periods.

.. autoclass:: fme.downscaling.inference.output.TimeRangeConfig
   :show-inheritance:
   :noindex:


Example TimeRangeConfig with :class:`TimeSlice <fme.core.dataset.time.TimeSlice>`:

.. code-block:: yaml

     name: "CONUS_full_year"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         start_time: "2023-01-01T00:00:00"
         stop_time: "2023-12-31T18:00:00"

Example TimeRangeConfig with :class:`Slice <fme.core.typing_.Slice>`:

.. code-block:: yaml

     name: "first_year_indices"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         start: 0
         stop: 36

Example TimeRangeConfig with :class:`RepeatedInterval <fme.core.dataset.time.RepeatedInterval>`:

.. code-block:: yaml

     name: "weekly_snapshots"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         interval_length: "1d"
         block_length: "7d"
         start: "0d"


Common Configuration Patterns
------------------------------

Renaming model variables
^^^^^^^^^^^^^^^^^^^^^^^^^

You can rename input/output variables for the model loaded from the checkpoint.
This is useful if the model input variables' names are not the same as the variable names in the coarse input dataset, or if the model output variables are not the same as the variable names you want to save.

For example, ACE outputs coarse grid 10m winds as ``UGRD10m`` and ``VGRD10m``, while the downscaling checkpoint was created using data with variable names ``eastward_wind_at_ten_meters`` and ``northward_wind_at_ten_meters``. Thus, the ``model`` configuration in the example requires the following ``rename`` fields

.. code-block:: yaml

   model:
      checkpoint_path: /HiRO.ckpt
      rename:
         eastward_wind_at_ten_meters: UGRD10m
         northward_wind_at_ten_meters: VGRD10m


Multiple Outputs
^^^^^^^^^^^^^^^^

You can mix ``EventConfig`` and ``TimeRangeConfig`` outputs in a single configuration file. Outputs are processed sequentially:

.. code-block:: yaml

   outputs:
     - name: "event_1"
       event_time: "2023-06-15T00:00:00"
       save_vars: ["PRATEsfc"]
       n_ens: 128
       max_samples_per_gpu: 8
     - name: "time_range_1"
       time_range:
           start_time: "2023-01-01T00:00:00"
           stop_time: "2023-03-31T18:00:00"
       n_ens: 8
       max_samples_per_gpu: 8

Spatial Extent Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both ``EventConfig`` and ``TimeRangeConfig`` support spatial extent configuration via ``lat_extent`` and ``lon_extent``. These define the latitude and longitude bounds for the output region:

.. code-block:: yaml

   lat_extent:
       start: 25.0
       stop: 50.0
   lon_extent:
       start: 230.0
       stop: 295.0

Latitude bounds must be within (-88, 88) degrees. Longitude can be in the range (-180, 360) degrees. If not specified, the generated dataset region will default to the latitude range used in training (-66, 70) degrees. **Note- this will generate a very large output dataset!**

Ensemble Size Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``n_ens`` field specifies the total number of ensemble members to generate. The ``max_samples_per_gpu`` field controls how many time and/or ensemble samples are included in a single GPU batch, which affects memory usage and generation time. If not provided, the default value for ``max_samples_per_gpu`` is 4.

.. code-block:: yaml

   n_ens: 128
   max_samples_per_gpu: 4


Variable Selection
^^^^^^^^^^^^^^^^^^

Use the ``save_vars`` field to specify which variables to save to the output zarr file. If ``save_vars`` is ``null`` or not specified, all variables from the model output will be saved:

.. code-block:: yaml

   save_vars: ["PRATEsfc"]

Patch Prediction for Large Domains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For domains larger than the model's patch size, subdivision of the full domain into patches for prediction must be configured in the top-level ``patch`` section.
Generation for region sizes smaller than the size the model was trained on is not supported.


.. code-block:: yaml

   patch:
       divide_generation: true
       composite_prediction: true
       coarse_horizontal_overlap: 0

.. autoclass:: fme.downscaling.predictors.composite.PatchPredictionConfig
   :show-inheritance:
   :noindex:

Related Configuration Classes
-----------------------------

.. autoclass:: fme.downscaling.data.config.DataLoaderConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.downscaling.models.CheckpointModelConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.downscaling.predictors.cascade.CascadePredictorConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.downscaling.predictors.PatchPredictionConfig
   :show-inheritance:
   :noindex:

.. autoclass:: fme.core.logging_utils.LoggingConfig
   :show-inheritance:
   :noindex:
