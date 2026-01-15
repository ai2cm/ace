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

.. code-block:: yaml

   experiment_dir: /results
   model:
       checkpoint_path: /checkpoints/best.ckpt
   data:
       coarse:
       - data_path: /climate-default/X-SHiELD-AMIP-downscaling
         engine: zarr
         file_pattern: 100km.zarr
       num_data_workers: 2
       strict_ensemble: False
   patch:
       divide_generation: true
       composite_prediction: true
       coarse_horizontal_overlap: 1
   outputs:
     - name: "WA_AR_20230206"
       save_vars: ["PRATEsfc"]
       n_ens: 128
       max_samples_per_gpu: 8
       event_time: "2023-02-06T06:00:00"
       lat_extent:
           start: 36.0
           stop:  52.0
       lon_extent:
           start: 228.0
           stop: 244.0
     - name: "CONUS_2023"
       save_vars: ["PRATEsfc"]
       n_ens: 8
       max_samples_per_gpu: 8
       time_range:
          start_time: "2023-01-01T00:00:00"
          end_time: "2023-12-31T18:00:00"
       lat_extent:
           start: 22.0
           stop:  50.0
       lon_extent:
           start: 230.0
           stop: 295.0
   logging:
       log_to_screen: true
       log_to_wandb: false
       log_to_file: true
       project: downscaling
       entity: my_organization

Configuration Structure
-----------------------

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure. The top-level configuration is the :class:`fme.downscaling.inference.inference.InferenceConfig` class.

.. autoclass:: fme.downscaling.inference.inference.InferenceConfig
   :show-inheritance:
   :noindex:

The configuration consists of the following main sections:

- **model**: Model specification to load for generation. Can be either a :class:`fme.downscaling.models.CheckpointModelConfig` (single model) or :class:`fme.downscaling.predictors.cascade.CascadePredictorConfig` (cascaded models). For most use cases (including the public release model) this is a `fme.downscaling.models.CheckpointModelConfig`.

- **data**: Base data loader configuration (:class:`fme.downscaling.data.config.DataLoaderConfig`) that is shared across all outputs. This includes the coarse input data source and data loading settings. Each output selects its own spatial and temporal extents.

- **experiment_dir**: Directory where generated zarr datasets and logs will be saved.

- **outputs**: List of output specifications. Each output is either an :class:`fme.downscaling.inference.output.EventConfig` or :class:`fme.downscaling.inference.output.TimeRangeConfig`. Each output generates a separate zarr file.

- **logging**: Logging configuration (:class:`fme.core.logging_utils.LoggingConfig`) for screen, file, and wandb logging.

- **patch**: Default patch prediction configuration (:class:`fme.downscaling.predictors.PatchPredictionConfig`) for specifying how to handle composite generation of domains larger than the model's patch size.


Output Configuration Types
----------------------------

The ``outputs`` list can contain two types of configurations: ``EventConfig`` for single time snapshots and ``TimeRangeConfig`` for time ranges. Both inherit from :class:`fme.downscaling.inference.output.DownscalingOutputConfig`.

.. autoclass:: fme.downscaling.inference.output.DownscalingOutputConfig
   :show-inheritance:
   :noindex:

EventConfig
^^^^^^^^^^^

:class:`fme.downscaling.inference.output.EventConfig` is used for generating a single time snapshot over a spatial region. This is useful for capturing specific events like hurricane landfall, extreme weather events, or any single-timestep high-resolution snapshot of a region.

If ``n_ens > max_samples_per_gpu``, this event can be run in a distributed manner where each GPU generates a subset of the ensemble members for the event.

Required Fields:
  - **event_time**: Timestamp or integer index of the event. If string, must match ``time_format``.

Optional Fields:
  - **time_format**: strptime format for parsing ``event_time`` string. Default: ``"%Y-%m-%dT%H:%M:%S"`` (ISO 8601).
  - **lat_extent**: Latitude bounds in degrees [-88, 88]. Default: full extent of the underlying data.
  - **lon_extent**: Longitude bounds in degrees [-180, 360]. Default: full extent of the underlying data.

Example EventConfig:

.. code-block:: yaml

   - name: "hurricane_landfall_2023"
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

   - name: "event_at_index_100"
     save_vars: ["PRATEsfc"]
     n_ens: 32
     max_samples_per_gpu: 4
     event_time: 100
     lat_extent:
         start: 30.0
         stop: 40.0

.. autoclass:: fme.downscaling.inference.output.EventConfig
   :show-inheritance:
   :noindex:

TimeRangeConfig
^^^^^^^^^^^^^^^

:class:`fme.downscaling.inference.output.TimeRangeConfig` is used for generating a time segment over a spatial region. This is the most common and flexible configuration, suitable for generating downscaled data over regions like CONUS, continental areas, or custom domains over extended time periods.

Required Fields:
  - **time_range**: Time selection specification. Can be one of three formats (see below).

Optional Fields:
  - **lat_extent**: Latitude bounds in degrees [-88, 88]. Default: full extent of the underlying data.
  - **lon_extent**: Longitude bounds in degrees [-180, 360]. Default: full extent of the underlying data.

Time Range Formats
"""""""""""""""""""

The ``time_range`` field supports three formats:

1. **TimeSlice** (timestamp-based): Use start and stop timestamps.

   .. code-block:: yaml

      time_range:
          start_time: "2023-01-01T00:00:00"
          end_time: "2023-12-31T18:00:00"

2. **Slice** (index-based): Use integer indices.

   .. code-block:: yaml

      time_range:
          start: 0
          stop: 36

3. **RepeatedInterval** (repeating pattern): Use a repeating time pattern.

   .. code-block:: yaml

      time_range:
          interval_length: "1d"
          block_length: "7d"
          start: "2d"

   This example selects 1 day of data starting after 2 days, repeated every 7 days. All three values can be either integers (for index-based) or strings representing timedeltas (e.g., "1d", "7d", "2d").

Example TimeRangeConfig with TimeSlice:

.. code-block:: yaml

   - name: "CONUS_full_year"
     save_vars: ["PRATEsfc"]
     n_ens: 8
     max_samples_per_gpu: 8
     time_range:
         start_time: "2023-01-01T00:00:00"
         end_time: "2023-12-31T18:00:00"
     lat_extent:
         start: 22.0
         stop: 50.0
     lon_extent:
         start: 230.0
         stop: 295.0

Example TimeRangeConfig with Slice:

.. code-block:: yaml

   - name: "first_year_indices"
     save_vars: ["PRATEsfc"]
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         start: 0
         stop: 365
     lat_extent:
         start: 30.0
         stop: 45.0

Example TimeRangeConfig with RepeatedInterval:

.. code-block:: yaml

   - name: "weekly_snapshots"
     n_ens: 16
     max_samples_per_gpu: 8
     time_range:
         interval_length: "1d"
         block_length: "7d"
         start: "0d"
     lat_extent:
         start: 35.0
         stop: 50.0

.. autoclass:: fme.downscaling.inference.output.TimeRangeConfig
   :show-inheritance:
   :noindex:

Common Configuration Patterns
------------------------------

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
           end_time: "2023-03-31T18:00:00"
       save_vars: ["PRATEsfc", "TMP2m"]
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

Latitude bounds must be within (-88, 88) degrees. Longitude can be in the range (-180, 360) degrees. If not specified, the generated dataset region will default to these ranges. **Note- this will generate a very large output dataset!**

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

For domains larger than the model's patch size, subdivision of the full domain into patches for prediction can be enabled. Configure this in the top-level ``patch`` section or override per-output:

.. code-block:: yaml

   patch:
       divide_generation: true
       composite_prediction: true
       coarse_horizontal_overlap: 0

- ``divide_generation``: Enable patch-based generation for large domains.
- ``composite_prediction``: Composite patches together (recommended for seamless outputs).
- ``coarse_horizontal_overlap``: Overlap between patches in coarse grid cells is averaged in final prediction (0 means no overlap).

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
