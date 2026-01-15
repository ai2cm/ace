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


Output Configuration Types
----------------------------

The ``outputs`` list can contain two types of configurations: ``EventConfig`` for single time snapshots and ``TimeRangeConfig`` for time ranges. Both inherit from :class:`fme.downscaling.inference.output.DownscalingOutputConfig`.

.. autoclass:: fme.downscaling.inference.output.DownscalingOutputConfig
   :show-inheritance:
   :noindex:

EventConfig
^^^^^^^^^^^

:class:`fme.downscaling.inference.output.EventConfig` is used for generating a single time snapshot over a spatial region. This is useful for capturing specific events like hurricane landfall, extreme weather events, or any single-timestep high-resolution snapshot of a region.

.. autoclass:: fme.downscaling.inference.output.EventConfig
   :show-inheritance:
   :noindex:

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


TimeRangeConfig
^^^^^^^^^^^^^^^

:class:`fme.downscaling.inference.output.TimeRangeConfig` is used for generating a time segment over a spatial region. This is the most common and flexible configuration, suitable for generating downscaled data over regions like CONUS, continental areas, or custom domains over extended time periods.

.. autoclass:: fme.downscaling.inference.output.TimeRangeConfig
   :show-inheritance:
   :noindex:


Example TimeRangeConfig with TimeSlice:

.. code-block:: yaml

   - name: "CONUS_full_year"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         start_time: "2023-01-01T00:00:00"
         end_time: "2023-12-31T18:00:00"

Example TimeRangeConfig with Slice:

.. code-block:: yaml

   - name: "first_year_indices"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         start: 0
         stop: 36

Example TimeRangeConfig with RepeatedInterval:

.. code-block:: yaml

   - name: "weekly_snapshots"
     n_ens: 4
     max_samples_per_gpu: 4
     time_range:
         interval_length: "1d"
         block_length: "7d"
         start: "0d"


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
