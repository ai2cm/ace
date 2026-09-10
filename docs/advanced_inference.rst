.. _advanced_inference:

==================
Advanced inference
==================

Data writer configuration
=========================

This section provides an example of how to use the
:class:`fme.ace.DataWriterConfig` section of the inference configuration.
The data writer configuration is used to specify how the output from an
inference run is saved to disk. For flexibility, it is recommended to use
the ``files`` section, which allows for different output files to be written
with different frequencies and different variables. For example, the following
configuration will write the monthly means of all available variables to a
netCDF file called ``monthly_means.nc``, while also writing the daily mean
(assuming a 6-hour time step model) values of the ``total_water_path``
and ``TMP2m`` variables to a zarr dataset called ``daily_mean.zarr``:

.. literalinclude:: data-writer-example.yaml
   :language: yaml
   :caption: Example ACE Data Writer Configuration

.. testcode::
   :hide:

   from fme.ace import DataWriterConfig
   import yaml
   import dacite

   with open('data-writer-example.yaml', 'r') as f:
      config_dict = yaml.safe_load(f)

   config = dacite.from_dict(
      DataWriterConfig,
      data=config_dict,
      config=dacite.Config(strict=True)
   )
   print("Loaded successfully")

.. testoutput::
   :hide:

   Loaded successfully

Segmented inference
===================

To complete a long inference run on a computing system with a wall clock
limit, it may be necessary to chain multiple smaller segments together.
This can be done automatically by specifying a value for the ``--segments``
parameter in the ``fme.ace.inference`` entrypoint:

.. code-block:: bash

    python -m fme.ace.inference config-inference.yaml --segments 3

The above example will result in three segments being run with
``n_forward_steps`` each. Output from each segment will be stored in
subdirectories under the ``experiment_dir``, labeled by the segment start
time of the first (or only) ensemble member. If a segment directory already
exists, it will be skipped.

For example, running inference with the example configuration defined in the
:ref:`example-yaml-configuration` on the
:doc:`inference_config` page with 3 segments would result in the following
segment directories:

.. code-block:: text

    inference_output/
    ├── segment_19400101T00
    ├── segment_19400410T00
    └── segment_19400719T00

Manual segmented inference
==========================

The segmented run API requires that the configuration, modulo the initial
conditions and experiment subdirectory, will be held constant between segments.
It can sometimes be helpful to have finer grained control, for example if you
would like to run segments of different lengths and/or run segments with
different data writer configurations. In these sorts of cases, one can write a
shell script to run the segments manually.

As an illustration, say you would like to run a 71-year AMIP simulation starting
from 1940-01-01T00:00:00, but split off the final 10 years as a separate run to
save its time means and other aggregrated diagnostics separately; this is
useful, for instance if the 10-year period corresponded to the test period held
out from training.

.. code-block:: bash

    #!/bin/bash

    override="\
        experiment_dir=segment_19400101T00 \
        n_forward_steps=89124 \
        initial_condition.start_indices.n_initial_conditions=1 \
    "
    python -m fme.ace.inference config-inference.yaml --override $override

    override="\
        experiment_dir=segment_20010101T00 \
        n_forward_steps=13148 \
        initial_condition.path=segment_19400101T00/restart.nc \
        initial_condition.start_indices.n_initial_conditions=1 \
    "
    python -m fme.ace.inference config-inference.yaml --override $override

By adjusting the :class:`~fme.ace.DataWriterConfig` section of the inference config,
this strategy can be used to customize the frequency or other aspects of the output
saved from each run segment.
