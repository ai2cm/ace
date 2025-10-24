=================
Coupled Modeling
=================
.. _coupled:

.. tip:: We highly recommend familiarizing yourself with the standalone atmosphere only model (ACE) before using a coupled atmosphere-ocean model. See :ref:`quickstart <quickstart>` for running, evaluating, and training ACE.

Currently, coupled modeling supports coupling an atmosphere model (e.g. ACE) with an ocean model (e.g. Samudra).

Commands
========

The following commands are available, and can be run with ``--help`` for more information:

- ``python3 -m fme.coupled.validate_config`` - Validate a configuration file
- ``python3 -m fme.coupled.inference`` - Run a saved model checkpoint

Accessing coupled checkpoints and datasets
==========================================

Checkpoints for running coupled inference are available in the `ACE Collection on Hugging Face <https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4>`_.
You can find an example coupled model checkpoint under the name ``SamudrACE-CM4-piControl``.
The minimum requirements for running inference with SamudrACE are:

- a coupled model checkpoint
- initial conditions files containing all prognostic variables for the atmosphere and ocean respectively
- forcing dataset containing all input-only variables for the atmosphere

Save a ``config-inference.yaml`` file based on the :ref:`example config <coupled-inference-config>` with updated initial conditions and forcing paths for the downloaded data.
Specifically, ``initial_condition.atmosphere.path`` is the local initial condition file for the atmosphere,
and ``initial_condition.ocean.path`` for the ocean. ``forcing_loader.atmosphere.dataset.data_path`` should be the local directory containing the forcing data files for the atmosphere. ``forcing_loader.ocean.dataset.data_path`` is not required because there are no ocean specific forcing data.

Then in the ``fme`` conda environment, run inference with:

.. code-block:: bash

    python -m fme.coupled.inference config-inference.yaml

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.coupled.validate_config config-inference.yaml --config_type inference

Coupled inference config
========================
The example uses paths relative to the directory you run the command.
You should have a directory structure like:

::

   .
   ├── ckpt.tar
   ├── initial_conditions
   │   ├── atmosphere-ic.nc
   │   └── ocean-ic.nc
   ├── forcing_data
   │   ├── forcing_data_00.nc
   │   └── forcing_data_....nc

including a model checkpoint (``ckpt.tar``), atmosphere and ocean forcing and initial condition files.
You can find the checkpoint and forcing and initial condition data in the `SamudrACE Hugging Face page`_.

.. _SamudrACE Hugging Face page: https://huggingface.co/allenai/SamudrACE-CM4-piControl

Example inference YAML Configuration
------------------------------------

.. _coupled-inference-config:

.. literalinclude:: coupled-inference-config.yaml
   :language: yaml
   :caption: Example YAML Configuration

.. testcode::
   :hide:

   from fme.coupled import InferenceConfig
   import yaml
   import dacite

   with open('coupled-inference-config.yaml', 'r') as f:
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

Configuration structure
-----------------------

We use the :ref:`Builder pattern <Builder Pattern>` to load this configuration into a multi-level dataclass structure.
The configuration is divided into several sub-configurations, each with its own dataclass.
The top-level configuration is the :class:`fme.coupled.InferenceConfig` class.

.. autoclass:: fme.coupled.InferenceConfig
   :show-inheritance:
   :noindex:

Information for :class:`fme.coupled.CoupledInitialConditionConfig`
------------------------------------------------------------------

Initial condition configuration is similar to standalone ACE as documtned in :ref:`inference config <inference-config>`,
but in coupled configuration, `start_indices` correspond to the **ocean** initial condition file
, and the atmosphere initial condition file must contain the same time stamps.