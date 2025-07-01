==========
Quickstart
==========

Install
=======

To install the latest release directly from PyPI, use:

.. code-block:: shell

    pip install fme

If desired, see the :ref:`installation <installation>` page for more information on installing from source or using conda.

Commands
========

The following commands are available, and can be run with ``--help`` for more information:

- ``python3 -m fme.ace.validate_config`` - Validate a configuration file
- ``python3 -m fme.ace.train`` - Train a model
- ``python3 -m fme.ace.inference`` - Run a saved model checkpoint
- ``python3 -m fme.ace.evaluator`` - Run a saved model checkpoint and compare to target data

Accessing ACE checkpoints and datasets
======================================

We have made multiple versions of ACE publicly available and citable via its `Hugging Face collection`_.
This is the recommended way of downloading ACE checkpoints and datasets, and the collection is updated with new checkpoints as they become available.
For a given model checkpoint, we generally provide the checkpoint, and (as described below) initial conditions, forcing, and training/evaluation data appropriate for that version of the ACE model.

Checkpoints and datasets can be downloaded from Hugging Face either via the web interface or using the `huggingface_hub`_ Python package. Installing the package
allows downloading checkpoints via the command line or programmatically, which can be helpful for large data files.

In addition to the methods described above, ACE checkpoints and datasets may be accessed through other means, though these may not be comprehensive:

- Zenodo: Selected checkpoints and data subsets are archived and citable via Zenodo. For example, see the `ACE-climSST Zenodo repository`_.
- Google Cloud Storage: Some checkpoints and datasets are hosted in a public `requester pays`_ GCS bucket; see :ref:`gcs-access` for more information.
- Globus guest collection: Some datasets are available via this method; see `Hugging Face collection`_ for more information.

.. _Hugging Face collection: https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4
.. _huggingface_hub: https://huggingface.co/docs/huggingface_hub/index
.. _ACE-climSST Zenodo repository: https://zenodo.org/doi/10.5281/zenodo.10791086
.. _requester pays: https://cloud.google.com/storage/docs/requester-pays

Running a Checkpoint (Inference)
================================

The minimum requirements for running inference with ACE are:

- a model checkpoint
- an initial conditions file containing all prognostic variables
- a forcing dataset containing all input-only variables

The initial conditions and forcing files may include more variables than the minimum required, but only the required variables will be used.
The code will run an ensemble of predictions starting from each time specified in the initial conditions file, or a subset of these times can be specified in the configuration file.
The forcing dataset must contain data for the times specified in the initial conditions file, as well as all timesteps required for the prediction period.

For example, for the `ACE2-ERA5` model, the initial conditions and forcing files can be downloaded via the `ACE2-ERA5 Hugging Face page`_

Save a ``config-inference.yaml`` file based on the :ref:`example config <inference-config>` with updated initial conditions and forcing paths for the downloaded data.
Specifically, ``initial_condition.path`` should be the local initial conditions file, and ``forcing_loader.dataset.data_path`` should be the local directory containing the forcing data files.

Then in the ``fme`` conda environment, run inference with:

.. code-block:: bash

    python -m fme.ace.inference config-inference.yaml

See the :ref:`inference-config` section for more information on the configuration.

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.ace.validate_config config-evaluator.yaml --config_type inference

.. tip::

    While inference can be performed without a GPU, it may be very slow. If running on a Mac, set the environmental variable
    ``export FME_USE_MPS=1`` to enable using the `Metal Performance Shaders`_ framework for GPU acceleration. Note this backend is
    not fully featured and it may not work with all inference features or for training. It is recommended to use the latest version
    of torch if using MPS.

.. _ACE2-ERA5 Hugging Face page: https://huggingface.co/allenai/ACE2-ERA5
.. _zarr: https://zarr.readthedocs.io/en/stable/
.. _Metal Performance Shaders: https://developer.apple.com/metal/pytorch/

Evaluating a Checkpoint
=======================

When target data is available, it is possible to evaluate the model using the ``fme.ace.evaluator`` module.
This requires a dataset, referred to as target data or alternatively training and validation data, that includes all input and output variables for the prediction period.

For example, for the `ACE2-ERA5` model, a 1-year (1940) subsample of the target data is available via the `ACE2-ERA5 Hugging Face page`_.

Alternatively, the entire 1940-2022 dataset is available via the public `requester pays`_ Google Cloud Storage bucket; see :ref:`gcs-access` for more information.
Note the dataset is large, meaning it may take a long time to download and may result in significant transfer costs.

Save a ``config-evaluator.yaml`` file based on the :ref:`example config <evaluator-config>` with updated paths for the downloaded data.
Then in the ``fme`` conda environment, run evaluation with:

.. code-block:: bash

    python -m fme.ace.evaluator config-evaluator.yaml

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.ace.validate_config config-evaluator.yaml --config_type evaluator


Training a Model
================

Like evaluation, training a model requires datasets with all input and output variables.

For the `ACE2-ERA5` model, 1-year (1940) subsample of the target dataset is available via the `ACE2-ERA5 Hugging Face page`_.

Alternatively, the entire 1940-2022 dataset is available via the public `requester pays`_ Google Cloud Storage bucket; see :ref:`gcs-access` for more information.
Note the dataset is large, meaning it may take a long time to download and may result in significant transfer costs.

You will also require scaling files (``centering.nc``, ``scaling-full-field.nc``, and ``scaling-residual.nc`` in the example training config) containing scalar values for the mean and standard deviation of each input and output variable.
These files are available in the `ACE2-ERA5 Hugging Face page`_ under ``training_validation_data/normalization``.
They can also be generated using the script located at ``scripts/data_process/get_stats.py``.

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
   # These are referenced in the paragraph just above, if they change then
   # update both the docs and this test!
   print(config.stepper.step.config["normalization"]["network"]["global_means_path"])
   print(config.stepper.step.config["normalization"]["network"]["global_stds_path"])
   print(config.stepper.step.config["normalization"]["loss"]["global_means_path"])
   print(config.stepper.step.config["normalization"]["loss"]["global_stds_path"])

.. testoutput::
   :hide:

   centering.nc
   scaling-full-field.nc
   centering.nc
   scaling-residual.nc

Save a ``config-train.yaml`` file based on the :ref:`example config <train-config>` with updated paths for the downloaded data.
Then in the ``fme`` conda environment, run evaluation with:

.. code-block:: bash

    torchrun --nproc_per_node RANK_COUNT -m fme.ace.train config-train.yaml

where ``RANK_COUNT`` is how many processors you want to run on.
This will typically be the number of GPUs you have available.
If running on a single GPU, you can omit the `torchrun` command and use ``python -m`` instead.

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.ace.validate_config config-train.yaml --config_type train


Wandb Integration
=================

For the optional Weights and Biases (wandb) integration, you will need to set the API key::

    export WANDB_API_KEY=wandb-api-key

where ``wandb-api-key`` is created and retrieved from the "API Keys" section of the `Wandb`_ settings page.
See also :class:`fme.ace.LoggingConfig` for configuration of logging to wandb.

.. _Wandb: https://wandb.ai/settings