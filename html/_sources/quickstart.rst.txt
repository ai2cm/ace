==========
Quickstart
==========

Install
=======

Clone this repository. Then assuming `conda`_ is available, run::

    make create_environment

to create a conda environment called ``fme`` with dependencies and source code installed. Alternatively, a Docker image can be built with ``make build_docker_image``. You may verify installation by running ``pytest fme/``.

.. _conda: https://docs.conda.io/en/latest/

Wandb Integration
=================

For the optional Weights and Biases (wandb) integration, you will need to set the API key::

    export WANDB_API_KEY=wandb-api-key

where `wandb-api-key` is created and retrieved from the "API Keys" section of the `Wandb`_ settings page.

.. _Wandb: https://wandb.ai/settings

Commands
========

The following commands are available, and can be run with ``--help`` for more information:

- ``python3 -m fme.ace.validate_config`` - Validate a configuration file
- ``python3 -m fme.ace.train`` - Train a model
- ``python3 -m fme.ace.inference`` - Run a saved model checkpoint
- ``python3 -m fme.ace.evaluator`` - Run a saved model checkpoint and compare to target data

Running a Checkpoint
====================

To run a model checkpoint, you need an initial conditions file containing all model inputs, and a forcing dataset containing all input-only variables.
The files may include more variables, as in the example datasets below, but only the required variables will be used.
The code will run an ensemble of predictions starting from each time specified in the initial conditions file.
The forcing dataset must contain data for the times specified in the initial conditions file, as well as all timesteps required for the prediction period.

An initial condition file is available via a public `requester pays`_ Google Cloud Storage bucket.

.. code-block:: bash

    gsutil -u YOUR_GCP_PROJECT cp gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/initial_condition/ic_0011_2021010100.nc initial_condition.nc

The checkpoint and a 1-year subsample of the validation data are available at this `Zenodo repository`_.
This validation data can be used as forcing data for the checkpoint.

Alternatively, if interested in the complete dataset, this is available via a public `requester pays`_ Google Cloud Storage bucket.
For example, the 10-year validation data (approx. 190GB) can be downloaded with:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .

It is possible to download a portion of the dataset only, but it is necessary to have enough data to span the desired prediction period. The checkpoint is also available on GCS at `gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/checkpoints/ace_ckpt.tar`.

.. _Zenodo repository: https://zenodo.org/doi/10.5281/zenodo.10791086
.. _requester pays: https://cloud.google.com/storage/docs/requester-pays

Save a ``inference-config.yaml`` file based on the :ref:`example config <inference-config>` with updated paths for the downloaded data.
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
    not fully featured and it may not work with all inference features or for training.

.. _Metal Performance Shaders: https://developer.apple.com/metal/pytorch/

Evaluating a Checkpoint
=======================

When target data is available, it is possible to evaluate the model using the ``fme.ace.evaluator`` module.
This requires a dataset including all input and output variables for the prediction period.
The checkpoint and a 1-year subsample of the validation data are available at this `Zenodo repository`_.
Download these to your local filesystem.

Alternatively, if interested in the complete dataset, this is available via a public `requester pays`_ Google Cloud Storage bucket.
For example, the 10-year validation data (approx. 190GB) can be downloaded with:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .

Save a ``config-evaluator.yaml`` file based on the :ref:`example config <evaluator-config>` with updated paths for the downloaded data.
Then in the ``fme`` conda environment, run evaluation with:

.. code-block:: bash

    python -m fme.ace.evaluator config-evaluator.yaml

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.ace.validate_config config-evaluator.yaml --config_type evaluator

Training a Model
================

Like inference, training a model requires datasets with all input and output variables.

The complete training dataset is available via a public `requester pays`_ Google Cloud Storage bucket.
Note the dataset is large, meaning it may take a long time to download and may result in significant transfer costs.
The 100-year training data (approx. 1.9 TB) can be downloaded with:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/train .

It is advisable to use a separate datset for validation.
The 10-year validation data (approx. 190GB) can be downloaded with:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .

You will also require scaling files (``centering.nc`` and ``scaling.nc`` in the example training config) containing scalar values for the mean and standard deviation of each input and output variable.
These are generated using the script located at ``scripts/data_process/get_stats.py``.

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
   print(config.stepper.normalization.global_means_path)
   print(config.stepper.normalization.global_stds_path)

.. testoutput::
   :hide:

   centering.nc
   scaling.nc

Save a ``config-train.yaml`` file based on the :ref:`example config <train-config>` with updated paths for the downloaded data.
Then in the ``fme`` conda environment, run evaluation with:

.. code-block:: bash

    torchrun --nproc_per_node RANK_COUNT -m fme.ace.train config-train.yaml

where RANK_COUNT is how many processors you want to run on.
This will typically be the number of GPUs you have available.
If running on a single GPU, you can omit the `torchrun` command and use ``python -m`` instead.

If you run into configuration issues, you can validate your configuration with

.. code-block:: bash

    python -m fme.ace.validate_config config-train.yaml --config_type train
