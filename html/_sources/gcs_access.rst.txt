.. _gcs-access:

=======================================
Accessing data via Google Cloud Storage
=======================================

The following is an example of how to access the `ACE-climSST` checkpoint and datasets via Google Cloud Storage.

(Note that additional model datasets are available via the same bucket, including:

- `ACE2-ERA5` training/validation data, located at: ``gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr``
- `ACE2-SOM` training/validation data, located at ``gs://ai2cm-public-requester-pays/2024-12-05-ai2-climate-emulator-v2-som/SHiELD-SOM-C96``)

It requires that the user have an account with Google Cloud Platform (GCP) and a GCP project set up.
The user must also have the ``gsutil`` command line tool installed, which is part of the `Google Cloud SDK`_.

The data is hosted in a public `requester pays`_ Google Cloud Storage bucket, which means the user will be billed for data transfer costs incurred while downloading the data.



To download the ACE checkpoint from the public requester pays Google Cloud Storage bucket, run:

.. code-block:: bash

    gsutil -u YOUR_GCP_PROJECT cp gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/checkpoints/ace_ckpt.tar ace_ckpt.tar

To download the initial condition file, run:

.. code-block:: bash

    gsutil -u YOUR_GCP_PROJECT cp gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/initial_condition/ic_0011_2021010100.nc initial_condition.nc

The validation data can be used as forcing data for the checkpoint. For example, the 10-year validation data (approx. 190GB) can be downloaded with:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/validation .

It is possible to download a portion of the dataset only, but it is necessary to have enough data to span the desired prediction period. Alternatively, if interested in the complete training dataset, this is also available via the public bucket:

.. code-block:: bash

    gsutil -m -u YOUR_GCP_PROJECT cp -r gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs/train .

.. _requester pays: https://cloud.google.com/storage/docs/requester-pays
.. _Google Cloud SDK: https://cloud.google.com/sdk/docs/install