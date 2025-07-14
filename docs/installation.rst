.. highlight:: shell

.. _installation:

============
Installation
============

All commands here are run from the top-level directory of the repository, unless otherwise stated.

This is unsupported, pre-alpha software: use at your own risk! We are actively developing this software
and will be making breaking changes to the API.

PyPI
----

To install the latest release directly from PyPI, use:

.. code-block:: shell

    pip install fme

Conda
-----

For convenience, we provide an easy way to create a conda environment with `fme` installed.
First, clone the repository:

.. code-block:: shell

    git clone git@github.com:ai2cm/ace.git

A make target is available to build a conda environment:

.. code-block:: shell

    make create_environment

This will create an environment named ``fme``. If you would like a different name, set the ENVIRONMENT_NAME variable:

.. code-block:: shell

    ENVIRONMENT_NAME=<environment_name> make create_environment

Development
-----------

To install directly from source for development, clone the repository:

.. code-block:: shell

    git clone git@github.com:ai2cm/ace.git

Once downloaded, you can install the sources in development mode (``-e`` flag) with the extra dependencies for development (``[dev]``) and versions pinned to the ones we use in development (``-c constraints.txt``) with the following command:

.. code-block:: shell

    pip install -c constraints.txt -e fme[dev]

Docker
------

A make target is available to build the Docker image:

.. code-block:: shell

    make build_docker_image
