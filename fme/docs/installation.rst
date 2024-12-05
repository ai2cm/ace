.. highlight:: shell

============
Installation
============

All commands here are run from the top-level directory of the repository, unless otherwise stated.

Experimental release
--------------------

There is no stable release. This is unsupported, pre-alpha software: use at your own risk!

To install directly from github, you can run:

.. code-block:: shell

    pip install 'git://github.com/ai2cm/full-model.git#egg=fme&subdirectory=fme'

Conda
-----

To install with Conda, you must retrieve the sources from github:

.. code-block:: shell

    git clone git@github.com:ai2cm/full-model.git

A make target is available to build a conda environment:

.. code-block:: shell

    make create_environment

This will create an environment named ``fme``. If you would like a different name, set the ENVIRONMENT_NAME variable:

.. code-block:: shell

    ENVIRONMENT_NAME=<environment_name> make create_environment

Development
-----------

The package is not yet available on PyPI. Before installing, you must retrieve the sources from github:

.. code-block:: shell

    git clone git@github.com:ai2cm/full-model.git

Once downloaded, you can install the sources in development mode (``-e`` flag) with the extra dependencies for development (``[dev]``) and versions pinned to the ones we use in development (``-c constraints.txt``) with the following command:

.. code-block:: shell

    pip install -c constraints.txt -e fme[dev]

Docker
------

A make target is available to build the Docker image:

.. code-block:: shell

    make build_docker_image
