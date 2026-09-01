fme: Full Model Emulation
======================================

**fme** ("full model emulation") is a python package for training, running
and evaluating climate model emulators, such as the Ai2 Climate Emulator (ACE),
SamudrACE and HiRO-ACE.

Why use **fme**?
----------------
- **fme** provides a unified interface for training, running and evaluating AI models
  with a range of architectures (SFNO, GNNs, UNets) and applications (atmosphere,
  ocean and sea ice modeling, downscaling).
- Built by climate modelers for climate modelers! We follow similar configuration and
  evaluation practices as traditional climate models, making **fme** intuitive
  to adopt and use.
- Flexible data loading and writing capabilities. **fme** supports netCDF and zarr,
  as well as streaming directly from/to cloud object storage. At inference time, reductions
  such as monthly means can be computed on the fly, saving time and storage space.

.. warning::
  This codebase is primarily developed to support internal research efforts of the
  `Ai2 Climate Modeling <https://allenai.org/climate-modeling>`_ group. Use at your own risk!
  We are actively developing this software and sometimes make breaking changes to the API.

Table of contents
-----------------
.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Single component (e.g. ACE):

   Training <training_config>
   Inference <inference_config>
   Evaluator <evaluator_config>

.. toctree::
   :maxdepth: 1
   :caption: Coupled (e.g. SamudrACE)

   Inference <coupled>

.. toctree::
   :maxdepth: 1
   :caption: Downscaling (e.g. HiRO-ACE)

   Inference <downscaling_inference>

.. toctree::
   :maxdepth: 1
   :caption: Other topics:

   gcs_access
   builder
   Module registry <modules>
   Step registry <steps>
   api
