fme: Full Model Emulation
======================================

**fme** ("full model emulation") is a python package for training, running
and evaluating climate model emulators, such as the Ai2 Climate Emulator (ACE),
SamudrACE and HiRO-ACE.

Why use **fme**?
----------------
- **fme** provides a unified interface for training, running and evaluating AI models
  with a range of architectures (SFNO, GNNs, UNets) and applications (e.g. global atmosphere,
  ocean and sea ice modeling, regional downscaling).
- We strive for a balance of flexibility to enable easy prototyping of new modeling strategies
  with ease-of-use and performance.
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

Existing capabilities
---------------------

This codebase enables the training and evaluation of models with a variety of
architectures and applications. For example, we support:

**Global atmosphere modeling**:
  - deterministic global weather-climate atmospheric models like
    `ACE2-ERA5 <https://huggingface.co/allenai/ACE2-ERA5>`_
  - stochastic global weather-climate atmospheric models like
    `ACE2S-SHiELD+ <https://huggingface.co/allenai/ACE2S-SHiELD-plus>`_

**Global ocean and sea ice modeling**:
  - ocean and sea ice models like
    `SamudraI <https://huggingface.co/allenai/SamudrACE-CM4-piControl>`_
    and `FloeNet <https://huggingface.co/M2LInES/FloeNet-OM4>`_

**Coupled climate modeling**:
  - coupled atmosphere-ocean-sea ice models like SamudrACE
    (`CM4-piControl <https://huggingface.co/allenai/SamudrACE-CM4-piControl>`_,
    `E3SMv3 <https://huggingface.co/allenai/SamudrACE-E3SMv3>`_)

**Regional downscaling**:
  - diffusion-based downscaling models like
    `HiRO <https://huggingface.co/allenai/HiRO-ACE>`_

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

   Module registry <modules>
   Step registry <steps>
   gcs_access
   builder
   api
