fme: Full Model Emulation
======================================

**fme** ("full model emulation") is a python package for training, running
and evaluating climate model emulators, such as the Ai2 Climate Emulator.

Why use **fme**?
----------------
- **fme** provides a unified interface for training, running and evaluating AI models
  with a range of architectures (SFNO, GNNs, UNets) and applications (atmosphere modeling,
  ocean modeling, downscaling).
- Built by climate modelers for climate modelers! We follow similar configuration and
  evaluation practices as traditional climate models, making **fme** intuitive
  to adopt and use.
- Flexible data loading and writing capabilities. **fme** supports netCDF and zarr,
  as well as streaming directly from/to cloud object storage. At inference time, reductions
  such as monthly means can be computed on the fly, saving time and storage space.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   quickstart
   training_config
   inference_config
   downscaling_inference
   evaluator_config
   coupled
   gcs_access
   builder
   modules
   steps
   api
