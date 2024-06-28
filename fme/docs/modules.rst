=======
Modules
=======

ACE's code uses a module registry system to allow different machine learning architectures to plug into the framework.
This is managed by the :class:`fme.ace.ModuleSelector` configuration class, which is used to select a module type and version.

.. autoclass:: fme.ace.ModuleSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

The following module types are available:

.. include:: available_modules.rst

.. autofunction:: fme.ace.get_available_module_types

We primarily use Spherical Fourier Neural Operator (SFNO) networks in our work.

.. autoclass:: fme.ace.SphericalFourierNeuralOperatorBuilder
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: fme.ace.SFNO_V0_1_0
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

