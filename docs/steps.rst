=====
Steps
=====

ACE's code uses a "step" registry system to allow various emulation configuration step objects to be specified. (A step object consists of a
specific configuration of NN module calls and other operations such as normalization, denormalization, correction, etc.).
In ACE's hierarchy, a stepper contains the step object, which in turn may contain one or more NN modules.
Step registry is managed by the :class:`fme.ace.StepSelector` configuration class, which is used to select a step type and version.

.. autoclass:: fme.ace.StepSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

The following step types are available:

.. include:: available_steps.rst

.. autofunction:: fme.core.step.StepSelector.get_available_types

The following step builders are available:

.. autoclass:: fme.core.step.SingleModuleStepConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: fme.core.step.MultiCallStepConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: fme.core.step.SeparateRadiationStepConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex: