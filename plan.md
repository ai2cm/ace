# Overview

Currently the Corrector in ace is responsible for diagnosing derived output values used in the loss, because no dedicated pathway existed for doing so. This leads to strange behaviors like having neural network output channels that are ignored (e.g. advection of moisture), and other outputs that are interpreted as something other than what they are (e.g. hfds network output being its value under sea ice but final value being hfds over the full gridcell). This makes the code significantly more confusing to think about, leading to complex consequences for example when thinking about pre-corrector optimization and corrector regularization.

Let's talk specifically about surface precipitation. In our critical path model, the corrector uses the network output for PRATEsfc in combination with the water budget defined by each other moisture field, and determines the magnitude needed for PRATEsfc to close the water budget. In other words, the network output is really the pattern of precipitation, and the corrector is responsible for deriving the final value of PRATEsfc. Afterwards, the corrector also clips PRATEsfc so it is strictly positive - this is a correction applied to the predicted PRATEsfc value.

Currently, there is no way to distinguish between these two actions applied to PRATEsfc. They both appear as a "delta", and it is not possible for example to regularize the magnitude of predicted negative precipitation without also regularizing the "derivation" of PRATEsfc. Applying pre-corrector optimization on the zero clipping may (or may not) make sense, but applying it to the derivation of PRATEsfc never would, as it would remove all gradient pathway to the budget non-closure in the other moisture fields.

In a sense, the code currently lies about the neural network output being PRATEsfc, which can confuse external users into thinking this output is meant to represent the physical field the code says it represents (e.g. Chapman et al. 2026 https://arxiv.org/pdf/2607.18416). In this plan, we address these issues by updating the presentation to match what the code is doing. This has a secondary benefit of making it easier to reason about corrector regularization and pre-corrector optimization - it should be scientifically valid to "apply corrector regularization to all corrected fields", but it currently is not. This would be one step in that direction.

If we later decided the network output should in fact be PRATEsfc and not just its relative pattern, the code would be updated accordingly so that it continues to reflect what our model actually does.

# Plan

We’d like to decouple these responsibilities by formalizing two classes of output variables:
- Values computed by the Module, but which are never optimized in the loss (computed-unoptimized, in this draft).
- Values which are derived from Module-output values, and are optimized in the loss (derived-optimized, in this draft).

Certain corrector features treat these two values as one identical value, with overwriting.

This first slice will target one such feature - precipitation in the context of global-mean moisture conservation - to prove out the code framework. It will introduce a field `PRATEsfc_relative_pattern` which is output by the neural network when this type of correction is enabled, has no target data, will not be optimized in the loss. It should be reported in aggregators in the way any other derived value without target data is reported. `PRATEsfc` itself will be a value derived from this value, which is directly optimized in the loss despite not being a NN output. Because of this, there will be no corrector "delta" assigned to PRATEsfc - it is derived, not corrected. In this pass, the responsibility for deriving PRATEsfc will remain in the corrector. Any renaming of the corrector or other refactoring to make this clearer is left for future work.


# Separation of concerns

This concern will live in the Corrector and SingleModuleStep. The Corrector will own all involved field names, exposing lists of field names as needed to the Step (e.g. so it can construct its own name lists, and including the output Packer). Objects above the Step shouldn’t require any changes.

The Step will need to use a modified normalizer/scaler to allow it to use the Corrector’s specification of how to normalize pre-derived fields. In the future, for example, we may want fields to be 0-1 scaled explicitly (e.g. this _would_ be the case for PRATEsfc_relative_pattern, except that we care about backwards compatibility at this stage). We will ensure the Corrector owns these details by providing a `wrap_module_output_normalizer` method the Step applies to its normalizer. This ensures the normalization library code itself doesn’t need to worry about Corrector concerns.
