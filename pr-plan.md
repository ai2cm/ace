# Overview

Currently the Corrector in ace is responsible for diagnosing derived output values used in the loss, because no dedicated pathway existed for doing so. This leads to strange behaviors like having neural network output channels that are ignored (e.g. advection of moisture), and other outputs that are interpreted as something other than what they are (e.g. hfds network output being its value under sea ice but final value being hfds over the full gridcell). This makes the code significantly more confusing to think about, leading to complex consequences for example when thinking about pre-corrector optimization and corrector regularization.

We’d like to decouple these responsibilities by formalizing two classes of output variables:
- Values computed by the Module, but which are never optimized in the loss (computed-unoptimized, in this draft).
- Values which are derived from Module-output values, and are optimized in the loss (derived-optimized, in this draft).

Certain corrector features treat these two values as one identical value, with overwriting.

This first slice will target one such feature - precipitation in the context of global-mean moisture conservation - to prove out the code framework. It will introduce a field `PRATEsfc_relative_pattern` which is output by the neural network when this type of correction is enabled, has no target data, will not be optimized in the loss. It should be reported in aggregators in the way any other derived value without target data is reported. `PRATEsfc` itself will be a value derived from this value, which is directly optimized in the loss despite not being a NN output.

Note this is meant to make the current behavior of our critical path models more clearly reflected in the output of the Step. If we decided to use pre-corrector optimization for this field, we would need to remove this behavior or add a configuration toggle, which would be straightforward for an experiment branch. This is unlikely to be a good idea for this specific field - doing so would mean conservative closure biases in other moisture fields have no gradient pathway to the loss, as they would only exist as an unoptimized delta applied to precipitation. We should not design this feature for a case we do not use, and retain the freedom to redesign/modify this feature in the case we do use it or something similar.

# Separation of concerns

This concern will live in the Corrector and SingleModuleStep. The Corrector will own all involved field names, exposing lists of field names as needed to the Step (e.g. so it can construct its own name lists, and including the output Packer). Objects above the Step shouldn’t require any changes.

The Step will need to use a modified normalizer/scaler to allow it to use the Corrector’s specification of how to normalize pre-derived fields. In the future, for example, we may want fields to be 0-1 scaled explicitly (e.g. this _would_ be the case for PRATEsfc_relative_pattern, except that we care about backwards compatibility at this stage). We will ensure the Corrector owns these details by providing a `wrap_module_output_normalizer` method the Step applies to its normalizer. This ensures the normalization library code itself doesn’t need to worry about Corrector concerns.
