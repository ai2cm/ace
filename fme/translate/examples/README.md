# Example training configurations (intended config surface)

Full example training configurations for the two target programs of the
`fme/translate` PR series, written against the *intended* config surface
through the whole series. The `component_pool:` and
`train_data:`/`validation_data:` blocks are implemented; everything else
is the shape the follow-on PRs must parse. They exist to be critiqued
now, before more code exists, and to let the critique drive the design —
they are not runnable.

| Config block | PR |
|---|---|
| `component_pool:` (domains / transforms / backbones, freeze, checkpoint init) | 1 — modulo channels-on-transforms (domain load lists derived), int-declared latent blocks expanded in backbone `in_names`/`out_names`, and identity normalization |
| `train_data:` / `validation_data:` stream lists (time-pairing derived from objective co-occurrence, not configured) | 2 — implemented; the pairing groups come from the `objectives:` list, which PR 3 supplies |
| `objectives:` (`translation`, `forward_prediction` over one-backbone component chains), `optimization:`, weighted-sum trainer | 3 |
| `inference:` composites (one-backbone component chains, Stepper-compatible export) | 4 |
| `latent_consistency` objective type, noise-conditioned encoder and latent-resampler registry entries | 6 |

PR 5 (SFNO cut-point decomposition) doesn't appear in either config: it
would show up as `parameter_init.weights_path` warm-starts on
encoder/backbone/decoder, and as a bare-processor backbone in the
latent-splice transfer variant.

- [multi-resolution-latent.yaml](multi-resolution-latent.yaml) —
  1°/2°/4° encoders/decoders into per-resolution latent domains, learned
  resamplers between latent resolutions (so the 1° and 2° paths into the
  4° latent share modules), a forward stepper in the 4° latent, forward
  prediction at all three resolutions through chained components against
  one shared backbone, and stochastic latent-consistency constraints
  between adjacent latent resolutions.
- [transfer-learning.yaml](transfer-learning.yaml) — learned ERA5↔SHiELD
  translators around a fully-frozen C96-SHiELD donor stepper, end-to-end
  rollout scored on ERA5, cycle consistency in both directions.
