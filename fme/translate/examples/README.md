# Example training configurations (intended config surface)

Full example training configurations for the two target programs of the
`fme/translate` PR series, written against the *intended* config surface
through the whole series. Only the `component_pool:` block is implemented
in PR 1 (the skeleton); everything else is the shape the follow-on PRs
must parse. They exist to be critiqued now, before more code exists, and
to let the critique drive the design — they are not runnable.

| Config block | PR |
|---|---|
| `component_pool:` (domains / transforms / backbones, freeze, checkpoint init) | 1 — modulo latent-block-name expansion in backbone `in_names`/`out_names` and identity normalization |
| `train_data:` / `validation_data:` streams, `sampling:` | 2 |
| `objectives:` (`translation`, `forward_prediction`), `optimization:`, weighted-sum trainer | 3 |
| `inference:` composites (Stepper-compatible export) | 4 |
| `latent_consistency` objective type, noise-conditioned encoder registry entries | 6 |

PR 5 (SFNO cut-point decomposition) doesn't appear in either config: it
would show up as `parameter_init.weights_path` warm-starts on
encoder/backbone/decoder, and as a bare-processor backbone in the
latent-splice transfer variant.

- [multi-resolution-latent.yaml](multi-resolution-latent.yaml) —
  1°/2°/4° per-resolution encoders/decoders into a shared 4° latent, a
  forward stepper in that latent, forward prediction at all three
  resolutions against one shared backbone, and stochastic
  latent-consistency constraints between adjacent resolutions.
- [transfer-learning.yaml](transfer-learning.yaml) — learned ERA5↔SHiELD
  translators around a fully-frozen C96-SHiELD donor stepper, end-to-end
  rollout scored on ERA5, cycle consistency in both directions.
