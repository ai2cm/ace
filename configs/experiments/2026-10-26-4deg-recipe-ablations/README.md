# 4-degree ablations of ACE2.2's recipe

Ablations of ACE2.2's recipe at 4 degrees daily in AIMIP mode. Each arm removes or swaps one of
ACE2.2's recipe choices, starting from an ACE2.2-like control. The whole-recipe comparison is the
ACE2.5 recipe in `../2026-10-23-ace25-aimip-4deg/`. Every arm has seeds 0-3 and is scored on its
1979-2014 trend and its +4 K response.

| arm | pretrain | fine-tune | change from the control |
|---|---|---|---|
| `ace22-like-4deg` | `ace22-pretrain.yaml` | `ace22-finetune.yaml` | control: ACE2.2's recipe |
| `ace22-like-4deg-no-gmr` | `ace22-pretrain-no-gmr.yaml` | `ace22-finetune.yaml` | no global mean removal |
| `ace22-like-4deg-diagnostic-near-surface` | `ace22-pretrain-diagnostic-near-surface.yaml` | `ace22-finetune.yaml` | TMP2m, Q2m, UGRD10m, VGRD10m diagnosed, not inputs |
| `ace22-like-4deg-no-bottleneck` | `ace22-pretrain-no-bottleneck.yaml` | `ace22-finetune.yaml` | all channels through one spectral filter |
| `ace22-like-4deg-ace25-ft` | the control's (no new pretrain) | `ace22-finetune-ace25.yaml` | control pretrain + ACE2.5's fine-tune (10 epochs, 3 steps, loss on every step) |

`ace22-like-4deg-ace25-ft` is not the ACE2.5 recipe: it fine-tunes the control's ACE2.2-like
pretrains with ACE2.5's fine-tune schedule. The ACE2.5 recipe (its own pretrain and fine-tune) is
the set of runs in `../2026-10-23-ace25-aimip-4deg/`, reused here and not rerun.

The control starts from the ACE2.5 recipe's pretrain config. It adds ACE2.2's recipe choices:
- global mean removal;
- latent global-mean clipping;
- prognostic near-surface fields;
- no h500 loss weight;
- ACE2.2's outputs.

Its fine-tune is ACE2.2's: 40 epochs on 1, 2, 4, 12 or 20 steps, with the loss on the last step.
It keeps ACE2.2's step counts, so at a daily step its windows run up to 20 days. The evaluated
checkpoint is the fine-tune's last EMA checkpoint: epoch 40, or epoch 10 for ACE2.5's fine-tune.

All arms share the ACE2.5 runs' code (ace `49ccc95`), image, data store and normalization, and
training and validation years. They also use its +4 K pair. The clip acts only at inference, so
every arm runs its +4 K pair and a 1979-2014 rollout (`long36-evaluator.yaml`) twice. One run uses
the checkpoint as trained. The other uses a copy with the clip disabled (`clip_off.py`). The
control and no-bottleneck arms also run a 1940-1978 rollout (`pre1979-evaluator.yaml`, clip on),
on years outside training and validation.

Constraints:

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout). No rollout
  in these configs reaches 2015.

Launch with `bash run.sh`. It is safe to rerun: each run launches whatever stage has become ready.
