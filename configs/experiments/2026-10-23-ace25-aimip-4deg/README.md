# ACE2S paper recipe at 4 degrees, AIMIP mode

The ACE2S paper model's recipe (Jeremy's 4-degree arm, `configs/baselines/era5-ace2s-ablations/`
at ace `49ccc95`), retrained under the AIMIP protocol, to compare its in-sample warming trend and
+4 K SST response with the no-CO2 4-degree model `im4ecamc` and with ACE2.2-ERA5.

Changes from the 4-degree paper arm:

- no `global_mean_co2` input or next-step forcing;
- AIMIP split: train 1979-2013 without 1994, validate on 1994 and 2014;
- the inline 1940-2020 rollout replaced by `long_36year` (1979-2014, 8 initial conditions);
  `n_forward_steps` 13141 so the last-starting member also ends on 2014-12-31.

| stage | config | notes |
|---|---|---|
| pretrain | `pretrain.yaml` | 40 epochs, 1-step |
| fine-tune | `finetune.yaml` | 10 epochs, 3-step detached, from the pretrain's `best_ckpt.tar` |
| evaluate | `evaluate-control.yaml` | seed 0 only: the fine-tune's `ema_ckpt_0010.tar` against ERA5; 4 years from 1979-01-01 |
| +4 K | `inference-{control,p4k}.yaml` | the same checkpoint and window, run with `fme.ace.inference`, which can perturb SST; same window as `im4ecamc`'s +4 K run (wandb `muu8pjbc`), without its initial-condition perturbation |

Constraints:

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout).
- All stages run at ace `49ccc95` with image `oliverwm/fme-deps-only-54045d546`, as the paper
  arm did; the configs are passed to the jobs inline.

Four seeds (0-3) run the full chain; the in-sample trend is read from each fine-tune's epoch-10
`long_36year` rollout. The matching seed replicates of `im4ecamc`'s recipe are in
`../2026-10-23-im4ecamc-seeds/`.

Launch with `bash run.sh`. It is safe to rerun: each run launches whatever stage has become ready.
