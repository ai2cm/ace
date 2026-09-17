"""Combine the residual and full-field ocean checkpoints into single
ensemble-stepper checkpoints, one per blend weight.

The coupled evaluator takes a single ocean checkpoint path, so blends are
evaluated coupled by saving the ensemble stepper (which round-trips through
Stepper.from_state) as an ordinary checkpoint and mounting that.

Inputs (fetched from beaker):
  residfixbest: 01M1FKAM7N8MKN7ZTANMB2N38Y training_checkpoints/best_ckpt.tar
  pretrain0:    01KW2BQ83EGZ90WZ74CZ4TJATN
                training_checkpoints/best_inference_ckpt.tar
Outputs: combined_{tag}.tar with {"stepper": state}, uploaded as beaker
datasets (2026-09-17): ff10 01M2PBJV6K5BWE9DN6HJWT6JMV,
ff20 01M2PBNF2NMRTHN471WTKWB3C7, ff50 01M2PBWR0WFP90214RMY6CK9F4.
"""

import argparse

import torch

from fme.ace.stepper import load_stepper_ensemble

ARMS = {"ff10": 0.1, "ff20": 0.2, "ff50": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("residual_ckpt")
    parser.add_argument("full_field_ckpt")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS), choices=sorted(ARMS))
    args = parser.parse_args()
    for arm in args.arms:
        ff_fraction = ARMS[arm]
        stepper = load_stepper_ensemble(
            [args.residual_ckpt, args.full_field_ckpt],
            [round(1.0 - ff_fraction, 4), ff_fraction],
        )
        out = f"{args.out_dir}/combined_{arm}.tar"
        torch.save({"stepper": stepper.get_state()}, out)
        print(out)


if __name__ == "__main__":
    main()
