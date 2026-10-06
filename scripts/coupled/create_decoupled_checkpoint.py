import argparse
import logging

import torch

from fme.coupled.stepper import load_coupled_stepper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component", type=str, help="Component to extract: 'ocean' or 'atmosphere'."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to an existing CoupledStepper training checkpoint.",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path for the new component checkpoint."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    component = args.component
    in_path = args.input_path
    out_path = args.output_path

    stepper = load_coupled_stepper(
        in_path
    )  # ensure this is a valid CoupledStepper checkpoint

    if component == "atmosphere":
        ckpt = {"stepper": stepper.atmosphere.get_state()}
    elif component == "ocean":
        ckpt = {"stepper": stepper.ocean.get_state()}
    else:
        raise ValueError(f"Unrecognized --component arg '{component}'")

    torch.save(ckpt, out_path)
    logging.info(f"New {component} saved to {out_path}")
