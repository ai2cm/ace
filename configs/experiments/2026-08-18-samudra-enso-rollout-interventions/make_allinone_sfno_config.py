#!/usr/bin/env python3
"""Generate the SFNO all-in-one arm: same task, spectral trunk.

Identical to the all-in-one config (same 103-in/101-out variable set, data,
loss, corrector, masks) with the Samudra ConvNeXt trunk swapped for the
deterministic SFNO recipe ACE pretrains on this same CM4 1-degree data
(embed_dim 384, 8 layers, dhconv - from configs/baselines/cm4-piControl).
The architecture comparison is the arm's single change: a global spectral
operator against a local convolutional one on the joint ocean+atmosphere
prediction task.

Run make_allinone_config.py first; this transforms its output.
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "wave1_configs" / "allinone-pretrain.yaml"
OUT = HERE / "wave1_configs" / "allinonesfno-pretrain.yaml"

SFNO_BUILDER = {
    "type": "SphericalFourierNeuralOperatorNet",
    "config": {
        "embed_dim": 384,
        "filter_type": "linear",
        "hard_thresholding_fraction": 1.0,
        "use_mlp": True,
        "normalization_layer": "instance_norm",
        "num_layers": 8,
        "operator_type": "dhconv",
        "scale_factor": 1,
        "separable": False,
        "spectral_layers": 3,
        "spectral_transform": "sht",
    },
}


def main() -> None:
    with open(BASE) as f:
        c = yaml.safe_load(f)

    cfg = c["stepper"]["step"]["config"]
    assert cfg["builder"]["type"] == "Samudra"
    cfg["builder"] = SFNO_BUILDER

    OUT.write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"wrote {OUT}")
    print(f"in={len(cfg['in_names'])} out={len(cfg['out_names'])} trunk=SFNO")


if __name__ == "__main__":
    main()
