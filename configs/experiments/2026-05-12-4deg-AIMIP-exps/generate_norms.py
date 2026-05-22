#!/usr/bin/env python3
"""Generate 4 training configs for the normalization ablation study.

Produces norm-base.yaml, norm-gmst.yaml, norm-resid.yaml, and
norm-gmst-resid.yaml by varying remove_global_mean_surface_temperature and
residual_prediction on top of ace-train-config-4deg-AIMIP.yaml.
"""

import copy
import pathlib

import yaml

SCRIPT_DIR = pathlib.Path(__file__).parent
BASE_CONFIG = SCRIPT_DIR / "ace-train-config-4deg-AIMIP.yaml"

COMBINATIONS = [
    {"remove_gmst": False, "residual_pred": False, "name": "base"},
    {"remove_gmst": True, "residual_pred": False, "name": "gmst"},
    {"remove_gmst": False, "residual_pred": True, "name": "respred"},
    {"remove_gmst": True, "residual_pred": True, "name": "gmst-respred"},
]


def main():
    header_lines = []
    with open(BASE_CONFIG) as f:
        for line in f:
            if line.startswith("# arg:"):
                header_lines.append(line.rstrip())
            else:
                break

    with open(BASE_CONFIG) as f:
        base_config = yaml.safe_load(f)

    for combo in COMBINATIONS:
        config = copy.deepcopy(base_config)

        config["stepper"]["step"]["config"]["normalization"]["network"][
            "remove_global_mean_surface_temperature"
        ] = combo["remove_gmst"]
        config["stepper"]["step"]["config"]["residual_prediction"] = combo[
            "residual_pred"
        ]
        config["logging"]["project"] = "GMST_ResPred"
        config["logging"]["entity"] = "ai2cm"

        output_path = SCRIPT_DIR / f"norm-{combo['name']}.yaml"
        with open(output_path, "w") as f:
            if header_lines:
                f.write("\n".join(header_lines) + "\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Written {output_path}")


if __name__ == "__main__":
    main()
