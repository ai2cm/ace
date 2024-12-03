import argparse

import torch


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path for input checkpoint.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Where to save checkpoint without optimizer state.",
    )
    parser.add_argument("--strip-optimization", action="store_true")
    parser.add_argument("--cast-coords-to-float32", action="store_true")
    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()
    checkpoint = torch.load(args.input_path, map_location=torch.device("cpu"))
    if args.strip_optimization:
        del checkpoint["optimization"]
    if args.cast_coords_to_float32:
        if "area" in checkpoint["stepper"]:
            checkpoint["stepper"]["area"] = checkpoint["stepper"]["area"].float()
        if "vertical_coordinate" in checkpoint["stepper"]:
            coords = checkpoint["stepper"]["vertical_coordinate"]
            coords["ak"] = coords["ak"].float()
            coords["bk"] = coords["bk"].float()
    torch.save(checkpoint, args.output_path)
