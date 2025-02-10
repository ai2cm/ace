import argparse

from .inference import main

parser = argparse.ArgumentParser()
parser.add_argument("yaml_config", type=str)
parser.add_argument(
    "--segments",
    type=int,
    default=None,
    help="If provided, number of times to repeat the inference in time, saving each "
    "segment in a separate folder labeled as 'segment_0000', 'segment_0001' etc. "
    "WARNING: this feature is experimental and its API is subject to change.",
)
args = parser.parse_args()
main(yaml_config=args.yaml_config, segments=args.segments)
