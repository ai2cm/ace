import argparse

from .inference import main

parser = argparse.ArgumentParser()
parser.add_argument("yaml_config", type=str)
args = parser.parse_args()
main(yaml_config=args.yaml_config)
