import argparse

from fme.ace.train.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(yaml_config=args.yaml_config)
