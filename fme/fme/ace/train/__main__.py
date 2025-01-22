import argparse

from fme.ace.train.train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str, help="Path to the YAML config file.")
    parser.add_argument(
        "--override",
        nargs="*",
        help="A dotlist of key=value pairs to override the config. "
        "For example, --override a.b=1 c=2, where a dot indicates nesting.",
    )

    args = parser.parse_args()

    main(args.yaml_config, override_dotlist=args.override)
