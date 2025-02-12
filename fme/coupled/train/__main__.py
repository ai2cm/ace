from fme.core.cli import get_parser
from fme.coupled.train.train import main

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(yaml_config=args.yaml_config, override_dotlist=args.override)
