from fme.core.cli import get_parser

from .inference import main

parser = get_parser()
args = parser.parse_args()
main(args.yaml_config, override_dotlist=args.override)
