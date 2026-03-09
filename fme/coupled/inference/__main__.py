from fme.core.cli import get_parser
from fme.core.distributed.distributed import Distributed

from .inference import main

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    with Distributed.context():
        main(args.yaml_config, override_dotlist=args.override)
