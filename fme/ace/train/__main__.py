from fme.ace.train.train import main
from fme.core.cli import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.yaml_config, override_dotlist=args.override, h_parallel_size =args.h_parallel_size , w_parallel_size=args.w_parallel_size )
