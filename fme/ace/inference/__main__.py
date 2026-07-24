from fme.core.cli import get_parser
from fme.core.distributed.distributed import Distributed

from .inference import DEFAULT_SEGMENT_LABEL_FORMAT, main

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "--segments",
        type=int,
        default=None,
        help=(
            "If provided, number of times to repeat the inference in time, "
            "saving each segment in a separate folder labeled by the start "
            "time of its first (or only) ensemble member. "
            "WARNING: this feature is experimental and its API is subject "
            "to change."
        ),
    )
    parser.add_argument(
        "--segment-label-format",
        type=str,
        default=DEFAULT_SEGMENT_LABEL_FORMAT,
        help=(
            "strftime format used to render each segment's start time into its "
            "folder/wandb-run label. Only used when --segments is provided. "
            "Defaults to hour precision; pass a more precise format (e.g. "
            "'segment_%%Y%%m%%dT%%H%%M%%S') if the timestep or initial "
            "condition time require it."
        ),
    )
    args = parser.parse_args()
    with Distributed.context():
        main(
            args.yaml_config,
            segments=args.segments,
            override_dotlist=args.override,
            segment_label_format=args.segment_label_format,
        )
