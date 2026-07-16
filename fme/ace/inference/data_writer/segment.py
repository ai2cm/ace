import dataclasses


@dataclasses.dataclass(frozen=True)
class SegmentContext:
    """
    Runtime description of the current segment of a segmented inference run,
    used by data writers that accumulate a single whole-run output across
    segments (e.g. zarr stores written via region writes).

    Parameters:
        segment_index: Zero-based index of the current segment.
        total_segments: Total number of segments in the run.
        run_dir: The root experiment directory shared by all segments.
        segment_dir: The current segment's output directory.
        previous_segment_dir: The previous segment's output directory, or None
            for the first segment. Stateful writers restore their accumulator
            snapshots from here so that re-running a partially completed
            segment does not double-count.
    """

    segment_index: int
    total_segments: int
    run_dir: str
    segment_dir: str
    previous_segment_dir: str | None

    def __post_init__(self):
        if self.segment_index < 0 or self.segment_index >= self.total_segments:
            raise ValueError(
                f"segment_index {self.segment_index} out of range for "
                f"{self.total_segments} segments"
            )
        if self.segment_index > 0 and self.previous_segment_dir is None:
            raise ValueError(
                "previous_segment_dir is required for segments after the first"
            )
