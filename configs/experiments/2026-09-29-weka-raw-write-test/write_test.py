"""Raw filesystem write throughput test, independent of zarr and ACE code.

Each stream is a separate process writing shard-sized files with plain os.write,
the same call path the ACE zarr writer's LocalStore takes, so the result is the
filesystem's ceiling for our write pattern on this node. Prints one summary line
per configuration; nothing else is recorded.
"""

import argparse
import multiprocessing
import os
import shutil
import socket
import time
import uuid

BLOCK_BYTES = 8 * 2**20


def _write_files(
    directory: str, n_files: int, file_bytes: int, fsync: bool, result_queue
):
    os.makedirs(directory, exist_ok=True)
    block = os.urandom(BLOCK_BYTES)
    start = time.perf_counter()
    for i in range(n_files):
        fd = os.open(os.path.join(directory, f"{i:05d}.bin"), os.O_WRONLY | os.O_CREAT)
        remaining = file_bytes
        while remaining > 0:
            remaining -= os.write(fd, block[: min(BLOCK_BYTES, remaining)])
        if fsync:
            os.fsync(fd)
        os.close(fd)
    result_queue.put(time.perf_counter() - start)


def run_streams(
    root: str, n_streams: int, n_files: int, file_bytes: int, fsync: bool
) -> tuple[float, float]:
    """Return aggregate MB/s and the slowest stream's MB/s."""
    run_dir = os.path.join(root, f"streams{n_streams}-{uuid.uuid4().hex[:6]}")
    queue: multiprocessing.Queue = multiprocessing.Queue()
    procs = [
        multiprocessing.Process(
            target=_write_files,
            args=(os.path.join(run_dir, f"s{s}"), n_files, file_bytes, fsync, queue),
        )
        for s in range(n_streams)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    stream_seconds = [queue.get() for _ in procs]
    shutil.rmtree(run_dir, ignore_errors=True)
    stream_bytes = n_files * file_bytes
    aggregate = n_streams * stream_bytes / max(stream_seconds) / 1e6
    slowest = stream_bytes / max(stream_seconds) / 1e6
    return aggregate, slowest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory to write into; created if missing.")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--file-mb", type=float, default=26.0, help="ACE shard size.")
    parser.add_argument("--gb-per-config", type=float, default=6.0)
    parser.add_argument("--fsync", action="store_true", help="fsync every file.")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    file_bytes = int(args.file_mb * 2**20)
    host = socket.gethostname()
    print(f"host={host} root={args.root} label={args.label} fsync={args.fsync}")
    for n_streams in args.streams:
        n_files = max(1, int(args.gb_per_config * 2**30 / file_bytes / n_streams))
        aggregate, slowest = run_streams(
            args.root, n_streams, n_files, file_bytes, args.fsync
        )
        print(
            f"label={args.label} streams={n_streams} files_per_stream={n_files} "
            f"file_mb={args.file_mb:g} aggregate_mb_per_s={aggregate:.0f} "
            f"slowest_stream_mb_per_s={slowest:.0f}"
        )


if __name__ == "__main__":
    main()
