import pathlib
import tempfile

from fme.core.benchmark.run import main


def test_run():
    # Just test that the main function runs without error on a simple benchmark
    # We don't care about the output here, just that it completes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        main(
            name="csfno_block",  # just one for speed
            iters=1,
            output_dir=output_dir,
            child=None,
        )
