import pathlib
import tempfile

import pytest
import torch

from fme.core.benchmark.run import main


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_run():
    # Just test that the main function runs without error on a simple benchmark
    # We don't care about the output here, just that it completes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        main(
            benchmark_name="csfno_block",  # just one for speed
            iters=1,
            output_dir=output_dir,
            child=None,
        )
