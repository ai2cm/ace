import pathlib
import tempfile

import pytest
import torch

from fme.core.benchmark.run import _json_default, main


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


def test_json_default():
    assert _json_default(torch.tensor(1)) == 1
    assert _json_default(torch.tensor(1.0)) == 1.0
    assert _json_default(torch.ones((2, 2))) == [[1.0, 1.0], [1.0, 1.0]]
