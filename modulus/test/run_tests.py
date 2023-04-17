import GPUtil
import os
import pytest
import argparse
from pytest import ExitCode
from termcolor import colored

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--testdir", default=".")
    args = parser.parse_args()

    os.system("nvidia-smi")
    availible_gpus = GPUtil.getAvailable(limit=8)
    if len(availible_gpus) == 0:
        print(colored(f"No free GPUs found on DGX 4850", "red"))
        raise RuntimeError()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(availible_gpus[-1])
        print(colored(f"=== Using GPU {availible_gpus[-1]} ===", "blue"))

    retcode = pytest.main(["-x", args.testdir])

    if ExitCode.OK == retcode:
        print(colored("UNIT TESTS PASSED! :D", "green"))
    else:
        print(colored("UNIT TESTS FAILED!", "red"))
        raise ValueError(f"Pytest returned error code {retcode}")
