import datetime
import tempfile
import json
import logging
import os
import typer


logging.basicConfig(
    format="%(asctime)s:%(levelname)-s:%(name)s:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__file__)

WD = os.getcwd()


def run(time: datetime.datetime, model: str, output: str, test=False):
    day = 4
    week = 7 * day
    with tempfile.TemporaryDirectory() as d:
        obj = {
            "ensemble_members": 8,
            "noise_amplitude": 0.001,
            "simulation_length": day if test else 8 * week,
            "weather_event": {
                "properties": {"name": "ECWMFValid", "start_time": time.isoformat()},
                "domains": [
                    {
                        "name": "global",
                        "type": "Window",
                        "diagnostics": [
                            {"type": "raw", "channels": ["tcwv", "t2m", "z500"]}
                        ],
                    }
                ],
            },
            "output_path": d,
            "output_frequency": 1,
            "fcn_model": model,
            "seed": 12345,
            "use_cuda_graphs": False,
            "ensemble_batch_size": 8,
            "autocast_fp16": False,
            "perturbation_strategy": "gaussian",
            "noise_reddening": 2,
        }
        with open(f"{d}/c.json", "w") as f:
            json.dump(obj, f)
        assert not os.system(
            f"torchrun --nproc_per_node 8 bin/inference_ensemble.py {d}/c.json"
        )
        assert not os.system(f"ncra -O {d}/ensemble_*.nc {output}")


def main(
    year: int,
    model: str,
    root: str = "hindcast",
    test: bool = False,
):
    """
    Args:
        root: the root directory of the output
    """
    time = datetime.datetime(year, 1, 1)
    frequency = datetime.timedelta(weeks=1)
    times = []

    while time.year == year:
        times.append(time)
        time = time + frequency

    if test:
        times = times[:2]

    for count, initial_time in enumerate(times):
        start = time.now()
        output = f"{root}/{model}"
        output_file = f"{initial_time.isoformat()}.nc"
        os.makedirs(output, exist_ok=True)
        run(initial_time, model, os.path.join(output, output_file))
        stop = time.now()
        elapsed = stop - start
        eta = (len(times) - count) * elapsed
        logger.info(
            f"{initial_time} done. Elapsed: {elapsed.total_seconds()}s. "
            f"Time-remaining: {eta}"
        )


if __name__ == "__main__":
    typer.run(main)
