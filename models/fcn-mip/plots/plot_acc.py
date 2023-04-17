import xarray
import numpy as np
import matplotlib.pyplot as plt

import argparse


def plot_with_timedelta(x, dim="lead_time", **kwargs):
    dt = np.timedelta64(1, "h")
    y = x.assign_coords({dim: x[dim] / dt})
    y[dim].attrs["units"] = "hours"
    y.plot(**kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", required=True, type=str)
parser.add_argument("--variable", "-v", default="z500", type=str)
parser.add_argument("files", nargs="+")
args = parser.parse_args()


data = {}
for f in args.files:
    ds = xarray.open_dataset(f)
    var = ds.acc.sel(channel=args.variable)
    data[ds.attrs["model"]] = var


for key in data:
    plot_with_timedelta(data[key], label=key)

plt.legend()
plt.savefig(args.output)
