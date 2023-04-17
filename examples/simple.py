import datetime
import logging

import matplotlib.pyplot as plt

# need to import pandas first sometimes to avoid glibc error
import pandas  # noqa
import torch

import networks
from fcn_mip import geometry, initial_conditions, schema

logging.basicConfig(level=logging.INFO)

device = "cuda:0"
model = networks.get_model("sfno_73ch").to(device)
time = datetime.datetime(2018, 1, 1)

ds = initial_conditions.ic(
    time,
    model.grid,
    n_history=model.n_history,
    channel_set=model.channel_set,
    source=schema.InitialConditionSource.era5,
)

x = torch.from_numpy(ds.values).to(device=device)
# need a batch dimension of length 1
x = x[None, :, model.channels]


for k, data in enumerate(model.run_steps(x, 40)):
    print(f"Step {k} done")
    tcwv = geometry.sel_channel(model, ds.channel, data, ["tcwv"])
    tcwv = tcwv.cpu().numpy()
    plt.clf()
    plt.imshow(tcwv[0, 0])
    plt.colorbar(orientation="horizontal")
    plt.savefig(f"{k:06d}.png")
