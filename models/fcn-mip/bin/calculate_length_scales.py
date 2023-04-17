# flake8: noqa
import torch
import h5py
import numpy as np
from tqdm import tqdm

# Load means and stds for normalisation
mean_path = "/lustre/fsw/sw_climate_fno/34Vars/stats/global_means.npy"
std_path = "/lustre/fsw/sw_climate_fno/34Vars/stats/global_stds.npy"
means = torch.tensor(np.load(mean_path)[0, :34, 0, 0])
stds = torch.tensor(np.load(std_path)[0, :34, 0, 0])
# Load last 6 years in training set
years = [2015, 2014, 2013, 2012, 2011, 2010]
data = []
for year in years:
    if year <= 2015:
        f = h5py.File(f"/lustre/fsw/sw_climate_fno/34Vars/train/{year}.h5", "r")
    elif year < 2018:
        f = h5py.File(f"/lustre/fsw/sw_climate_fno/34Vars/test/{year}.h5", "r")
    elif year == 2018:
        f = h5py.File(f"/lustre/fsw/sw_climate_fno/34Vars/out_of_sample/{year}.h5", "r")
    obs = torch.tensor(
        f["fields"][::100, :34, :720]
    )  # Only use every 100th timestep for memory efficiency
    obs -= means[:, None, None]
    obs /= stds[:, None, None]
    data.append(obs)

data = torch.cat(data)
print(data.shape)
n_samples = data.shape[0]

# Get [140*140, 140*140]-dimensional tensor with spatial correlations for given length-scale
def k(length_scale):
    coords = torch.stack(
        [torch.tensor(x, dtype=float) for x in np.ndindex(140, 140)], dim=1
    ).cuda()
    d = torch.linalg.norm(coords[..., None] - coords[..., None, :], dim=0)
    corrs = torch.exp(-d / (2 * length_scale))
    return corrs


length_scales = []
for c in range(34):

    length_scale = torch.tensor([30.0], requires_grad=True, device="cuda")
    optimizer = torch.optim.Adam([length_scale], lr=1)

    # Calculate spatial correlations between positions in data (just subset in center for efficiency)
    s = data[:, c, 290:-290, 650:-650]
    s = s.reshape(n_samples, -1).permute(1, 0).cuda()
    corr = torch.corrcoef(s)

    # Optimize length-scale to match correlations
    for i in range(200):
        optimizer.zero_grad()
        loss = torch.mean((k(length_scale) - corr) ** 2)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(int(length_scale.item()), loss.item())

    length_scales.append(length_scale.item())

torch.save(
    torch.tensor(length_scales),
    "/lustre/fsw/sw_climate_fno/ensemble_init_stats/length_scales.pth",
)
