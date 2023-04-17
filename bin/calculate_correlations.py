import torch
import h5py
import numpy as np
from tqdm import tqdm

# Load means and stds for normalisation
mean_path = "/lustre/fsw/sw_climate_fno/34Vars/stats/global_means.npy"
std_path = "/lustre/fsw/sw_climate_fno/34Vars/stats/global_stds.npy"
means = torch.tensor(np.load(mean_path)[0, :34, 0, 0])
stds = torch.tensor(np.load(std_path)[0, :34, 0, 0])

# Load last 6 years in training set to calculate inter-variable correlation matrices
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

# Calculate correlation matrix for each spatial position and save to disk
corr = torch.zeros(720, 1440, 34, 34)
for i in tqdm(range(720)):
    for j in range(1440):
        corr[i, j] = torch.corrcoef(data[:, :, i, j].permute(1, 0))

torch.save(corr, "/lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth")
