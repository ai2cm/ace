import xarray as xr
import os


DATA_PATH = '/Users/oliverwm/scratch/fv3gfs-fourcastnet/fourcastnet_vanilla_1_degree'
OUTPUT_PATH = '/Users/oliverwm/scratch/fv3gfs-fourcastnet/stats'
os.makedirs(OUTPUT_PATH, exist_ok=True)

ds = xr.open_mfdataset(os.path.join(DATA_PATH, '*.nc'), chunks=None)
with xr.set_options(keep_attrs=True):
    global_means = ds.mean(dim=['time', 'grid_xt', 'grid_yt'])
    global_stds = ds.std(dim=['time', 'grid_xt', 'grid_yt'])

global_means.to_netcdf(os.path.join(OUTPUT_PATH, 'fv3gfs-mean.nc'))
global_stds.to_netcdf(os.path.join(OUTPUT_PATH, 'fv3gfs-stddev.nc'))

print("means: ", global_means)
print("stds: ", global_stds)
