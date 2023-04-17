import sys
import xarray
import metadata
import matplotlib.pyplot as plt


def plot_tcwv_weekly(ds):
    tcwv = ds.fields.sel(channel="tcwv")
    tcwv.attrs["long_name"] = "total column water vapor"
    tcwv.attrs["units"] = "mm"
    tcwv.resample(time="7D").nearest().plot(col="time", col_wrap=3, vmax=80, vmin=0)


zarr_file = sys.argv[1]
output = sys.argv[2]

ds = xarray.open_zarr(zarr_file).assign_coords(channel=metadata.channels[:26])
plot_tcwv_weekly(ds)
plt.savefig(output)
