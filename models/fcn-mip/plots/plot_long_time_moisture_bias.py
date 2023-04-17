import sys
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as crs
import datetime
import pathlib

from fcn_mip import initial_conditions, schema


case = sys.argv[1]
output = pathlib.Path(sys.argv[2])
output.mkdir(exist_ok=True)

root = xarray.open_dataset(case)
ds = xarray.open_dataset(case, group="global").assign_coords(root.coords)

# %%
time = datetime.datetime(2018, 1, 1)
verification = initial_conditions.open_era5_xarray(
    time, channel_set=schema.ChannelSet.var34
)


def plot(ax, z, **kwargs):
    im = z.plot(
        ax=ax,
        x="lon",
        y="lat",
        transform=crs.PlateCarree(),
        add_colorbar=False,
        **kwargs,
    )
    plt.colorbar(im, ax=ax, orientation="horizontal")
    ax.coastlines()


def map_bias(
    pred_avg, average, kw=dict(vmin=0, vmax=60, cmap="viridis"), diff_scale=30
):
    fig = plt.figure(figsize=(13, 3))
    start, stop = ds.time[[0, -1]].dt.strftime("%Y-%m-%d").values.tolist()
    fig.suptitle(
        f"{pred_avg.long_name} ({pred_avg.units}). Average period={start}--{stop}"
    )
    projection = crs.PlateCarree()

    ax = fig.add_subplot(131, projection=projection)
    plot(ax, average, **kw)
    ax.set_title("Truth")

    ax = fig.add_subplot(132, projection=projection)
    plot(ax, pred_avg, **kw)
    ax.set_title("Prediction")

    ax = fig.add_subplot(133, projection=projection)
    z = pred_avg - average
    plot(ax, z, vmin=-diff_scale, vmax=diff_scale, cmap="RdBu_r")
    ax.set_title("Bias")


average = verification.sel(time=ds.time).sel(channel="tcwv").mean("time").load()
pred_avg = ds.mean("time").tcwv.assign_attrs(
    long_name="total column water vapor", units="mm"
)
map_bias(pred_avg, average)
filename = output / (pred_avg.name + ".png")
plt.savefig(filename)

# %%
average = verification.sel(time=ds.time).sel(channel="t2m").mean("time").load()
pred_avg = ds.mean("time").t2m.assign_attrs(
    long_name="Two-meter temperature", units="K"
)
map_bias(pred_avg, average, kw={}, diff_scale=50)
filename = output / (pred_avg.name + ".png")
plt.savefig(filename)
