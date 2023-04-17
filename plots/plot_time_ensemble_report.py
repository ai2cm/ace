import sys
import xarray
import numpy as np
import matplotlib.pyplot as plt
import datapane
import ensemble

case = sys.argv[1]
output = sys.argv[2]

ds = ensemble.open(path=case)
ds["speed10"] = np.sqrt(ds.u10**2 + ds.v10**2)
ds["speed10"].attrs.update({"units": "m/s", "long_name": "wind speed at 10 m"})
cities = ["NY", "Miami", "Boston", "San Juan", "Houston", "DC"]

ds = ds.assign(city=xarray.DataArray(cities, dims=["npoints"])).set_index(
    npoints="city"
)
saffir_simpson_scale_kmph = np.array([119, 154, 178, 209, 252])


def get_hurricane_category(x_m_per_s):
    x_kph = np.asarray(x_m_per_s) * 3600 / 1000
    return np.searchsorted(saffir_simpson_scale_kmph, x_kph)


get_hurricane_category([0])


def plot_ensemble_wind_performance(speed_ms):

    factor = 2.23694
    speed = speed_ms * factor
    z = speed.assign_attrs(units="mph", long_name="10 m wind speed")

    fig, (a, b) = plt.subplots(nrows=2, sharex=True, sharey=True)

    a.plot(ds.time, z.T, alpha=0.1, color="blue")
    plt.plot(ds.time, z[:50].T, alpha=0.1, color="blue")

    line = z.quantile(0.999)
    a.axhline(line, color="k")
    a.set_title(f"n={speed_ms.sizes['ensemble']}", loc="left")
    b.set_title("n=50", loc="left")
    b.axhline(line, color="k")
    a.grid()
    b.grid()

    fig.suptitle(f"{z.long_name} ({z.units}) at {z.npoints.item()}")
    plt.xticks(rotation=45)


plot_ensemble_wind_performance(ds.sel(npoints="NY").speed10)


def plot_ensemble_temperature_performance(temp):

    celcius = temp - 273.15
    f = celcius * 9 / 5 + 32
    z = f.assign_attrs(units="deg F", long_name="2-meter temperature")

    fig, (a, b) = plt.subplots(nrows=2, sharex=True, sharey=True)

    a.plot(temp.time, z.T, alpha=0.1, color="blue")
    plt.plot(temp.time, z[:50].T, alpha=0.1, color="blue")

    a.set_title(f"n={z.sizes['ensemble']}", loc="left")
    b.set_title("n=50", loc="left")
    a.grid()
    b.grid()

    fig.suptitle(f"{z.long_name} ({z.units}) at {z.npoints.item()}")
    plt.xticks(rotation=45)
    return a


k = 0


def plot(label, fig=None):
    global k
    fig = fig or plt.gcf()
    fname = f"{k}.png"
    fig.savefig(fname)
    plt.close(fig)
    k += 1
    return datapane.Media(file=fname, label=label)


winds = []
for point in ds.npoints:
    plot_ensemble_wind_performance(ds.sel(npoints=point).speed10)
    winds.append(plot(label=point.item()))

winds = datapane.Select(blocks=winds, label="Wind Speed")

temp = []
for point in ds.npoints:
    a = plot_ensemble_temperature_performance(ds.sel(npoints=point).t2m)
    a.set_ylim(bottom=20, top=80)
    temp.append(plot(label=point.item()))

temp = datapane.Select(blocks=temp, label="Temperature 2 M")

app = datapane.App(blocks=[datapane.Select(blocks=[winds, temp])])
app.save(output)
