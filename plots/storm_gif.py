import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import json
import argparse
import matplotlib.animation as animation


def main():
    argparse.ArgumentParser(description="Flip maps")
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", type=str, default="z850")
    parser.add_argument("--ensemble-memeber", type=int, default=0)
    parser.add_argument("--path", type=str)

    variable = parser.parse_args().variable
    ens_m = parser.parse_args().ensemble_memeber
    full_path = parser.parse_args().path
    config = json.loads(open(full_path + "config.in").read())
    forecast_name = config["forecast_name"]

    data = nc.Dataset(full_path + "/ensemble_out.nc", "r")
    imin = data.variables["imin"][0]
    imax = data.variables["imax"][0]
    jmin = data.variables["jmin"][0]
    jmax = data.variables["jmax"][0]

    land_mask_global = np.load("./land_sea_mask.npy")
    land_mask = land_mask_global[imin:imax, jmin:jmax]

    time = data.variables["time"] - data.variables["time"][0]
    lat = data.variables["lat"]
    lon = data.variables["lon"]
    if variable == "V":
        u850 = data.variables["u850"]
        v850 = data.variables["v850"]
        var = np.sqrt(np.square(u850) + np.square(v850))
    elif variable == "warm_core":
        z850 = data.variables["z850"]
        z250 = data.variables["z250"]
        var = np.subtract(z250, z850)
    else:
        var = data.variables[variable]
    [Lat, Lon] = np.meshgrid(lon, lat)
    fig, ax = plt.subplots()

    def animate(j):
        ax.clear()
        plt.title(variable + " at " + str(int(time[j])) + " hours")
        ax.contourf(Lat, Lon, var[ens_m, j, :, :], cmap="RdBu_r")
        ax.contour(Lat, Lon, np.subtract(1.0, land_mask), [0.25], cmap="gray")

    ani = animation.FuncAnimation(
        fig, animate, np.shape(time)[0], interval=50, blit=False
    )
    writergif = animation.PillowWriter(fps=5)
    ani.save(full_path + forecast_name + ".gif", writer=writergif)

    return


if __name__ == "__main__":
    main()
