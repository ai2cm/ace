"""Plot TempestExtremes TC tracks on a global map (matplotlib + cartopy).

Reads a tracks.csv written by detect_tc_tracks.py (columns track_id, time, lon,
lat, slp, wind; lon in 0-360, slp in Pa) and draws each track as a polyline
colored by its peak 10m wind, with a genesis marker. Output PNG goes to the
scratch directory so it does not pollute git.

Usage:
    python scratch/plot_tc_tracks.py scratch/tc_2013/tracks.csv scratch/tc_2013/tracks_map.png
"""

import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

csv_path = sys.argv[1] if len(sys.argv) > 1 else "scratch/tc_2013/tracks.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "scratch/tc_2013/tracks_map.png"

df = pd.read_csv(csv_path, parse_dates=["time"])
n_tracks = df["track_id"].nunique()
y0, y1 = int(df["time"].dt.year.min()), int(df["time"].dt.year.max())
year = f"{y0}" if y0 == y1 else f"{y0}–{y1}"

# Color each whole track by its peak (max) 10m wind.
peak_wind = df.groupby("track_id")["wind"].max()
norm = Normalize(vmin=0.0, vmax=float(np.ceil(peak_wind.max())))
cmap = plt.get_cmap("plasma")

proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#e9e6df", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#dCe7ef", zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555555", zorder=1)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
gl.top_labels = gl.right_labels = False

data_crs = ccrs.PlateCarree()
for tid, t in df.groupby("track_id"):
    t = t.sort_values("time")
    lon = t["lon"].to_numpy()
    lat = t["lat"].to_numpy()
    # Break the polyline at dateline crossings so segments don't streak across
    # the whole map when consecutive longitudes jump ~360 deg.
    lon_plot = lon.copy()
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]
    lon_plot = lon_plot.astype(float)
    lat_plot = lat.astype(float)
    for j in jumps:
        lon_plot = np.insert(lon_plot, j + 1, np.nan)
        lat_plot = np.insert(lat_plot, j + 1, np.nan)
    color = cmap(norm(float(peak_wind[tid])))
    # Thin lines and fade / drop genesis markers as the track count grows, so
    # the map stays legible from ~100 to ~10k tracks.
    if n_tracks <= 150:
        lw, ms, alpha = 1.3, 3.5, 0.9
    elif n_tracks <= 2000:
        lw, ms, alpha = 0.7, 1.8, 0.75
    else:
        # Low alpha so overlapping tracks accumulate into a visible density.
        lw, ms, alpha = 0.4, 0.0, 0.18
    ax.plot(
        lon_plot, lat_plot, color=color, linewidth=lw, alpha=alpha,
        transform=data_crs, zorder=2,
    )
    # Genesis marker (first point); skipped for very large sets (ms == 0).
    if ms > 0:
        ax.plot(
            lon[0], lat[0], marker="o", markersize=ms, color=color,
            markeredgecolor="black", markeredgewidth=0.25,
            transform=data_crs, zorder=3,
        )

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
cbar.set_label("Peak 10 m wind (m s$^{-1}$)")

ax.set_title(
    f"TempestExtremes TC tracks — X-SHiELD 1° AMIP, {year}  "
    f"({n_tracks} tracks, {len(df)} points; dots = genesis)",
    fontsize=13,
)
fig.canvas.draw()
fig.savefig(out_path, dpi=150)
print(f"Wrote {out_path}  ({n_tracks} tracks, {len(df)} points)")
