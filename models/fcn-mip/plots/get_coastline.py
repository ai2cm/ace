import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="50m")


# get lat and lon coordinates of the coastline
x = []
y = []
for line in cfeature.COASTLINE.geometries():
    for point in line.coords:
        x.append(point[0])
        y.append(point[1])

print(max(x), min(x), max(y), min(y))
# loop over x,y pairs
west_coast_x = []
west_coast_y = []

for i in range(len(x)):

    if x[i] > -160 and x[i] < -120 and y[i] > 30 and y[i] < 60:

        west_coast_x.append(x[i])
        west_coast_y.append(y[i])


ax.scatter(west_coast_x, west_coast_y, transform=ccrs.PlateCarree(), color="red", s=5)
ax.scatter(x, y, transform=ccrs.PlateCarree(), color="blue", s=0.1)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.gridlines(draw_labels=True)
plt.savefig("coastline.png")

# transform the coordinates from long [-180, 180] to [0, 360]
west_coast_x = [i + 360 if i < 0 else i for i in west_coast_x]
x = [i + 360 if i < 0 else i for i in x]
np.save("west_coast_x.npy", west_coast_x)
np.save("west_coast_y.npy", west_coast_y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.scatter(west_coast_x, west_coast_y, color="red", s=5)
ax.scatter(x, y, color="blue", s=0.1)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid()
plt.savefig("coastline2.png")
