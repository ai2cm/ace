import pathlib
import xarray
import matplotlib.pyplot as plt
import numpy as np

path = pathlib.Path(
    """/lustre/fsw/nvresearch/yacohen/ensemble_out/hackathon_tests/
        EastCoast_sOutput.hafno_baseline_26ch_edim512_mlp2.EastCoast.08_01_00_00_00/"""
)
ensemble_files = list(path.glob("*_?.nc"))
ds = xarray.concat(
    [xarray.open_dataset(f, group="Somewhere") for f in ensemble_files], dim="ensemble"
)
template = xarray.open_dataset(ensemble_files[0])
ds = ds.assign_coords(time=template.time)
print(np.shape(ds.u10))
point = ds.isel(npoints=1, time=100)
plt.figure()
point.tcwv.plot.hist(bins="auto")
plt.xlabel("total column water vapor [g/kg]")
plt.title("Histogram of 1000 members in miami")
plt.ylabel("PDF")
plt.savefig("figures/tcwv_distribution_1000_miami.pdf", format="pdf")

plt.figure()
point.tcwv.isel(ensemble=slice(0, 50)).plot.hist(bins="auto")
plt.title("Histogram of 50 members in miami [g/kg]")
plt.xlabel("total column water vapor [g/kg]")
plt.ylabel("PDF")
plt.savefig("figures/tcwv_distribution_50_miami.pdf", format="pdf")

plt.figure()
point.t2m.plot.hist(bins="auto")
plt.xlabel("2m temperature [K]")
plt.title("Histogram of 1000 members in miami")
plt.ylabel("PDF")
plt.savefig("figures/t2m_distribution_1000_miami.pdf", format="pdf")

plt.figure()
point.t2m.isel(ensemble=slice(0, 50)).plot.hist(bins="auto")
plt.title("Histogram of 50 members in miami")
plt.xlabel("2m temperature [K]")
plt.ylabel("PDF")
plt.savefig("figures/t2m_distribution_50_miami.pdf", format="pdf")

V = np.sqrt(np.add(np.power(point.u10, 2), np.power(point.v10, 2)))

plt.figure()
V.plot.hist(bins="auto")
plt.xlabel("10m wind speed [m/s]")
plt.title("Histogram of 1000 members in miami")
plt.ylabel("PDF")
plt.savefig("figures/V_distribution_1000_miami.pdf", format="pdf")

plt.figure()
V.isel(ensemble=slice(0, 50)).plot.hist(bins="auto")
plt.title("Histogram of 50 members in miami")
plt.xlabel("10m wind speed [m/s]")
plt.ylabel("PDF")
plt.savefig("figures/V_distribution_50_miami.pdf", format="pdf")
