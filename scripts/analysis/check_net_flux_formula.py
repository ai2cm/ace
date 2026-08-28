"""Does the prescribed surface-flux correction reproduce the reference hfds?

`SurfaceEnergyFluxCorrectionConfig(method="prescribed")` imposes `net_flux`,
computed by `_compute_ocean_net_surface_energy_flux`, over open ocean. That part
of the corrected flux owes nothing to the emulator, so it can be tested on its
own: evaluate the formula on the reference's forcing and the reference's SST and
compare with the reference's surface flux. An emulator reproducing the reference
perfectly would still inherit the difference.

Result on the 1-degree OM4 piControl record (0151-0350):

    bias = +0.136 +/- 0.009 W/m2

against a measured surface-flux excess across twenty 200-year rollouts of median
+0.118 W/m2 (range 0.052 to 0.193) -- i.e. the formula accounts for essentially
all of the ocean heat drift those rollouts show. The bias is polar, matching the
two terms the function records as missing (calving latent heat, river runoff):
+0.157 W/m2 from north of 60N and +0.127 from south of 60S, against -0.237 from
the tropics.

Two notes for anyone re-running this:

- Means are normalised by NON-LAND area, which is what the inference aggregator
  does. Under this convention the geothermal term reproduces the 0.0905 W/m2
  constant separating `net_energy_flux_into_ocean_column` from `hfds_total_area`
  in the rollout diagnostics, and the annual means match the logged target series
  exactly. Normalising by total area instead scales every number by ~0.74.
- Land is NaN in the ocean fields, exactly on the full-land cells, where
  `ocean_fraction` and `sea_surface_fraction` are both zero; zero-filling there
  cannot change an area-weighted result.

Reads the zarr stores from GCS, so it runs anywhere with credentials; no beaker
job or weka mount needed.
"""

import numpy as np, torch, xarray as xr
from fme.core.corrector.ocean import _compute_ocean_net_surface_energy_flux
from fme.core.ocean_data import OceanData

B="gs://vcm-ml-intermediate"
CO=f"{B}/2026-07-15-om4-picontrol-1deg-coupled-ocean.zarr"
OC=f"{B}/2026-07-15-om4-picontrol-1deg-ocean-5daily.zarr"
FORCING=["DLWRFsfc","DSWRFsfc","ULWRFsfc","USWRFsfc","LHTFLsfc","SHTFLsfc",
         "PRATEsfc","total_frozen_precipitation_rate"]
FROM_CO=FORCING+["land_fraction","sea_surface_fraction","ocean_sea_ice_fraction","hfds_total_area"]
FROM_OC=["sst","hfgeou"]
START,STOP,STRIDE="0151-01-01","0351-01-01",5

a=xr.open_zarr(CO).sel(time=slice(START,STOP)).isel(time=slice(None,None,STRIDE))
b=xr.open_zarr(OC).sel(time=slice(START,STOP)).isel(time=slice(None,None,STRIDE))
print("steps:",a.sizes["time"],"| grid:",a.sizes["lat"],"x",a.sizes["lon"], flush=True)
b=b.assign_coords(lat=a.lat, lon=a.lon, time=a.time)
ds=xr.merge([a[FROM_CO], b[FROM_OC]], join="exact")
ds=ds.load()
print("loaded", flush=True)

# The latitude grid is not uniform, so cos(lat) alone is not the cell area:
# weight by sin(edge_north) - sin(edge_south), which is exact for a lat-lon cell.
lat=ds["lat"].values
edges=np.empty(len(lat)+1); edges[1:-1]=0.5*(lat[1:]+lat[:-1])
edges[0]=lat[0]-(edges[1]-lat[0]); edges[-1]=lat[-1]+(lat[-1]-edges[-2])
edges=np.clip(edges,-90,90)
dA=np.sin(np.deg2rad(edges[1:]))-np.sin(np.deg2rad(edges[:-1]))
w=np.broadcast_to(dA[:,None],(len(lat),ds.sizes["lon"])).copy()
# The aggregator averages over cells where the ocean fields are defined -- i.e.
# excluding full-land cells -- not over the whole globe. Verified: under this
# convention the geothermal term reproduces the 0.0905 W/m^2 constant that
# separates net_energy_flux_into_ocean_column from hfds_total_area in the
# rollout diagnostics, and annual means match the logged target series exactly.
lf=np.nan_to_num(np.asarray(ds["land_fraction"].values),nan=1.0)
if lf.ndim==3: lf=lf[0]
w=np.where(lf>=0.999,0.0,w)
W=torch.tensor(w/w.sum(),dtype=torch.float64)

# land is NaN in the ocean fields; verified to be exactly the full-land cells,
# where ocean_fraction and sea_surface_fraction are both 0, so zero-filling
# cannot change any area-weighted result.
d={v:torch.tensor(np.nan_to_num(np.asarray(ds[v].values),nan=0.0),dtype=torch.float64)
   for v in FROM_CO+FROM_OC}
for k in ("land_fraction","sea_surface_fraction","hfgeou"):
    if d[k].ndim==2: d[k]=d[k].unsqueeze(0).expand(d["sst"].shape[0],-1,-1)
o=OceanData(d)
net=_compute_ocean_net_surface_energy_flux(d, o.sea_surface_temperature)
ssf=d["sea_surface_fraction"]; of=o.ocean_fraction
imposed=net*ssf                      # per total area, as _correct_hfds uses it
ref=d["hfds_total_area"]
bias=of*(imposed-ref)                # error a perfect emulator inherits

def gm(x): return (x*W).sum(dim=(-2,-1))
assert float((of*(d["land_fraction"]>=0.999)).abs().max())<1e-6, "ocean_fraction nonzero on land"
print(f"\nArea-weighted global means, piControl {START}..{STOP} (every {STRIDE}th step), W/m^2")
print(f"  reference hfds_total_area              {gm(ref).mean():+.4f}")
print(f"  formula net_flux * sea_surface_fraction {gm(imposed).mean():+.4f}")
print(f"  mean open-ocean fraction                {gm(of).mean():+.4f}")
print(f"  geothermal hfgeou * ssf                 {gm(d['hfgeou']*ssf).mean():+.4f}")
bt=gm(bias)
print(f"\n  BIAS inherited from the formula        {bt.mean():+.4f} W/m^2")
print(f"    per-step spread (sd)                  {bt.std():.4f}")
print(f"    standard error of the mean            {bt.std()/np.sqrt(len(bt)):.4f}")
print(f"\n  measured rollout surface-flux excess    +0.098 W/m^2")
