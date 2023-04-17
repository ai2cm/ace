import torch
import torch.distributed as dist
import config
import numpy as np
from datetime import datetime
from modulus.metrics.general.histogram import linspace, _count_bins, _update_bins_counts
from modulus.metrics.general.crps import _crps_from_counts
from modulus.metrics.climate.reduction import (
    global_mean,
    global_var,
    zonal_mean,
    zonal_var,
)
from modulus.metrics.general.ensemble_metrics import Mean, Variance
from modulus.distributed import DistributedManager
from tempfile import TemporaryDirectory
from fcn_mip.weather_events import CWBDomain, Window, MultiPoint
from fcn_mip import weather_events
from fcn_mip import filesystem
from fcn_mip import geometry
from fcn_mip.initial_conditions import ic
from fcn_mip.schema import Grid
from netCDF4._netCDF4 import Group
from typing import Union


class Diagnostics:
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        self.group, self.domain, self.grid, self.lat, self.lon = (
            group,
            domain,
            grid,
            lat,
            lon,
        )
        self.diagnostic = diagnostic
        self.device = device

        self._init_subgroup()
        self._init_dimensions()
        self._init_variables()

    def _init_subgroup(
        self,
    ):
        if self.diagnostic.type == "raw":
            self.subgroup = self.group
        else:
            self.subgroup = self.group.createGroup(self.diagnostic.type)

    def _init_dimensions(
        self,
    ):
        if self.domain.type == "MultiPoint":
            self.domain_dims = ("npoints",)
        else:
            self.domain_dims = ("lat", "lon")

    def _init_variables(
        self,
    ):
        dims = self.get_dimensions()
        dtypes = self.get_dtype()
        for channel in self.diagnostic.channels:
            if self.diagnostic.type == "histogram":
                pass
            else:
                self.subgroup.createVariable(
                    channel, dtypes[self.diagnostic.type], dims[self.diagnostic.type]
                )

    def get_dimensions(
        self,
    ):
        raise NotImplementedError

    def get_dtype(
        self,
    ):
        raise NotImplementedError

    def get_variables(
        self,
    ):
        raise NotImplementedError

    def update(
        self,
    ):
        raise NotImplementedError

    def finalize(
        self,
    ):
        raise NotImplementedError


class Raw(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

    def get_dimensions(self):
        return {"raw": ("ensemble", "time") + self.domain_dims}

    def get_dtype(self):
        return {"raw": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                output[:, c].cpu().numpy()
            )

    def finalize(self, *args):
        pass


class Absolute(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

        if self.diagnostic.function == "mean":
            self.function = global_mean
        elif self.diagnostic.function == "var":
            self.function = global_var
        else:
            raise ValueError(
                "Function type of "
                + self.diagnostic.function
                + " is not implemented for Diagnostic: Absolute"
            )

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group.createGroup("absolute_" + self.diagnostic.function)

    def get_dimensions(self):
        return {"absolute": ("ensemble", "time")}

    def get_dtype(self):
        return {"absolute": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        lat = torch.as_tensor(self.group["lat"][:]).to(self.device)
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                self.function(output[:, c], lat).cpu().numpy()
            )

    def finalize(self, *args):
        pass


class Zonal(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

        if self.diagnostic.function == "mean":
            self.function = zonal_mean
        elif self.diagnostic.function == "var":
            self.function = zonal_var
        else:
            raise ValueError(
                "Function type of "
                + self.diagnostic.function
                + " is not implemented for Diagnostic: Absolute"
            )

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group.createGroup("zonal_" + self.diagnostic.function)

    def get_dimensions(self):
        return {
            "zonal": ("ensemble", "time")
            + tuple(x for x in self.domain_dims if x != "lat")
        }

    def get_dtype(self):
        return {"zonal": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        lat = torch.as_tensor(self.group["lat"][:]).to(self.device)
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                self.function(output[:, c], lat).cpu().numpy()
            )

    def finalize(self, *args):
        pass


class Meridional(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

        if self.diagnostic.function == "mean":
            self.function = lambda x: torch.mean(x, dim=-1)
        elif self.diagnostic.function == "var":
            self.function = lambda x: torch.var(x, dim=-1)
        else:
            raise ValueError(
                "Function type of "
                + self.diagnostic.function
                + " is not implemented for Diagnostic: Absolute"
            )

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group.createGroup("meridional_" + self.diagnostic.function)

    def get_dimensions(self):
        return {
            "meridional": ("ensemble", "time")
            + tuple(x for x in self.domain_dims if x != "lon")
        }

    def get_dtype(self):
        return {"meridional": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                self.function(output[:, c]).cpu().numpy()
            )

    def finalize(self, *args):
        pass


class Histogram(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        self.tmpdir = TemporaryDirectory()
        self.file_path = self.tmpdir.name + "/" + "_t_"
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

    def _init_dimensions(self):
        super()._init_dimensions()

    def _init_subgroup(
        self,
    ):
        super()._init_subgroup()
        for channel in self.diagnostic.channels:
            self.subgroup.createGroup(channel)

    def get_dimensions(self):
        # Use variable length string
        # return {"bin_edges": ("time",), "bin_counts": ("time",)}
        return {}

    def get_dtype(self):
        # Use variable length string
        # return {"bin_edges": str, "bin_counts": str}
        return {}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):

        # TODO lift this file I/O to another place.
        mean_path = filesystem.download_cached(config.MEAN)
        scale_path = filesystem.download_cached(config.SCALE)
        mean_np = np.load(mean_path)
        scale_np = np.load(scale_path)
        mean = torch.squeeze(torch.as_tensor(mean_np).to(self.device))
        scale = torch.squeeze(torch.as_tensor(scale_np).to(self.device))

        for c, channel in enumerate(self.diagnostic.channels):
            try:
                # See if there is existing data for this time_index
                old_bin_edges = torch.load(
                    self.file_path
                    + str(time_index)
                    + "_"
                    + "bin_edges_"
                    + channel
                    + ".pt"
                ).to(self.device)
                old_counts = torch.load(
                    self.file_path
                    + str(time_index)
                    + "_"
                    + "bin_counts_"
                    + channel
                    + ".pt"
                ).to(self.device)

                # Note that _update_bins_counts synchronizes
                # bins and counts across devices.
                new_bin_edges, new_counts = _update_bins_counts(
                    output[:, c], old_bin_edges, old_counts
                )
            except FileNotFoundError:
                # If no existing data, initialize bins and edges
                # Build bin_edges
                start = (mean[c] - 10 * scale[c]) * torch.ones(
                    output[0, c].shape, device=self.device, dtype=torch.float32
                )
                end = (mean[c] + 10 * scale[c]) * torch.ones(
                    output[0, c].shape, device=self.device, dtype=torch.float32
                )
                maxs = torch.max(output[:, c], dim=0)[1]
                mins = torch.min(output[:, c], dim=0)[1]
                start = torch.where(start <= mins, start, mins)
                end = torch.where(end >= maxs, end, maxs)
                new_bin_edges = linspace(start, end, self.diagnostic.nbins)
                new_counts = _count_bins(output[:, c], new_bin_edges)

                if DistributedManager.is_initialized() and dist.is_initialized():
                    dist.all_reduce(new_counts, op=dist.ReduceOp.SUM)

            # Save new bin_edges and counts
            new_counts = new_counts.type(torch.int32)
            torch.save(
                new_bin_edges,
                self.file_path + str(time_index) + "_" + "bin_edges_" + channel + ".pt",
            )
            torch.save(
                new_counts,
                self.file_path
                + str(time_index)
                + "_"
                + "bin_counts_"
                + channel
                + ".pt",
            )

    def finalize(self, nc_time: np.ndarray, weather_event, channel_set):

        for c, channel in enumerate(self.diagnostic.channels):
            for it, time in enumerate(nc_time):
                bin_edges = torch.load(
                    self.file_path + str(it) + "_" + "bin_edges_" + channel + ".pt"
                ).to(self.device)
                counts = torch.load(
                    self.file_path + str(it) + "_" + "bin_counts_" + channel + ".pt"
                ).to(self.device)

                nbins = counts.shape[0]
                self.subgroup[channel].createDimension(
                    "bin_counts" + "_t_" + str(it), nbins
                )
                self.subgroup[channel].createVariable(
                    "bin_counts" + "_t_" + str(it),
                    int,
                    ("bin_counts" + "_t_" + str(it), "lat", "lon"),
                )
                self.subgroup[channel].variables["bin_counts" + "_t_" + str(it)][
                    :
                ] = counts.cpu().numpy()

                self.subgroup[channel].createDimension(
                    "bin_edges" + "_t_" + str(it), nbins + 1
                )
                self.subgroup[channel].createVariable(
                    "bin_edges" + "_t_" + str(it),
                    float,
                    ("bin_edges" + "_t_" + str(it), "lat", "lon"),
                )
                self.subgroup[channel].variables["bin_edges" + "_t_" + str(it)][
                    :
                ] = bin_edges.cpu().numpy()


class CRPS(Histogram):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

    def _init_dimensions(self):
        super()._init_dimensions()

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group.createGroup("crps")

    def _init_variables(self):
        super()._init_variables()

    def get_dimensions(self):
        return {**super().get_dimensions(), **{"crps": ("time",) + self.domain_dims}}

    def get_dtype(self):
        return {**super().get_dtype(), **{"crps": float}}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        super().update(output, time_index, batch_id, batch_size)

    def finalize(self, nc_times: np.ndarray, weather_event, channel_set):
        lat_sl, lon_sl = geometry.get_bounds_window(self.domain, self.lat, self.lon)
        for it, time in enumerate(nc_times):
            date_obj = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
            target_data = ic(
                n_history=0,
                grid=self.grid,
                time=date_obj,
                channel_set=channel_set,
                source=weather_event.properties.initial_condition_source,
            )
            for c, channel in enumerate(self.diagnostic.channels):
                target_channel = np.squeeze(
                    np.asarray(target_data.sel(channel=channel))
                )
                bin_edges = torch.load(
                    self.file_path + str(it) + "_" + "bin_edges_" + channel + ".pt"
                ).to(self.device)
                counts = torch.load(
                    self.file_path + str(it) + "_" + "bin_counts_" + channel + ".pt"
                ).to(self.device)
                crps = _crps_from_counts(
                    bin_edges,
                    counts,
                    torch.as_tensor(target_channel[lat_sl, lon_sl]).to(self.device),
                )
                self.subgroup[channel][it] = crps.cpu().numpy()
        return


class EnsembleMean(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)
        self.tmpdir = TemporaryDirectory()
        self.file_path = self.tmpdir.name + "/" + "_t_"

    def _init_variables(self):
        super()._init_variables()

        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup.variables[channel][0] = np.zeros(
                (len(self.group["lat"][:]), len(self.group["lon"][:])), dtype=np.float32
            )

    def get_dimensions(self):
        return {"ensemble_mean": ("time",) + self.domain_dims}

    def get_dtype(self):
        return {"ensemble_mean": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        input_shape = (len(self.diagnostic.channels), len(self.lat), len(self.lon))
        m = Mean(input_shape)
        try:
            # See if there is existing data for this time_index
            old_n = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
            ).to(self.device)

            old_sum = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
            ).to(self.device)

            m.n = old_n
            m.sum = old_sum
            _ = m.update(output)
        except FileNotFoundError:
            _ = m(output)
            m.n = torch.as_tensor(m.n)

        torch.save(
            m.n, self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
        )
        torch.save(
            m.sum, self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
        )

    def finalize(self, nc_time: np.ndarray, weather_event, channel_set):
        for it, time in enumerate(nc_time):
            n = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_number_" + ".pt")
                .cpu()
                .numpy()
            )

            sums = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_sum_" + ".pt")
                .cpu()
                .numpy()
            )

            for c, channel in enumerate(self.diagnostic.channels):
                self.subgroup[channel][it] = sums[c] / n


class EnsembleVariance(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)
        self.tmpdir = TemporaryDirectory()
        self.file_path = self.tmpdir.name + "/" + "_t_"

    def get_dimensions(self):
        return {
            "ensemble_variance": ("time",) + self.domain_dims,
        }

    def get_dtype(self):
        return {"ensemble_variance": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        input_shape = (len(self.diagnostic.channels), len(self.lat), len(self.lon))
        m = Variance(input_shape)
        try:
            # See if there is existing data for this time_index
            old_n = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
            ).to(self.device)

            old_sum = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
            ).to(self.device)

            old_sum2 = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_sum2_" + ".pt"
            ).to(self.device)

            m.n = old_n
            m.sum = old_sum
            m.sum2 = old_sum2
            _ = m.update(output)
        except FileNotFoundError:
            _ = m(output)
            m.n = torch.as_tensor(m.n)

        torch.save(
            m.n, self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
        )
        torch.save(
            m.sum, self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
        )
        torch.save(
            m.sum2, self.file_path + str(time_index) + "_" + "ensemble_sum2_" + ".pt"
        )

    def finalize(self, nc_time: np.ndarray, weather_event, channel_set):
        for it, time in enumerate(nc_time):
            n = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_number_" + ".pt")
                .cpu()
                .numpy()
            )

            sum2 = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_sum2_" + ".pt")
                .cpu()
                .numpy()
            )

            for c, channel in enumerate(self.diagnostic.channels):
                self.subgroup[channel][it] = sum2[c] / (n - 1.0)


class EnsembleMeanSkill(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)
        self.tmpdir = TemporaryDirectory()
        self.file_path = self.tmpdir.name + "/" + "_t_"

    def _init_variables(self):
        super()._init_variables()

    def get_dimensions(self):
        return {
            "skill": ("time",) + self.domain_dims,
        }

    def get_dtype(self):
        return {"skill": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        input_shape = (len(self.diagnostic.channels), len(self.lat), len(self.lon))
        m = Mean(input_shape)
        try:
            # See if there is existing data for this time_index
            old_n = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
            ).to(self.device)

            old_sum = torch.load(
                self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
            ).to(self.device)

            m.n = old_n
            m.sum = old_sum
            _ = m.update(output)
        except FileNotFoundError:
            _ = m(output)
            m.n = torch.as_tensor(m.n)

        torch.save(
            m.n, self.file_path + str(time_index) + "_" + "ensemble_number_" + ".pt"
        )
        torch.save(
            m.sum, self.file_path + str(time_index) + "_" + "ensemble_sum_" + ".pt"
        )

    def finalize(self, nc_time: np.ndarray, weather_event, channel_set):
        lat_sl, lon_sl = geometry.get_bounds_window(self.domain, self.lat, self.lon)
        for it, time in enumerate(nc_time):
            date_obj = datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S")
            target_data = ic(
                n_history=0,
                grid=self.grid,
                time=date_obj,
                channel_set=channel_set,
                source=weather_event.properties.initial_condition_source,
            )
            n = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_number_" + ".pt")
                .cpu()
                .numpy()
            )

            mean = (
                torch.load(self.file_path + str(it) + "_" + "ensemble_sum_" + ".pt")
                .cpu()
                .numpy()
                / n
            )

            for c, channel in enumerate(self.diagnostic.channels):
                target_channel = np.squeeze(
                    np.asarray(target_data.sel(channel=channel))
                )
                skill = torch.sqrt(
                    (
                        torch.as_tensor(mean[c] - target_channel[lat_sl, lon_sl]).to(
                            self.device
                        )
                    )
                    ** 2
                )
                self.subgroup.variables[channel][it] = skill.cpu().numpy()
        return


#######################################################################################
DiagnosticTypes = {
    "raw": Raw,
    "absolute": Absolute,
    "meridional": Meridional,
    "zonal": Zonal,
    "histogram": Histogram,
    "crps": CRPS,
    "ensemble_mean": EnsembleMean,
    "ensemble_variance": EnsembleVariance,
    "skill": EnsembleMeanSkill,
}
#######################################################################################


def extended_best_track_reader(storm, t_init, t_finit):
    stormname = storm[:-4]
    stormyear = storm[-4:]
    ExtendedBestTrack = "./ExtendedBestTrack_combined.txt"
    vmax = []
    mslp = []
    rmax = []
    lat = []
    lon = []
    time = []
    with open(ExtendedBestTrack, "r") as EBT:
        for line in EBT:
            if stormname.upper() in line[9:20] and stormyear in line[28:32]:
                time_ebt = line[28:32] + line[21:27] + "00:00"
                date_obj = datetime.strptime(time_ebt, "%Y%m%d%H%M:%S")
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                mytime = 24 * day_of_year + hour_of_day - t_init
                if mytime < 0.0 or mytime > t_finit:
                    continue
                time.append(mytime)
                vmax.append(float(line[46:49]))
                rmax.append(float(line[55:58]))
                lat.append(float(line[34:38]))
                lon.append(float(line[41:45]))
                mslp.append(float(line[50:54]))

    vmax = np.array(vmax)
    vmax[vmax < 0.0] = np.nan
    rmax = np.array(rmax)
    rmax[rmax < 0.0] = np.nan
    lat = np.array(lat)
    lat[np.where(lat < 0.0)] = np.nan
    lon = np.array(lon)
    lon = np.subtract(360.0, lon)
    lon[np.where(lon < 0.0)] = np.nan
    mslp = np.array(mslp)
    mslp[mslp < 0.0] = np.nan
    return time, lat, lon, mslp, vmax, rmax


def tropical_cyclone_tracker(z_850, z_250, u_850, v_850):
    vorticity = compute_vorticity(u_850, v_850)
    dZ = np.subtract(z_250, z_850)
    i_z850, j_z850 = np.unravel_index(z_850.argmin(), z_850.shape)
    i_dz, j_dz = np.unravel_index(dZ.argmax(), dZ.shape)
    i_v, j_v = np.unravel_index(vorticity.argmax(), vorticity.shape)
    max_tilt_i = np.square(np.max([i_dz, i_v, i_z850]) - np.min([i_dz, i_v, i_z850]))
    max_tilt_j = np.square(np.max([j_dz, j_v, j_z850]) - np.min([j_dz, j_v, j_z850]))
    max_tilt = np.sqrt(max_tilt_i + max_tilt_j)
    if max_tilt > 6:  # tilt larger than 150km and its not a TC
        i_z850 = np.nan
        j_z850 = np.nan
    return i_z850, j_z850


def compute_vorticity(u, v, dx=25000.0, dy=25000.0):
    dudx = np.gradient(u, axis=0) / dx
    dvdy = np.gradient(v, axis=1) / dy
    vorticity = np.add(dvdy, dudx)
    return vorticity


def exceedance_probability(V, V_thresh=33.0):
    V = np.array(V)
    V_thresh = np.array(V_thresh)
    exceedance_prob = np.zeros(V.shape)
    exceedance_prob[V > V_thresh] = 1
    return exceedance_prob


def emanuel_damage_function(V, V_thresh=25.0, V_half=77.0):
    """
    Returns the Emanuel damage function,
    Equ. 1 in Emanuel 2011: Global warming effects on U.S. hurricane damage
    """
    Vn = np.divide(
        np.clip(np.subtract(V, V_thresh), 0.0, None), np.subtract(V_half, V_thresh)
    )
    Vn3 = np.power(Vn, 3)
    return np.divide(Vn3, 1.0 + Vn3)
