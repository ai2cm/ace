# Code for computing the insolation as in CM4 is derived from the astronomy
# and time_manager modules in GFDL's Flexible Modeling System (FMS) Fortran
# code base, which is released under an Apache 2.0 License. We include a
# copy of the license in this module for reference.

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/

#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

#    1. Definitions.

#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.

#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.

#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

import datetime
import warnings

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import HorizontalCoordinates
from fme.core.device import get_device

AUTUMNAL_EQUINOX = (1998, 9, 23, 5, 37, 0)
BASE_DATE = (1, 1, 1)
NUM_ANGLES = 3600
CFTIME_TYPES = {
    "noleap": cftime.DatetimeNoLeap,
    "standard": cftime.DatetimeGregorian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "julian": cftime.DatetimeJulian,
    "360_day": cftime.Datetime360Day,
    "all_leap": cftime.DatetimeAllLeap,
}
SECONDS_PER_DAY = 86400

# Define length of year based on definitions in the FMS time_manager module:
# https://github.com/NOAA-GFDL/FMS/blob/039d5f73fc4c7ce83117ca555f0b0761caf18e06/time_manager/time_manager.F90#L2326-L2362
LENGTH_OF_YEAR = {
    "noleap": datetime.timedelta(days=365),
    "standard": datetime.timedelta(days=365, seconds=20952),
    "proleptic_gregorian": datetime.timedelta(days=365, seconds=20952),
    "julian": datetime.timedelta(days=365, seconds=21600),
    "360_day": datetime.timedelta(days=360),
    "all_leap": datetime.timedelta(days=366),
}

# The longest timestep that the FMS code supports averaging over is 12 hours:
# https://github.com/NOAA-GFDL/FMS/blob/039d5f73fc4c7ce83117ca555f0b0761caf18e06/astronomy/astronomy.F90#L116-L123
MAXIMUM_TIMESTEP = datetime.timedelta(hours=12)


class CM4Insolation:
    def __init__(
        self, obliquity: float, eccentricity: float, longitude_of_perhelion: float
    ):
        self.obliquity = torch.as_tensor(obliquity)
        self.eccentricity = torch.as_tensor(eccentricity)
        self.longitude_of_perhelion = torch.as_tensor(longitude_of_perhelion)
        self._orbital_angle_lookup_table = _compute_orbital_angle_lookup_table(
            self.eccentricity, self.longitude_of_perhelion
        )

    def __call__(
        self,
        time: xr.DataArray,
        timestep: datetime.timedelta,
        horizontal_coordinates: HorizontalCoordinates,
        solar_constant: torch.tensor,
    ) -> torch.tensor:
        return _compute_insolation(
            time,
            timestep,
            horizontal_coordinates,
            solar_constant,
            self.obliquity,
            self.eccentricity,
            self.longitude_of_perhelion,
            self._orbital_angle_lookup_table,
        )


def _convert_lat_lon_from_degrees_to_radians(
    lat: torch.tensor, lon: torch.tensor
) -> tuple[torch.tensor, torch.tensor]:
    lat_range = lat.max() - lat.min()
    lon_range = lon.max() - lon.min()
    if lat_range < torch.pi or lon_range < 2 * torch.pi:
        warnings.warn(
            f"Range of latitude and/or longitude coordinates is smaller "
            f"than expected for units of degrees. Latitude range = {lat_range:0.2f}; "
            f"longitude range = {lon_range:0.2f}. Computing insolation assumes "
            f"latitude and longitude start out in units of degrees instead of "
            f"radians."
        )
    return torch.deg2rad(lat), torch.deg2rad(lon)


def _prepare_coordinates(
    time: xr.DataArray,
    timestep: datetime.timedelta,
    horizontal_coordinates: HorizontalCoordinates,
) -> tuple[np.ndarray, torch.tensor, torch.tensor]:
    # We are interested in the average of an interval of length timestep
    # ending at the given time. CM4's computation requires the time at
    # the beginning of the interval. We also would prefer time as a
    # NumPy array.
    time = time.to_numpy() - timestep

    # Obtain latitude and longitude fully broadcast in the horizontal,
    # and ensure they are on the current device.
    lat, lon = horizontal_coordinates.to(get_device()).meshgrid

    # Add leading time dimensions to broadcast with time array.
    for _ in range(time.ndim):
        lat = lat[None, ...]
        lon = lon[None, ...]

    # Add trailing dimensions to time array to broadcast with lat and lon.
    # Do so in a grid-agnostic way (e.g. lat-lon, HEALPix, etc.).
    for _ in range(lat.ndim - time.ndim):
        time = time[..., None]

    # Convert latitude and longitude from degrees to radians, warning if
    # either are unlikely in units degrees as inputs.
    lat, lon = _convert_lat_lon_from_degrees_to_radians(lat, lon)

    return time, lat, lon


def _infer_calendar(time: np.ndarray) -> str:
    return time.ravel()[0].calendar


def _orbital_time(time: np.ndarray, dtype=torch.dtype) -> torch.tensor:
    calendar = _infer_calendar(time)
    autumnal_equinox = CFTIME_TYPES[calendar](*AUTUMNAL_EQUINOX)
    t = (time - autumnal_equinox) / LENGTH_OF_YEAR[calendar]
    t = 2 * np.pi * (t - np.floor(t))
    t = t.astype(np.float64)
    return torch.as_tensor(t, device=get_device(), dtype=dtype)


def _universal_time(time: np.ndarray, dtype: torch.dtype) -> torch.tensor:
    calendar = _infer_calendar(time)
    base_date = CFTIME_TYPES[calendar](*BASE_DATE)
    delta = time - base_date
    seconds = delta % datetime.timedelta(days=1)
    seconds = seconds / datetime.timedelta(seconds=1)
    seconds = seconds.astype(np.float64)
    universal_time = 2 * torch.pi * (seconds / SECONDS_PER_DAY)
    return torch.as_tensor(universal_time, device=get_device(), dtype=dtype)


def _r_inv_squared(
    angle: torch.tensor,
    eccentricity: torch.tensor,
    longitude_of_perhelion: torch.tensor,
) -> torch.tensor:
    rad_per = torch.deg2rad(longitude_of_perhelion)
    r = (1 - eccentricity**2) / (1 + eccentricity * torch.cos(angle - rad_per))
    return r ** (-2)


def _compute_orbital_angle_lookup_table(
    eccentricity: torch.tensor, longitude_of_perhelion: torch.tensor
) -> torch.tensor:
    orbital_angle_lookup_table = torch.zeros(NUM_ANGLES, device=get_device())
    dt = 2 * torch.pi / NUM_ANGLES
    norm = torch.sqrt(1 - eccentricity**2)
    dt = dt * norm
    for i in range(NUM_ANGLES):
        d1 = dt * _r_inv_squared(
            orbital_angle_lookup_table[i - 1], eccentricity, longitude_of_perhelion
        )
        d2 = dt * _r_inv_squared(
            orbital_angle_lookup_table[i - 1] + 0.5 * d1,
            eccentricity,
            longitude_of_perhelion,
        )
        d3 = dt * _r_inv_squared(
            orbital_angle_lookup_table[i - 1] + 0.5 * d2,
            eccentricity,
            longitude_of_perhelion,
        )
        d4 = dt * _r_inv_squared(
            orbital_angle_lookup_table[i - 1] + d3, eccentricity, longitude_of_perhelion
        )
        d5 = d1 / 6.0 + d2 / 3.0 + d3 / 3.0 + d4 / 6.0
        orbital_angle_lookup_table[i] = orbital_angle_lookup_table[i - 1] + d5
    return orbital_angle_lookup_table


def _orbital_angle(
    orbital_time: torch.tensor, orbital_angle_lookup_table: torch.tensor
) -> torch.tensor:
    norm_time = orbital_time * NUM_ANGLES / (2 * torch.pi)
    norm_time_int = torch.floor(norm_time).to(torch.int64)
    norm_time_int = norm_time_int % NUM_ANGLES
    x = norm_time - torch.floor(norm_time)
    y = (1.0 - x) * orbital_angle_lookup_table[
        norm_time_int
    ] + x * orbital_angle_lookup_table[norm_time_int + 1]
    return y % (2 * torch.pi)


def _declination(orbital_angle: torch.tensor, obliquity: torch.tensor) -> torch.tensor:
    obliquity_radians = torch.deg2rad(obliquity)
    sin_declination = -torch.sin(obliquity_radians) * torch.sin(orbital_angle)
    return torch.arcsin(sin_declination)


def _half_day(lat: torch.tensor, declination: torch.tensor) -> torch.tensor:
    tan_declination = torch.tan(declination)
    lat = torch.where(lat == 0.5 * torch.pi, lat - 1.0e-5, lat)
    lat = torch.where(lat == -0.5 * torch.pi, lat + 1.0e-5, lat)
    cos_half_day = -torch.tan(lat) * tan_declination
    h = torch.where(
        (cos_half_day > -1.0) & (cos_half_day < 1.0), torch.arccos(cos_half_day), 0.0
    )
    h = torch.where(cos_half_day <= -1.0, torch.pi, h)
    h = torch.where(cos_half_day >= 1.0, 0.0, h)
    return h


def _diurnal_solar(
    time: xr.DataArray,
    timestep: datetime.timedelta,
    horizontal_coordinates: HorizontalCoordinates,
    obliquity: torch.tensor,
    eccentricity: torch.tensor,
    longitude_of_perhelion: torch.tensor,
    orbital_angle_lookup_table: torch.tensor,
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    time, lat, lon = _prepare_coordinates(time, timestep, horizontal_coordinates)

    # This limitation would not be that difficult to address, but since we do
    # not have an immediate use-case for this, we will punt on it for now. The
    # existing limitation is inherited from the implementation in FMS.
    if timestep >= MAXIMUM_TIMESTEP:
        raise NotImplementedError(
            f"Computing insolation via the CM4 implementation is not "
            f"implemented for a model timestep greater than or equal "
            f"to 12 hours. Timestep is {timestep!r}."
        )

    universal_time = _universal_time(time, dtype=lat.dtype)
    time_since_autumnal_equinox = _orbital_time(time, dtype=lat.dtype)

    orbital_angle = _orbital_angle(
        time_since_autumnal_equinox, orbital_angle_lookup_table
    )
    declination = _declination(orbital_angle, obliquity)
    rrsun = _r_inv_squared(orbital_angle, eccentricity, longitude_of_perhelion)

    aa = torch.sin(lat) * torch.sin(declination)
    bb = torch.cos(lat) * torch.cos(declination)

    t = universal_time + lon - torch.pi
    t = torch.where(t >= torch.pi, t - 2 * torch.pi, t)
    t = torch.where(t < -torch.pi, t + 2 * torch.pi, t)

    h = _half_day(lat, declination)
    dt = 2 * torch.pi * timestep / datetime.timedelta(hours=24)

    tt = t + dt
    st = torch.sin(t)
    stt = torch.sin(tt)
    sh = torch.sin(h)
    cosz = 0.0
    cosz = torch.where((t < -h) & (tt < -h), 0.0, cosz)
    cosz = torch.where(
        ((tt + h) != 0.0) & (t < -h) & (torch.abs(tt) <= h),
        aa + bb * (stt + sh) / (tt + h),
        cosz,
    )
    cosz = torch.where(
        (t < -h) & (h != 0.0) & (h < tt), aa + bb * (sh + sh) / (h + h), cosz
    )

    cosz = torch.where(
        (torch.abs(t) <= h) & (torch.abs(tt) <= h),
        aa + bb * (stt - st) / (tt - t),
        cosz,
    )
    cosz = torch.where(
        ((h - t) != 0.0) & (torch.abs(t) <= h) & (h < tt),
        aa + bb * (sh - st) / (h - t),
        cosz,
    )
    cosz = torch.where(
        ((2 * torch.pi - h) < tt) & ((tt + h - 2 * torch.pi) != 0.0) & (t <= h),
        (cosz * (h - t) + (aa * (tt + h - 2 * torch.pi)) + bb * (stt + sh))
        / ((h - t) + (tt + h - 2 * torch.pi)),
        cosz,
    )
    cosz = torch.where((h < t) & ((2 * torch.pi - h) >= tt), 0.0, cosz)
    cosz = torch.where(
        (h < t) & ((2 * torch.pi - h) < tt),
        aa + bb * (stt + sh) / (tt + h - 2 * torch.pi),
        cosz,
    )
    cosz = torch.maximum(torch.tensor(0.0), cosz)

    fracday = 0.0
    fracday = torch.where((t < -h) & (torch.abs(tt) <= h), (tt + h) / dt, fracday)
    fracday = torch.where((t < -h) & (h < tt), (h + h) / dt, fracday)
    fracday = torch.where(
        (torch.abs(t) <= h) & (torch.abs(tt) <= h), (tt - t) / dt, fracday
    )
    fracday = torch.where((torch.abs(t) <= h) & (h < tt), (h - t) / dt, fracday)
    fracday = torch.where(h < t, 0.0, fracday)
    fracday = torch.where(
        (2 * torch.pi - h) < tt, fracday + (tt + h - 2 * torch.pi) / dt, fracday
    )

    return cosz, fracday, rrsun


def _compute_insolation(
    time: xr.DataArray,
    timestep: datetime.timedelta,
    horizontal_coordinates: HorizontalCoordinates,
    solar_constant: torch.tensor,
    obliquity: torch.tensor,
    eccentricity: torch.tensor,
    longitude_of_perhelion: torch.tensor,
    orbital_angle_lookup_table: torch.tensor,
) -> torch.tensor:
    cosz, fracday, rrsun = _diurnal_solar(
        time,
        timestep,
        horizontal_coordinates.to(get_device()),
        obliquity,
        eccentricity,
        longitude_of_perhelion,
        orbital_angle_lookup_table,
    )
    solar_constant = solar_constant.to(get_device())
    insolation = solar_constant * rrsun * fracday * cosz
    return insolation.to(solar_constant.dtype)
