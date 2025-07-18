import numpy as np


def u_v_to_x_y_z_wind(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts u and v wind components to x, y, z wind components.

    The x-axis is defined as the vector from the center of the earth to the
    intersection of the equator and the prime meridian. The y-axis is defined
    as the vector from the center of the earth to the intersection of the
    equator and the 90 degree east meridian. The z-axis is defined as the
    vector from the center of the earth to the north pole.

    Args:
        u: u wind component
        v: v wind component
        lat: latitude, in degrees
        lon: longitude, in degrees

    Returns:
        wx: x wind component
        wy: y wind component
        wz: z wind component
    """
    # for a graphical proof of the equations used here, see
    # https://github.com/ai2cm/full-model/pull/355#issuecomment-1729773301

    # Convert to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    # Horizontal winds
    #
    # Contribution from u
    #
    # The u component of the wind is aligned with the longitude lines,
    # so all of its contributions are to the x and y components of the wind.
    # At the prime meridian (which lies on the x axis), the u component points
    # parallel to the y-axis, so it contributes only to the y component of the
    # wind. As we move eastward, the u component points more and more in the
    # negative x direction, so it contributes more and more to the x component
    # of the wind. At the 90 degree east meridian, the u component points
    # parallel to the x-axis (in the negative direction).
    #
    # This influence on the x component is captured by multiplying the u component
    # by the negative sine of the longitude, which is zero at the prime meridian
    # and then becomes negative until reaching -1 at 90 degrees east.
    #
    # The influence on the y component is captured by multiplying the u component
    # by the cosine of the longitude, which is 1 at the prime meridian and then
    # decreases until reaching 0 at 90 degrees east.
    #
    # Contribution from v
    #
    # The v component of the wind is aligned with the latitude lines,
    # with no contribution to the horizontal at the equator and full contribution
    # at the poles. This is captured by multiplying the v component by the sine
    # of the latitude, which is 0 at the equator and 1 at the poles.
    #
    # The direction of the horizontal contribution is the vector pointing inwards
    # towards the axis of rotation of the Earth. At the prime meridian, this
    # is the negative x direction. At the 90 degree east meridian, this is the
    # negative y direction.
    #
    # This influence on the x component is captured by multiplying the v component
    # by the negative cosine of the longitude, which is 1 at the prime meridian
    # and then decreases until reaching 0 at 90 degrees east.
    #
    # The influence on the y component is captured by multiplying the v component
    # by the negative sine of the longitude, which is 0 at the prime meridian
    # and then decreases until reaching -1 at 90 degrees east.
    #
    # An exact derivation proving this effect is captured by sine and cosine
    # can be done graphically. Generally for these kinds of problems on a sphere
    # or circle it's always sine and cosine, and the question is which one and
    # whether it's positive or negative.

    # Wind in the x-direction:
    wx = -u * np.sin(lon) - v * np.sin(lat) * np.cos(lon)

    # Wind in the y-direction:
    wy = u * np.cos(lon) - v * np.sin(lat) * np.sin(lon)

    # Wind in the z-direction:

    # As the u-component is along latitude lines, and latitude lines are
    # perpendicular to Earth's axis of rotation, u does not appear in wz.
    #
    # The v-component of the wind is entirely aligned with Earth's axis of rotation
    # at the equator, and is entirely perpendicular at the poles, an effect that
    # is captured by multiplying the v component by the cosine of the latitude.
    wz = v * np.cos(lat)

    return wx, wy, wz


def normalize_vector(*vector_components: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    The vector is assumed to be represented in an orthonormal basis, where
    each component is orthogonal to the others.

    Args:
        vector_components: components of the vector (e.g. x-, y-, and z-components)

    Returns:
        normalized vector, as a numpy array where each component has been
            concatenated along a new first dimension
    """
    scale = np.divide(
        1.0,
        np.sum(np.asarray([item**2.0 for item in vector_components]), axis=0) ** 0.5,
    )
    return np.asarray([item * scale for item in vector_components])


def lon_lat_to_xyz(lon, lat):
    """
    Convert (lon, lat) to (x, y, z).

    Args:
        lon: n-dimensional array of longitudes, in degrees
        lat: n-dimensional array of latitudes, in degrees

    Returns:
        x: n-dimensional array of x values
        y: n-dimensional array of y values
        z: n-dimensional array of z values
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    x, y, z = normalize_vector(x, y, z)
    return x, y, z


def xyz_to_lon_lat(x, y, z):
    """
    Convert (x, y, z) to (lon, lat).

    Args:
        x: n-dimensional array of x values
        y: n-dimensional array of y values
        z: n-dimensional array of z values

    Returns:
        lon: n-dimensional array of longitudes, in degrees
        lat: n-dimensional array of latitudes, in degrees
    """
    x, y, z = normalize_vector(x, y, z)
    # double transpose to index last dimension, regardless of number of dimensions
    lon = np.zeros_like(x)
    nonzero_lon = np.abs(x) + np.abs(y) >= 1.0e-10
    lon[nonzero_lon] = np.arctan2(y[nonzero_lon], x[nonzero_lon])
    negative_lon = lon < 0.0
    while np.any(negative_lon):
        lon[negative_lon] += 2 * np.pi
        negative_lon = lon < 0.0
    lat = np.arcsin(z)
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return lon, lat
