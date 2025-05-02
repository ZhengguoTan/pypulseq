"""
utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import math


# %%
def cart2sph(x, y, z):
    """
    Transforms Cartesian coordinates to spherical coordinates.

    Args:
        x: X coordinate (NumPy array or scalar).
        y: Y coordinate (NumPy array or scalar).
        z: Z coordinate (NumPy array or scalar).

    Returns:
        A tuple (azimuth, elevation, r) containing:
        - azimuth: Angle in the xy-plane from the x-axis (radians).
        - elevation: Angle from the xy-plane to the z-axis (radians).
        - r: Radial distance from the origin.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    return azimuth, elevation, r

# %%
def round_up_raster_time(t: float,
                         raster_time: float = 10E-6):
    """
    Ceil the input time (t) to the gradient raster time in s.

    Args:
        t: time in s.
        raster_time: gradient raster time in s. [default: 10e-6]

    Returns:
        time ceiled by the gradient raster time in s.
    """
    if t < 0:
        return 0

    # convert to second*10
    t_sec = int(t * 1E7)
    raster_time_sec = int(raster_time * 1E7)
    assert raster_time_sec > 0

    resid = t_sec % raster_time_sec

    if resid > 0:
        t_sec = t_sec + raster_time_sec - resid

    # return in us and allow only 9 decimal points
    return round(t_sec * 1E-7, 7)

# %%
def round_down_raster_time(t: float,
                           raster_time: float = 10E-6):
    """
    Floor the input time (t) to the gradient raster time in s.

    Args:
        t: time in s.
        raster_time: gradient raster time in s. [default: 10e-6]

    Returns:
        time floored by the gradient raster time in s.
    """
    if t < 0:
        return 0

    # convert to second
    t_sec = int(t * 1E7)
    raster_time_sec = int(raster_time * 1E7)
    assert raster_time_sec > 0

    resid = t_sec % raster_time_sec

    if resid > 0:
        t_sec = t_sec - resid

    # return in us and allow only 9 decimal points
    return round(t_sec * 1E-7, 7)
