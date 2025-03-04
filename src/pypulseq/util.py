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
def ceil(t: float,
         raster_time: float = 10E-6):
    """
    Ceil the input time (t) to the gradient raster time in s.

    Args:
        t: time in s.
        grad_raster_time: gradient raster time in s. [default: 10e-6]

    Returns:
        time ceiled by the gradient raster time in s.
    """

    return math.ceil(t / raster_time) * raster_time
