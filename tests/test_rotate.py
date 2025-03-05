"""
Tests for the rotate module

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

from copy import deepcopy
import pytest
from pypulseq import eps, Opts, util, make_trapezoid, rotate, rotate3
import pypulseq as pp
import numpy as np

system = Opts(
    B0=2.89,
    max_grad=78,
    grad_unit='mT/m',
    grad_raster_time=10E-6,
    max_slew=150,
    slew_unit='T/m/s',
    rf_ringdown_time=10E-6,
    rf_dead_time=100E-6,
    adc_dead_time=0,
)


def test_rotate_along_the_same_axis():
    """
    rotate a gradient along its own axis. `rotate` should return the same gradient as input.
    """
    fov = 220E-3
    delta_k = 1/fov

    blip_duration = util.round_up_raster_time(
        2*np.sqrt(delta_k/system.max_slew),
        system.grad_raster_time
        )

    for ax in ['x', 'y', 'z']:

        gy = make_trapezoid(ax, system=system,
                            area=-delta_k,
                            duration=blip_duration)

        gy_rot = rotate(gy, angle=np.pi/2, axis=ax, system=system)

        assert gy_rot[0] == gy

def test_rotate_orthogonal_axis():
    """
    rotate a gradient along its orthogonal axis. `rotate` should return the same gradient as input.
    """
    fov = 220E-3
    delta_k = 1/fov

    blip_duration = util.round_up_raster_time(
        2*np.sqrt(delta_k/system.max_slew),
        system.grad_raster_time
        )

    axes = ['x', 'y', 'z']

    for ax in axes:

        g = make_trapezoid(ax, system=system,
                           area=-delta_k,
                           duration=blip_duration)

        ax_excl = [a for a in axes if a != ax]

        for ae in ax_excl:

            g_rot = rotate(g, angle=np.pi/2, axis=ae, system=system)

            ax_last = [a for a in ax_excl if a != ae]

            assert g_rot[0].channel == ax_last[0]

def test_rotate3():
    """
    rotate a gradient along x, y, and z axis sequentially
    """
    fov = 220E-3
    delta_k = 1/fov

    blip_duration = util.round_up_raster_time(
        2*np.sqrt(delta_k/system.max_slew),
        system.grad_raster_time
        )

    axes = ['x', 'y', 'z']

    rads = np.random.normal(0, 1, len(axes))

    for g_ax in axes:

        g = make_trapezoid(g_ax, system=system,
                           area=-delta_k,
                           duration=blip_duration)

        if g_ax == 'x':
            vec = np.array([[g.amplitude],[0],[0]])
        elif g_ax == 'y':
            vec = np.array([[0],[g.amplitude],[0]])
        elif g_ax == 'z':
            vec = np.array([[0],[0],[g.amplitude]])


        g_rot = deepcopy(g)

        for n in range(len(axes)):

            g_rot = rotate3(g_rot, angle=rads[n], axis=axes[n])

        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(rads[0]), -np.sin(rads[0])],
                          [0, np.sin(rads[0]),  np.cos(rads[0])]])

        rot_y = np.array([[ np.cos(rads[1]), 0, np.sin(rads[1])],
                          [0, 1, 0],
                          [-np.sin(rads[1]), 0, np.cos(rads[1])]])

        rot_z = np.array([[np.cos(rads[2]), -np.sin(rads[2]), 0],
                          [np.sin(rads[2]),  np.cos(rads[2]), 0],
                          [0, 0, 1]])

        amp = np.matmul(rot_z, np.matmul(rot_y, np.matmul(rot_x, vec)))

        for n in range(len(axes)):
            gamp = getattr(g_rot[n], 'amplitude')
            assert eps > abs(gamp - amp[n])
