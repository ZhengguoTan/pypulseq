"""
Tests for the rotate module

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import pytest
from pypulseq import Opts, util, make_trapezoid, rotate
import pypulseq as pp
import numpy as np

def test_rotate_along_the_same_axis():
    """
    rotate a gradient along its own axis. `rotate` should return the same gradient as input.
    """
    system = pp.Opts(
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
    seq = pp.Sequence(system)

    fov = 220E-3
    delta_k = 1/fov

    blip_duration = util.round_to_grad_raster_time(
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
    system = pp.Opts(
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
    seq = pp.Sequence(system)

    fov = 220E-3
    delta_k = 1/fov

    blip_duration = util.round_to_grad_raster_time(
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
