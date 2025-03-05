"""
Tests for the util module

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import pytest
import numpy as np
from pypulseq import util

def test_round_raster_time():
    N = 100
    t = abs(np.random.normal(scale=20, size=N))*1E-6
    raster_time = 10E-6

    for case in ['up', 'down']:

        for n in range(N):

            ti = t[n]
            if case == 'up':
                to = util.round_up_raster_time(ti, raster_time)
            elif case == 'down':
                to = util.round_down_raster_time(ti, raster_time)

            if to % raster_time:
                pytest.raises(ValueError, match=r'not rounded.')
