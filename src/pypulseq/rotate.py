from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq.add_gradients import add_gradients
from pypulseq.opts import Opts
from pypulseq.scale_grad import scale_grad
from pypulseq.utils.tracing import trace, trace_enabled


def __get_grad_abs_mag(grad: SimpleNamespace) -> np.ndarray:
    if grad.type == 'trap':
        return abs(grad.amplitude)
    return np.max(np.abs(grad.waveform))


def rotate(*args: SimpleNamespace, angle: float, axis: str, system: Union[Opts, None] = None) -> List[SimpleNamespace]:
    """
    Rotates the corresponding gradient(s) about the given axis by the specified amount. Gradients parallel to the
    rotation axis and non-gradient(s) are not affected. Possible rotation axes are 'x', 'y' or 'z'.

    See also `pypulseq.Sequence.sequence.add_block()`.

    Parameters
    ----------
    axis : str
        Axis about which the gradient(s) will be rotated.
    angle : float
        Angle by which the gradient(s) will be rotated.
    args : SimpleNamespace
        Gradient(s).

    Returns
    -------
    rotated_grads : [SimpleNamespace]
        Rotated gradient(s).
    """
    if system is None:
        system = Opts.default

    axes = ['x', 'y', 'z']

    # Cycle through the objects and rotate gradients non-parallel to the given rotation axis. Rotated gradients
    # assigned to the same axis are then added together.

    # First create indexes of the objects to be bypassed or rotated
    i_rotate1 = []
    i_rotate2 = []
    i_bypass = []

    axes.remove(axis)
    axes_to_rotate = axes
    if len(axes_to_rotate) != 2:
        raise ValueError('Incorrect axes specification.')

    for i in range(len(args)):
        event = args[i]

        if (event.type != 'grad' and event.type != 'trap') or event.channel == axis:
            i_bypass.append(i)
        else:
            if event.channel == axes_to_rotate[0]:
                i_rotate1.append(i)
            else:
                if event.channel == axes_to_rotate[1]:
                    i_rotate2.append(i)
                else:
                    i_bypass.append(i)  # Should never happen

    # Now every gradient to be rotated generates two new gradients: one on the original axis and one on the other from
    # the axes_to_rotate list
    rotated1 = []
    rotated2 = []
    max_mag = 0  # Measure of relevant amplitude
    for i in range(len(i_rotate1)):
        g = args[i_rotate1[i]]
        max_mag = max(max_mag, __get_grad_abs_mag(g))
        rotated1.append(scale_grad(grad=g, scale=np.cos(angle)))
        g = scale_grad(grad=g, scale=np.sin(angle))
        g.channel = axes_to_rotate[1]
        rotated2.append(g)

    for i in range(len(i_rotate2)):
        g = args[i_rotate2[i]]
        max_mag = max(max_mag, __get_grad_abs_mag(g))
        rotated2.append(scale_grad(grad=g, scale=np.cos(angle)))
        g = scale_grad(grad=g, scale=-np.sin(angle))
        g.channel = axes_to_rotate[0]
        rotated1.append(g)

    # Eliminate zero-amplitude gradients
    threshold = 1e-6 * max_mag
    for i in range(len(rotated1) - 1, -1, -1):
        if __get_grad_abs_mag(rotated1[i]) < threshold:
            rotated1.pop(i)
    for i in range(len(rotated2) - 1, -1, -1):
        if __get_grad_abs_mag(rotated2[i]) < threshold:
            rotated2.pop(i)

    # Add gradients on the corresponding axis together
    g = []
    if len(rotated1) != 0:
        g.append(add_gradients(grads=rotated1, system=system))

    if len(rotated2) != 0:
        g.append(add_gradients(grads=rotated2, system=system))

    # Eliminate zero amplitude gradients
    for i in range(len(g) - 1, -1, -1):
        if __get_grad_abs_mag(g[i]) < threshold:
            g.pop(i)

    # Export
    bypass = np.take(args, i_bypass)
    rotated_grads = [*bypass, *g]

    if trace_enabled():
        for grad in rotated_grads:
            grad.trace = trace()

    return rotated_grads


def _rotation_matrix(axis: str = 'x',
                     angle: float = np.pi):

    c, s = np.cos(angle), np.sin(angle)

    if axis == 'x':
        mat = np.array([[1, 0,  0],
                        [0, c, -s],
                        [0, s,  c]])
    elif axis == 'y':
        mat = np.array([[ c, 0, s],
                        [ 0, 1, 0],
                        [-s, 0, c]])
    elif axis == 'z':
        mat = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])

    return mat

def _value_to_3x1_array(val: float, axis: str):

    if axis == 'x':
        output = np.array([[val], [0], [0]])
    elif axis == 'y':
        output = np.array([[0], [val], [0]])
    elif axis == 'z':
        output = np.array([[0], [0], [val]])

    return output

def rotate3(input: Union[SimpleNamespace, List[SimpleNamespace]],
            angle: float = np.pi/2,
            axis: str = 'x') -> List[SimpleNamespace]:

    # if input is a single SimpleNamesapce, convert it to list
    if isinstance(input, SimpleNamespace):
        input = [input]

    # allows only trapezoid gradients with the same timing
    for n in range(len(input)):
        assert input[n].type == 'trap'
        assert input[n].rise_time == input[0].rise_time
        assert input[n].flat_time == input[0].flat_time
        assert input[n].fall_time == input[0].fall_time

    # rotate each gradient to 3-axis gradients
    rot = _rotation_matrix(axis, angle)

    modified_tags = ['amplitude', 'area', 'flat_area']

    output = []
    for chan in ['x', 'y', 'z']:
        grad = deepcopy(input[0])
        grad.channel = chan

        for tag in modified_tags:
            setattr(grad, tag, 0.)

        output.append(grad)

    for n in range(len(input)):

        grad = input[n]

        for tag in modified_tags:

            val = getattr(grad, tag)
            val = val.item() if isinstance(val, np.ndarray) else val
            arr = _value_to_3x1_array(val, grad.channel)
            arr = np.matmul(rot, arr)
            for n in range(3): # 3 axes - x, y, z
                val = getattr(output[n], tag)
                setattr(output[n], tag, arr[n].item()+val)

    return output
