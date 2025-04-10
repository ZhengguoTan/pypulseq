import re
import warnings
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from pypulseq.calc_duration import calc_duration
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels_rf_use import get_supported_labels


def read(self, path: str, detect_rf_use: bool = False, remove_duplicates: bool = True) -> None:
    """
    Load sequence from file - read the given filename and load sequence data into sequence object.

    See also `pypulseq.Sequence.write_seq.write()`.

    Parameters
    ----------
    path : Path
        Path of sequence file to be read.
    detect_rf_use : bool, default=False
        Boolean flag to let the function infer the currently missing flags concerning the intended use of the RF pulses
        (excitation, refocusing, etc). These are important for the k-space trajectory calculation.
    remove_duplicates: bool, default=True
        Remove duplicate events from the sequence after reading

    Raises
    ------
    FileNotFoundError
        If no sequence file is found at `path`.
    RuntimeError
        If incompatible sequence files are attempted to be loaded.
    ValueError
        If unexpected sections are encountered when loading a sequence file.
    """
    try:
        input_file = open(path, 'r')
    except FileNotFoundError as e:
        raise FileNotFoundError(e) from e

    # Event libraries
    self.adc_library = EventLibrary()
    self.grad_library = EventLibrary()
    self.label_inc_library = EventLibrary()
    self.label_set_library = EventLibrary()
    self.rf_library = EventLibrary()
    self.shape_library = EventLibrary()
    self.trigger_library = EventLibrary()

    # Raster times
    self.grad_raster_time = self.system.grad_raster_time
    self.rf_raster_time = self.system.rf_raster_time

    self.block_events = {}
    self.definitions = {}
    self.extension_string_idx = []
    self.extension_numeric_idx = []

    jemris_generated = False
    version_combined = 0

    # Load data from file
    while True:
        section = __skip_comments(input_file)
        if section == -1:
            break
        if section == '[DEFINITIONS]':
            self.definitions = __read_definitions(input_file)

            # Gradient raster time
            if 'GradientRasterTime' in self.definitions:
                self.gradient_raster_time = self.definitions['GradientRasterTime']

            # Radio frequency raster time
            if 'RadiofrequencyRasterTime' in self.definitions:
                self.rf_raster_time = self.definitions['RadiofrequencyRasterTime']

            # ADC raster time
            if 'AdcRasterTime' in self.definitions:
                self.adc_raster_time = self.definitions['AdcRasterTime']

            # Block duration raster
            if 'BlockDurationRaster' in self.definitions:
                self.block_duration_raster = self.definitions['BlockDurationRaster']
            else:
                warnings.warn(f'No BlockDurationRaster found in file. Using default of {self.block_duration_raster}.')

        elif section == '[JEMRIS]':
            jemris_generated = True
        elif section == '[SIGNATURE]':
            temp_sign_defs = __read_definitions(input_file)
            if 'Type' in temp_sign_defs:
                self.signature_type = temp_sign_defs['Type']
            if 'Hash' in temp_sign_defs:
                self.signature_value = temp_sign_defs['Hash']
                self.signature_file = 'Text'
        elif section == '[VERSION]':
            version_major, version_minor, version_revision = __read_version(input_file)

            if version_major != self.version_major:
                raise RuntimeError(f'Unsupported version_major: {version_major}. Expected: {self.version_major}')

            version_combined = 1000000 * version_major + 1000 * version_minor + version_revision

            if version_combined < 1002000:
                raise RuntimeError(
                    f'Unsupported version {version_major}.{version_minor}.{version_revision}, only file '
                    f'format revision 1.2.0 and above are supported.'
                )

            if version_combined < 1003001:
                raise RuntimeError(
                    f'Loading older Pulseq format file (version '
                    f'{version_major}.{version_minor}.{version_revision}) some code may function not as '
                    f'expected'
                )
        elif section == '[BLOCKS]':
            if version_major == 0:
                raise RuntimeError('Pulseq file MUST include [VERSION] section prior to [BLOCKS] section')
            result = __read_blocks(
                input_file,
                block_duration_raster=self.block_duration_raster,
                version_combined=version_combined,
            )
            self.block_events, self.block_durations, delay_ind_temp = result
        elif section == '[RF]':
            if jemris_generated:
                self.rf_library = __read_events(input_file, (1, 1, 1, 1, 1), event_library=self.rf_library)
            else:
                if version_combined >= 1004000:  # 1.4.x format
                    self.rf_library = __read_events(
                        input_file,
                        (1, 1, 1, 1, 1e-6, 1, 1),
                        event_library=self.rf_library,
                    )
                else:  # 1.3.x and below
                    self.rf_library = __read_events(input_file, (1, 1, 1, 1e-6, 1, 1), event_library=self.rf_library)
        elif section == '[GRADIENTS]':
            if version_combined >= 1004000:  # 1.4.x format
                self.grad_library = __read_events(input_file, (1, 1, 1, 1e-6), 'g', self.grad_library)
            else:  # 1.3.x and below
                self.grad_library = __read_events(input_file, (1, 1, 1e-6), 'g', self.grad_library)
        elif section == '[TRAP]':
            if jemris_generated:
                self.grad_library = __read_events(input_file, (1, 1e-6, 1e-6, 1e-6), 't', self.grad_library)
            else:
                self.grad_library = __read_events(input_file, (1, 1e-6, 1e-6, 1e-6, 1e-6), 't', self.grad_library)
        elif section == '[ADC]':
            self.adc_library = __read_events(
                input_file, (1, 1e-9, 1e-6, 1, 1), event_library=self.adc_library, append=self.system.adc_dead_time
            )
        elif section == '[DELAYS]':
            if version_combined >= 1004000:
                raise RuntimeError('Pulseq file revision 1.4.0 and above MUST NOT contain [DELAYS] section')
            temp_delay_library = __read_events(input_file, (1e-6,))
        elif section == '[SHAPES]':
            self.shape_library = __read_shapes(input_file, version_major == 1 and version_minor < 4)
        elif section == '[EXTENSIONS]':
            self.extensions_library = __read_events(input_file)
        else:
            if section[:18] == 'extension TRIGGERS':
                extension_id = int(section[18:])
                self.set_extension_string_ID('TRIGGERS', extension_id)
                self.trigger_library = __read_events(input_file, (1, 1, 1e-6, 1e-6), event_library=self.trigger_library)
            elif section[:18] == 'extension LABELSET':
                extension_id = int(section[18:])
                self.set_extension_string_ID('LABELSET', extension_id)

                def l1(s):
                    return int(s)

                def l2(s):
                    return get_supported_labels().index(s) + 1

                self.label_set_library = __read_and_parse_events(input_file, l1, l2)
            elif section[:18] == 'extension LABELINC':
                extension_id = int(section[18:])
                self.set_extension_string_ID('LABELINC', extension_id)

                def l1(s):
                    return int(s)

                def l2(s):
                    return get_supported_labels().index(s) + 1

                self.label_inc_library = __read_and_parse_events(input_file, l1, l2)
            else:
                raise ValueError(f'Unknown section code: {section}')

    input_file.close()  # Close file

    if version_combined < 1002000:
        raise ValueError(
            f'Unsupported version {version_combined}, only file format revision 1.2.0 (1002000) and above '
            f'are supported.'
        )

    # Fix blocks, gradients and RF objects imported from older versions
    if version_combined < 1004000:
        # Scan through RF objects
        for i in self.rf_library.data:
            self.rf_library.update(i, None, (*self.rf_library.data[i][:3], 0, *self.rf_library.data[i][3:]))

        # Scan through the gradient objects and update 't'-s (trapezoids) und 'g'-s (free-shape gradients)
        for i in self.grad_library.data:
            if self.grad_library.type[i] == 't':
                if self.grad_library.data[i][1] == 0:  # noqa: SIM102
                    if abs(self.grad_library.data[i][0]) == 0 and self.grad_library.data[i][2] > 0:
                        d = self.grad_library.data[i]
                        self.grad_library.update(
                            i,
                            None,
                            (d[0], self.grad_raster_time, d[2] - self.grad_raster_time) + d[3:],
                            self.grad_library.type[i],
                        )

                if self.grad_library.data[i][3] == 0:  # noqa: SIM102
                    if abs(self.grad_library.data[i][0]) == 0 and self.grad_library.data[i][2] > 0:
                        d = self.grad_library.data[i]
                        self.grad_library.update(
                            i,
                            None,
                            d[:2] + (d[2] - self.grad_raster_time, self.grad_raster_time) + d[4:],
                            self.grad_library.type[i],
                        )

            if self.grad_library.type[i] == 'g':
                self.grad_library.update(
                    i,
                    None,
                    (
                        self.grad_library.data[i][:2],
                        0,
                        self.grad_library.data[i][2:],
                    ),
                    self.grad_library.type[i],
                )

        # For versions prior to 1.4.0 block_durations have not been initialized
        self.block_durations = {}
        # Scan through blocks and calculate durations
        for block_counter in self.block_events:
            # Insert delay as temporary block_duration
            self.block_durations[block_counter] = 0
            if delay_ind_temp[block_counter] > 0:
                self.block_durations[block_counter] = temp_delay_library.data[delay_ind_temp[block_counter]][0]

            block = self.get_block(block_counter)
            # Calculate actual block duration
            self.block_durations[block_counter] = calc_duration(block)

    # TODO: Is it possible to avoid expensive get_block calls here?
    grad_channels = ['gx', 'gy', 'gz']
    grad_prev_last = np.zeros(len(grad_channels))
    for block_counter in self.block_events:
        block = self.get_block(block_counter)
        block_duration = block.block_duration
        # We also need to keep track of the event IDs because some PyPulseq files written by external software may contain
        # repeated entries so searching by content will fail
        event_idx = self.block_events[block_counter]
        # Update the objects by filling in the 'first' and 'last' attributes not yet contained in the Pulseq file
        for j in range(len(grad_channels)):
            grad = getattr(block, grad_channels[j])
            if grad is None:
                grad_prev_last[j] = 0
                continue

            if grad.type == 'grad':
                if grad.delay > 0:
                    grad_prev_last[j] = 0

                # go to next channel, if grad.first and grad.last are already set
                if hasattr(grad, 'first') and hasattr(grad, 'last'):
                    grad_prev_last[j] = grad.last
                    continue

                # get grad.first and grad.last attributes from the grad_library if they have been set for the current amplitude_ID before
                amplitude_ID = event_idx[j + 2]
                if amplitude_ID in event_idx[2 : (j + 2)]:
                    if self.use_block_cache:
                        grad.first = self.grad_library.data[amplitude_ID][4]
                        grad.last = self.grad_library.data[amplitude_ID][5]
                    continue

                # get time_id from grad_library
                time_id = self.grad_library.data[amplitude_ID][2]

                # if grad.first is not set, set it to the last value of the previous gradient
                grad.first = grad_prev_last[j]

                # extended trapezoid: use last value of the gradient waveform as grad.last
                if time_id != 0:
                    grad.last = grad.waveform[-1]
                    grad_duration = grad.delay + grad.tt[-1]
                # arbitrary gradients: interpolate grad.last from the gradient waveform
                else:
                    # use a linear extrapolation identical to the one used in the make_arbitrary_grad.py file
                    grad.last = (3 * grad.waveform[-1] - grad.waveform[-2]) * 0.5
                    grad_duration = grad.delay + len(grad.waveform) * self.grad_raster_time

                # Set grad_prev_last to 0 if gradient does not end at block boundary
                eps = np.finfo(np.float64).eps
                if grad_duration + eps < block_duration:
                    grad_prev_last[j] = 0
                # Update grad_prev_last for the next iteration if gradient ends at block boundary
                else:
                    grad_prev_last[j] = grad.last

                # Update the grad_library with the new grad.first and grad.last values
                amplitude = self.grad_library.data[amplitude_ID][0]
                shape_id = self.grad_library.data[amplitude_ID][1]
                new_data = (
                    amplitude,
                    shape_id,
                    time_id,
                    grad.delay,
                    grad.first,
                    grad.last,
                )
                self.grad_library.update_data(amplitude_ID, None, new_data, 'g')

            else:
                grad_prev_last[j] = 0

    if detect_rf_use:
        # Find the RF pulses, list flip angles, and work around the current (rev 1.2.0) Pulseq file format limitation
        # that the RF pulse use is not stored in the file
        for k in self.rf_library.data:
            lib_data = self.rf_library.data[k]
            rf = self.rf_from_lib_data(lib_data)
            flip_deg = np.abs(np.sum(rf.signal[:-1] * (rf.t[1:] - rf.t[:-1]))) * 360
            offresonance_ppm = 1e6 * rf.freq_offset / self.system.B0 / self.system.gamma
            if flip_deg < 90.01:  # Add 0.01 degree to account for rounding errors encountered in very short RF pulses
                self.rf_library.type[k] = 'e'
            else:
                if rf.shape_dur > 6e-3 and -3.5 <= offresonance_ppm <= -3.4:  # Approx -3.45
                    self.rf_library.type[k] = 's'  # Saturation (fat-sat)
                else:
                    self.rf_library.type[k] = 'r'
            self.rf_library.data[k] = lib_data

            # Clear block cache for all blocks that contain the modified RF event
            for block_counter, events in self.block_events.items():
                if events[1] == k:
                    del self.block_cache[block_counter]

    # When removing duplicates, remove and remap events in the sequence without
    # creating a copy.
    if remove_duplicates:
        self.remove_duplicates(in_place=True)


def __read_definitions(input_file) -> Dict[str, str]:
    """
    Read the [DEFINITIONS] section of a sequence fil and return a map of key/value entries.

    Parameters
    ----------
    input_file : file object
        Sequence file.

    Returns
    -------
    definitions : dict{str, str}
        Dict object containing key value pairs of definitions.
    """
    definitions = {}
    line = __skip_comments(input_file)
    while line != -1 and not (line == '' or line[0] == '#'):
        tok = line.split(' ')
        try:  # Try converting every element into a float
            [float(x) for x in tok[1:]]
            value = np.array(tok[1:], dtype=float)
            if len(value) == 1:  # Avoid array structure for single elements
                value = value[0]
            definitions[tok[0]] = value
        except ValueError:  # Try clause did not work!
            definitions[tok[0]] = line[len(tok[0]) + 1 :].strip()
        line = __strip_line(input_file)

    return definitions


def __read_version(input_file) -> Tuple[int, int, int]:
    """
     Read the [VERSION] section of a sequence file.

    Parameters
    ----------
    input_file : file object
        Sequence file.

    Returns
    -------
    tuple
        Major, minor and revision number.
    """
    line = __strip_line(input_file)
    major, minor, revision = 0, 0, 0
    while line != '' and line[0] != '#':
        tok = line.split(' ')
        if tok[0] == 'major':
            major = int(tok[1])
        elif tok[0] == 'minor':
            minor = int(tok[1])
        elif tok[0] == 'revision':
            if len(tok[1]) != 1:  # Example: x.y.zpostN
                tok[1] = tok[1][0]
            revision = int(tok[1])
        else:
            raise RuntimeError(f'Incompatible version. Expected: {major}{minor}{revision}')
        line = __strip_line(input_file)

    return major, minor, revision


def __read_blocks(
    input_file, block_duration_raster: float, version_combined: int
) -> Tuple[Dict[int, np.ndarray], List[float], List[int]]:
    """
    Read the [BLOCKS] section of a sequence file and return the event table.

    Parameters
    ----------
    input_file : file
        Sequence file

    Returns
    -------
    event_table : dict
        Dict object containing key value pairs of Pulseq block ID and block definition.
    block_durations : list
        Block durations.
    delay_idx : list
        Delay IDs.
    """
    event_table = {}
    block_durations = {}
    delay_idx = {}
    line = __strip_line(input_file)

    while line != '' and line != '#':
        block_events = np.fromstring(line, dtype=int, sep=' ')

        if version_combined <= 1002001:
            event_table[block_events[0]] = np.array([0, *block_events[2:], 0])
        else:
            event_table[block_events[0]] = np.array([0, *block_events[2:]])

        delay_id = block_events[0]
        if version_combined >= 1004000:
            block_durations[delay_id] = block_events[1] * block_duration_raster
        else:
            delay_idx[delay_id] = block_events[1]

        line = __strip_line(input_file)

    return event_table, block_durations, delay_idx


def __read_events(
    input_file, scale: tuple = (1,), event_type: str = str(), event_library: EventLibrary = None, append=None
) -> EventLibrary:
    """
    Read an event section of a sequence file and return a library of events.

    Parameters
    ----------
    input_file : file object
        Sequence file.
    scale : list, default=(1,)
        Scale elements according to column vector scale.
    event_type : str, default=str()
        Attach the type string to elements of the library.
    event_library : EventLibrary, default=EventLibrary()
        Append new events to the given library.

    Returns
    -------
    event_library : EventLibrary
        Event library containing Pulseq events.
    """
    if event_library is None:
        event_library = EventLibrary()
    line = __strip_line(input_file)

    while line != '' and line != '#':
        data = np.fromstring(line, dtype=float, sep=' ')
        event_id = data[0]
        data = tuple(data[1:] * scale)
        if append is not None:
            data = (*data, append)
        if event_type == '':
            event_library.insert(key_id=event_id, new_data=data)
        else:
            event_library.insert(key_id=event_id, new_data=data, data_type=event_type)
        line = __strip_line(input_file)

    return event_library


def __read_and_parse_events(input_file, *args: callable) -> EventLibrary:
    """
    Read an event section of a sequence file and return a library of events. Event data elements are converted using
    the provided parser(s). Default parser is `int()`.

    Parameters
    ----------
    input_file : file
    args : callable
        Event parsers.

    Returns
    -------
    EventLibrary
        Library of events parsed from the events section of a sequence file.
    """
    event_library = EventLibrary()
    line = __strip_line(input_file)

    while line != '' and line != '#':
        list_of_data_str = re.split(r'(\s+)', line)
        list_of_data_str = [d for d in list_of_data_str if d != ' ']
        data = np.zeros(len(list_of_data_str) - 1, dtype=np.int32)
        event_id = int(list_of_data_str[0])
        for i in range(1, len(list_of_data_str)):
            if i > len(args):
                data[i - 1] = int(list_of_data_str[i])
            else:
                data[i - 1] = args[i - 1](list_of_data_str[i])
        event_library.insert(key_id=event_id, new_data=data)
        line = __strip_line(input_file)

    return event_library


def __read_shapes(input_file, force_convert_uncompressed: bool) -> EventLibrary:
    """
    Read the [SHAPES] section of a sequence file and return a library of shapes.

    Parameters
    ----------
    input_file : file

    Returns
    -------
    shape_library : EventLibrary
        `EventLibrary` object containing shape definitions.
    """
    shape_library = EventLibrary(numpy_data=True)

    line = __skip_comments(input_file)

    while line != -1 and (line != '' or line[0:8] == 'shape_id'):
        tok = line.split(' ')
        shape_id = int(tok[1])
        line = __skip_comments(input_file)
        tok = line.split(' ')
        num_samples = int(tok[1])
        data = []
        line = __skip_comments(input_file)
        while line != '' and line != '#':
            data.append(float(line))
            line = __strip_line(input_file)
        line = __skip_comments(input_file, stop_before_section=True)

        # Check if conversion is needed: in v1.4.x we use length(data)==num_samples
        # As a marker for the uncompressed (stored) data. In older versions this condition could occur by chance
        if force_convert_uncompressed and len(data) == num_samples:
            shape = SimpleNamespace()
            shape.data = data
            shape.num_samples = num_samples
            shape = compress_shape(decompress_shape(shape, force_decompression=True))
            data = np.array([shape.num_samples, *shape.data])
        else:
            data.insert(0, num_samples)
            data = np.asarray(data)
        shape_library.insert(key_id=shape_id, new_data=data)
    return shape_library


def __skip_comments(input_file, stop_before_section: bool = False) -> str:
    """
    Read lines of skipping blank lines and comments and return the next non-comment line.

    Parameters
    ----------
    input_file : file

    Returns
    -------
    line : str
        First line in `input_file` after skipping one '#' comment block. Note: File pointer is remembered, so
        successive calls work as expected.
    """
    temp_pos = input_file.tell()
    line = __strip_line(input_file)
    while line != -1 and (line == '' or line[0] == '#'):
        temp_pos = input_file.tell()
        line = __strip_line(input_file)

    if line != -1:
        if stop_before_section and line[0] == '[':
            input_file.seek(temp_pos, 0)
            next_line = ''
        else:
            next_line = line
    else:
        next_line = -1

    return next_line


def __strip_line(input_file) -> str:
    """
    Removes spaces and newline whitespaces.

    Parameters
    ----------
    input_file : file

    Returns
    -------
    line : str
        First line in input_file after spaces and newline whitespaces have been removed. Note: File pointer is
        remembered, and hence successive calls work as expected. Returns -1 for eof.
    """
    line = input_file.readline()  # If line is an empty string, end of the file has been reached
    return line.strip() if line != '' else -1
