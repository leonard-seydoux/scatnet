#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read seismic data."""

import glob
import h5py
import logging
import numpy as np
import os
import obspy
import warnings

from matplotlib import pyplot as plt
from matplotlib import dates as md
from obspy.core.trace import Trace
from scipy.signal import hanning
from sklearn.feature_extraction.image import extract_patches as buff
from statsmodels import robust

warnings.filterwarnings("ignore", category=FutureWarning)


def read_data(file_data='', patch_shape=2048, decimation=0, channels=None,
              trim=None, extraction_step=None, batch_size=None,
              date_fmt='%y-%m-%d %H:%M', **kwargs):
    """Read seismic data.

    Makes use of the `covnet.data.read` function family depending on
    the data format. Refer to the documentation therein for input arguments.

    Returns
    -------
        data: np.ndarray
            Data of shape (batch, channels, patch_shape) normalized by maximum.
        times: np.ndarray
            Time vector (1, n_samples) in datetime format.

    """
    # Define read method based on file formats.
    file_format = os.path.splitext(file_data)[-1]
    if file_format == 'h5':
        reader = h5read
        kwargs.update({'trim': trim})
        kwargs.update({'channel': channels[0]})
    elif file_format == 'mat':
        reader = matread
    else:
        reader = read

    # Get list of files
    files = sorted(glob.glob(file_data))

    # Read
    stream = reader(files[0], **kwargs)
    if channels is not None:
        if len(channels) > 1:
            for channel in channels[1:]:
                kwargs.update({'channel': channel})
                stream += reader(files[0], **kwargs)
    for f in files[1:]:
        if channels is None:
            stream += reader(f, **kwargs)
        else:
            if len(channels) > 1:
                for channel in channels[1:]:
                    kwargs.update({'channel': channel})
                    stream += reader(f, **kwargs)

    # Merge and trim data
    stream.merge(method=1)
    ti = [md.num2date(t).strftime(date_fmt) for t in stream.times[[0, -1]]]
    if trim is not None:
        stream.cut(*trim)

    # Decimate
    if decimation > 0:
        stream.decimate(decimation)

    # Speak.
    logging.info('{} (done)'.format(file_data))
    logging.info('init dates {} to {}'.format(*ti))

    return stream


def read(*args):
    """(Top-level). Read the data files specified in the datapath.

    This method uses the obspy's read method itself. A check for the
    homogeneity of the seismic traces (same number of samples) is done a
    the end. If the traces are not of same size, an warning message
    shows up.

    No homogeneity check is returned by the function.

    Arguments:
    ----------
        data_path (str or list): path to the data. Path to a single file
            (str), to several files using UNIX regexp (str), or a list of
            files (list). See obspy.read method for more details.
    """
    data = Stream()
    data.read(*args)
    return data


def h5read(*args, **kwargs):
    """Top-level read function, returns Stream object."""
    data = Stream()
    data.h5read(*args, **kwargs)
    return data


def matread(*args, **kwargs):
    """Top-level read function, returns Stream object."""
    data = Stream()
    data.matread(*args, **kwargs)
    return data


class Stream(obspy.core.stream.Stream):
    """A stream obsject with additional methods."""

    def __init__(self, *args, **kwargs):
        """Wrapper for obspy stream object."""
        super(Stream, self).__init__(*args, **kwargs)

    @property
    def times(self):
        """Extract time from the first trace and return matplotlib time."""
        # Obspy times are seconds from starttime
        times = self[0].times()

        # Turn into day fraction
        times /= 24 * 3600

        # Add matplotlib starttime in day fraction
        start = self[0].stats.starttime.datetime
        times += md.date2num(start)

        return times

    @property
    def stations(self):
        """List all the station names extracted from each trace."""
        return [s.stats.station for s in self]

    def read(self, data_path):
        """Read the data files specified in the datapath with obspy.

        This method uses the obspy's read method itself. A check for the
        homogeneity of the seismic traces (same number of samples) is done a
        the end. If the traces are not of same size, an warning message
        shows up.

        Arguments:
        ----------
            data_path (str or list): path to the data. Path to a single file
                (str), to several files using UNIX regexp (str), or a list of
                files (list). See obspy.read method for more details.

        Return:
        -------
            homogeneity (bool): True if the traces are all of the same size.

        """
        # If data_path is a str, then only a single trace or a bunch of
        # seismograms with regexp
        if isinstance(data_path, str):
            self += obspy.read(data_path)

        # If data_path is a list, read each fils in the list.
        elif isinstance(data_path, list):
            for index, path in enumerate(data_path):
                self += obspy.read(path)

    def h5read(self, path_h5, net='PZ', force_start=None, stations=None,
               channel='Z', trim=None):
        """Read seismograms in h5 format.

        Parameters
        ----------
            path_h5 : str
                Path to the h5 file.

        Keyword arguments
        -----------------
            net : str
                Name of the network to read.
            force_start : str
                The date at which the seismograms are supposed to start.
                Typically this is at midnight at a given day.
            channel : str
                The channel to extract; either "Z", "E" or "N"
            stations : list
                A list of desired seismic stations.
            trim : list
                A list of trim dates as strings. This allows to extract only
                a small part of the seismograms without loading the full day,
                and therefore to considerably improve the reading efficiency.
        """
        # Open file
        h5file = h5py.File(path_h5, 'r')

        # Meta data
        # ---------

        # Initialize stream header
        stats = obspy.core.trace.Stats()

        # Sampling rate
        sampling_rate = np.array(h5file['_metadata']['fe'])
        stats.sampling_rate = sampling_rate
        stats.channel = channel

        # Starting time
        if force_start is None:
            start = np.array(h5file['_metadata']['t0_UNIX_timestamp'])
            stats.starttime = obspy.UTCDateTime(start)
        else:
            stats.starttime = obspy.UTCDateTime(force_start)

        # Station codes
        if stations is None:
            station_codes = [k for k in h5file[net].keys()]
        else:
            station_codes = [k for k in h5file[net].keys() if k in stations]

        # Data extaction
        # --------------

        # Define indexes of trim dates in order to extract the time segments.
        # This modifies the start time to the start trim date.
        if trim is not None:
            i_start = int(obspy.UTCDateTime(trim[0]) - stats.starttime)
            i_end = int(obspy.UTCDateTime(trim[1]) - stats.starttime)
            i_start *= int(sampling_rate)
            i_end *= int(sampling_rate)
            stats.starttime = obspy.UTCDateTime(trim[0])
        else:
            i_start = 0
            i_end = -1

        # Collect data into stream
        for station, station_code in enumerate(station_codes):

            # Tries to read the data for a given station. This raises
            # a KeyError if the station has no data at this date.
            try:

                # Read data
                data = h5file[net][station_code][channel][i_start:i_end]

                # Include the specs of this trace in the corresponding header
                stats.npts = len(data)
                stats.station = station_code.split('.')[0]

                # Add to the main stream
                self += obspy.core.trace.Trace(data=data, header=stats)

            # If no data is present for this day at this station, nothing is
            # added to the stream. This may change the number of available
            # stations at different days.
            except KeyError:
                continue

    def matread(self, data_path, data_name='data', starttime=0,
                sampling_rate=25.0, decimate=1):
        """
        Read the data files specified in the datapath.

        Arguments
        ---------
        :datapath (str or list): datapath with a single data file or with
        UNIX regexp, or a list of files.

        Keyword arguments
        -----------------

        :sort (bool): whether or not the different traces are sorted in
        alphabetic order with respect to the station codes.
        """
        # Read meta
        traces = np.array(h5py.File(data_path, 'r')[data_name])
        n_stations, n_times = traces.shape

        # Header
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = sampling_rate
        stats.npts = n_times

        # Start time
        stats.starttime = obspy.UTCDateTime(starttime)

        # Collect data into data np.array
        for station in range(0, n_stations, decimate):
            data = traces[station, :]
            self += obspy.core.trace.Trace(data=data, header=stats)

    def set_data(self, data_matrix, starttime, sampling_rate):
        """Set the data from any external set of traces."""
        n_traces, n_times = data_matrix.shape

        # Header
        stats = obspy.core.trace.Stats()
        stats.sampling_rate = sampling_rate
        stats.starttime = obspy.UTCDateTime(starttime)
        stats.npts = n_times

        # Assign
        for trace_id, trace in enumerate(data_matrix):
            self += Trace(data=trace, header=stats)

    def cut(self, starttime, endtime, pad=True, fill_value=0):
        """Cut seismic traces between given start and end times.

        A wrapper to the :meth:`obspy.Stream.trim` method with string dates or
        datetimes.

        Parameters
        ----------
        starttime : str
            The starting date time.

        endtime : str
            The ending date time.

        Keyword arguments
        -----------------

        pad : bool
            Whether the data has to be padded if the starting and ending times
            are out of boundaries.

        fill_value : int, float or str
            Specifies the values to use in order to fill gaps, or pad the data
            if ``pad`` is set to True.

        """
        # Convert date strings to obspy UTCDateTime
        starttime = obspy.UTCDateTime(starttime)
        endtime = obspy.UTCDateTime(endtime)

        # Trim
        self.trim(starttime=starttime, endtime=endtime, pad=pad,
                  fill_value=fill_value)

    def batch(self, patch_shape=2048, batch_size=None, extraction_step=None,
              date_fmt='%y-%m-%d %H:%M', layers_kw=None):
        """Make baches from data."""
        if extraction_step is None:
            extraction_step = patch_shape

        # Make buffer matrix
        data = list()
        n_channels = len(self)
        for trace in self:

            # Normalize by maximum in order to avoid badly conditionned.
            signal = trace.data
            signal[np.isnan(signal)] = 0
            n_samples = len(signal)

            # Turn the signal into segments
            signal = buff(signal, patch_shape=patch_shape,
                          extraction_step=extraction_step)
            data.append(signal)
        data = np.array(data).transpose([1, 0, 2])

        assert np.all(data[0, 0, :] == self[0].data[:patch_shape])

        # Extract times
        times = self.times
        times = buff(times, patch_shape=patch_shape,
                     extraction_step=extraction_step)

        # Skip the first few segments in order to include the end of the signal
        if batch_size is not None:
            n_batches = data.shape[0] // batch_size
            skip = data.shape[0] - n_batches * batch_size
            data = data[skip:]
            times = times[skip:]

        # Pool times according to scattering network desin.
        if layers_kw is not None:
            times = times[:, ::layers_kw['decimation']]
            times = times[:, ::layers_kw['pooling'] // layers_kw['decimation']]
            times = times.reshape(-1)

        # Normalize data
        data /= data.max()

        # Logging
        t1 = [md.num2date(t).strftime(date_fmt) for t in self.times[[0, -1]]]
        t2 = [md.num2date(t).strftime(date_fmt) for t in times[[0, -1]]]
        shape_in = n_channels, n_samples
        shape_out = data.shape
        logging.info('from shape {} to {}'.format(shape_in, shape_out))
        logging.info('trim dates {} to {}'.format(*t1))
        logging.info('used dates {} to {}'.format(*t2))

        return data, times

    def homogenize(self, sampling_rate=20.0, method='linear',
                   start='2010-01-01', npts=24 * 3600 * 20):
        """Trim seismic data.

        Same prototype than homogenize but allows for defining the date in str
        format (instead of UTCDateTime).
        Same idea than with the cut method.
        """
        start = obspy.UTCDateTime(start)
        self.interpolate(sampling_rate, method, start, npts)

    def demad(self):
        r"""Normalize traces by their mean absolute deviation (MAD).

        The Mean Absolute Deviation :math:`m_i` of the trace :math:`i`
        describe the deviation of the data from its average :math:`\\bar{x}_i`
        obtained by the formula

        .. math::
            m_i = \\frac{1}{K}\\sum_{k=1}^{K}|x_i[k] - \\bar{x}_i|,

        where :math:`k` is the time index of the sampled trace. Each trace
        :math:x_i` is dvided by its corresponding MAD :math:`m_i`. This has
        the main effect to have the same level of background noise on each
        stream.

        """
        # Waitbar initialization
        # Binarize
        for index, trace in enumerate(self):
            mad = robust.mad(trace.data)
            if mad > 0:
                trace.data /= mad
            else:
                trace.data /= (mad + 1e-5)

    def show(self, ax=None, scale=.5, index=0, ytick_size=6, **kwargs):
        """Plot all seismic traces.

        The date axis is automatically defined with Matplotlib's numerical
        dates.

        Keyword arguments
        -----------------
        ax : :class:`matplotlib.axes.Axes`
            Previously instanciated axes. Default to None, and the axes are
            created.

        scale : float
            Scaling factor for trace amplitude.

        ytick_size : int
            The size of station codes on the left axis. Default 6 pts.

        kwargs : dict
            Other keyword arguments passed to
            :func:`matplotlib.pyplot.plot`.

        Return
        ------
        :class:`matplotlib.axes.Axes`
            The axes where the traces have been plotted.

        """
        # Parameters
        # ----------

        # Default parameters
        times = self.times
        kwargs.setdefault('rasterized', True)

        # Axes
        if ax is None:
            _, ax = plt.subplots(1, figsize=(7, 6))

        # Preprocess
        # ----------

        # Turn into array and normalize by multiple of max MAD
        traces = np.array([s.data for s in self])
        traces = traces / traces.max()
        if robust.mad(traces).max() > 0:
            traces /= robust.mad(traces).max()
        traces[np.isnan(traces)] = .0
        traces *= scale

        # Display
        # -------

        # Plot traces
        for index, trace in enumerate(traces):
            ax.plot(times, trace + index + 1, **kwargs)

        # Show station codes as yticks
        yticks = [' '] + [s.stats.station for s in self] + [' ']
        ax.set_yticks(range(len(self) + 2))
        ax.set_ylim([0, len(self) + 1])
        ax.set_ylabel('Seismic station code')
        ax.set_yticklabels(yticks, size=6)

        # Time axis
        ax.set_xlim(times[0], times[-1] + (times[2] - times[0]) / 2)
        xticks = md.AutoDateLocator()
        ax.xaxis.set_major_locator(xticks)
        ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

        return ax

    def fft(self, segment_duration_sec, bandwidth=None, step=.5,
            **kwargs):
        """Compute time and frequency reprensetation of the data."""
        # Time
        len_seg = int(segment_duration_sec * self[0].stats.sampling_rate)
        len_step = int(np.floor(len_seg * step))
        times = self.times[:1 - len_seg:len_step]
        n_times = len(times)

        # Frequency
        kwargs.setdefault('n', 2 * len_seg - 1)
        n_frequencies = kwargs['n']
        frequencies = np.linspace(
            0, self[0].stats.sampling_rate, n_frequencies)

        # Calculate spectra
        spectra_shape = len(self), n_times, n_frequencies
        spectra = np.zeros(spectra_shape, dtype=complex)
        for trace_id, trace in enumerate(self):
            tr = trace.data
            for time_id in range(n_times):
                start = time_id * len_step
                end = start + len_seg
                segment = tr[start:end] * hanning(len_seg)
                spectra[trace_id, time_id] = np.fft.fft(segment, **kwargs)

        # Times are extended with last time of traces
        t_end = self.times[-1]
        times = np.hstack((times, t_end))

        return times, frequencies, spectra


def show_spectrogram(times, frequencies, spectrum, ax=None, cax=None,
                     flim=None, step=.5, figsize=(6, 5), **kwargs):
    """Pcolormesh the spectrogram of a single seismic trace.

    The spectrogram (modulus of the short-time Fourier transform) is
    extracted from the complex spectrogram previously calculated from
    the :meth:`arrayprocessing.data.stft` method.

    The spectrogram is represented in log-scale amplitude normalized by
    the maximal amplitude (dB re max).

    The date axis is automatically defined with Matplotlib's dates.

    Parameters
    ----------

    times : :class:`np.ndarray`
        The starting times of the windows

    frequencies : :class:`np.ndarray`
        The frequency vector.

    spectrum : :class:`np.ndarray`
        The selected spectrogram matrix of shape ``(n_frequencies, n_times)``

    Keyword arguments
    -----------------

    code : int or str
        Index or code of the seismic station.

    step : float
        The step between windows in fraction of segment duration.
        By default, assumes a step of .5 meaning 50% of overlap.

    ax : :class:`matplotlib.axes.Axes`
        Previously instanciated axes. Default to None, and the axes are
        created.

    cax : :class:`matplotlib.axes.Axes`
        Axes for the colorbar. Default to None, and the axes are created.
        These axes should be given if ``ax`` is not None.

    kwargs : dict
        Other keyword arguments passed to
        :func:`matplotlib.pyplot.pcolormesh`

    Return
    ------

        If the path_figure kwargs is set to None (default), the following
        objects are returned:

        fig (matplotlib.pyplot.Figure) the figure instance.
        ax (matplotlib.pyplot.Axes) axes of the spectrogram.
        cax (matplotlib.pyplot.Axes) axes of the colorbar.

    """
    # Axes
    if ax is None:
        gs = dict(width_ratios=[50, 1])
        fig, (ax, cax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gs)

    # Safe
    spectrum = np.squeeze(spectrum)

    # Spectrogram
    spectrum = np.log10(np.abs(spectrum) / np.abs(spectrum).max())

    # Frequency limits
    if flim is not None:
        f1 = np.abs(frequencies - flim[0]).argmin()
        f2 = np.abs(frequencies - flim[1]).argmin()
        frequencies = frequencies[f1:f2]
        spectrum = spectrum[f1:f2, :]

    # Image
    kwargs.setdefault('rasterized', True)
    img = ax.pcolormesh(times, frequencies, spectrum, **kwargs)

    # Colorbar
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('Spectral amplitude (dB re max)')

    # Date ticks
    ax.set_xlim(times[[0, -1]])
    xticks = md.AutoDateLocator()
    ax.xaxis.set_major_locator(xticks)
    ax.xaxis.set_major_formatter(md.AutoDateFormatter(xticks))

    # Frequencies
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(frequencies[[1, -1]])

    return ax, cax
