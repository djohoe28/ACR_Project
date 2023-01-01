#  Copyright (c) 2022. Code by DJohoe28.

import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class Track (object):
    """
    A Track loaded into the program.

    :cvar class_name: The name of the class
    :type class_name: str
    :ivar path: The path (relative/absolute) to the file represented by the instance
    :type path: str
    :ivar title: The title of the track  # NOTE: Currently the file name, could be the track title
    :type title: str
    :ivar y: The time-series values of the track - its numerical representation
    :type y: np.ndarray
    :ivar sr: The Sample Rate of the track - the timespan used by y
    :type sr: int
    """
    class_name: str = "Track"
    """  # NOTE: Currently disabled, as these are instance variables.
    path: str
    title: str
    y: np.ndarray
    sr: int
    """

    def __init__(self: class_name, path: str = "", load: bool = True) -> None:
        """
        Track Constructor.

        :param path: Path to file
        :type path: str
        :param load: If true, loads file - else, leaves values uninitialized.
        :type load: bool
        """
        self.path: str = path
        """Path to Track's file."""
        self.title: str = ""
        """File name / title of Track."""
        self.y: np.ndarray = np.zeros(1)
        """Audio time series of Track (y values over samples)."""
        self.sr: int = 0
        """Sample rate of Track."""

        # Determine title.
        if "/" not in self.path[0] and "\\" not in self.path[0]:
            self.title = path  # File is saved locally.
        else:
            self.title = os.path.basename(self.path)

        # Load file (if load=true).
        if load:
            self.y, self.sr = librosa.load(self.path)

    def __str__(self: class_name):
        """A string description of the Track."""
        return f"Track: {self.title}"

    def __repr__(self: class_name):
        """A string representation of the Track."""
        return f"Track: {self.path}"

    def plot(self: class_name, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the waveform of self to the given axis if available, else plot to new axis.

        :param ax: The axis of the figure on which to plot the waveform
        :type ax: plt.Axes
        :return: Axis on which the waveform was plotted
        :rtype: plt.Axes
        """
        if ax is None:
            _fig, _ax = plt.subplots()  # {fig} is only used as tuple filler.
        else:
            _ax = ax
        librosa.display.waveshow(y=self.y, sr=self.sr, ax=ax)
        _ax.set(title=f"Waveform: {self.title}")
        _ax.label_outer()
        return _ax

    def get_stft(self: class_name) -> np.ndarray:
        """
        Get the Short-Time Fourier Transform (STFT) of the Track.

        :return: Track's STFT.
        :rtype: np.ndarray
        """
        return librosa.stft(self.y)

    def plot_stft(self: class_name):
        """
        Plot the STFT of the track in a new window.

        :return: None
        :rtype: NoneType
        """
        db = librosa.amplitude_to_db(np.abs(self.get_stft()), ref=np.max)
        librosa.display.specshow(db, sr=self.sr, y_axis='log', x_axis='time')

    def get_beat_times(self: class_name) -> np.ndarray:
        """
        Get an array of timestamps in which a beat was detected.

        :return: Array of beat timestamps.
        :rtype: np.ndarray
        """
        _tempo, _beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
        _beat_times = librosa.frames_to_time(_beat_frames, sr=self.sr)  # Frame indices of beat events -> Timestamps
        return _beat_times

    def plot_beat_times(self: class_name, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the waveform (with beat indicators) of self to the given axis if available, else plot to new axis.

        :param ax: The axis of the figure on which to plot the waveform
        :type ax: plt.Axes
        :return: Axis on which the waveform was plotted
        :rtype: plt.Axes
        """
        ax = self.plot(ax)
        beats = self.get_beat_times()
        [ax.axvline(x=_x, color='k', lw=0.5, linestyle='dashed') for _x in beats]  # NOTE: Plot v-lines @ beat x-values.
        return ax


track1 = Track(librosa.ex('choice', hq=True))
track2 = Track(librosa.example('nutcracker'))
track1.plot_beat_times()  # TODO: Seems to miss beats near the end of the file. Intentional?
plt.show()
track2.plot_stft()
plt.show()
