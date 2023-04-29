#  Copyright (c) 2022. Code by DJohoe28.
import os
from datetime import datetime
from enum import Enum
from typing import Optional, Union, Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice
import sounddevice as sd
import soundfile as sf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Initializations
DEF_REC_PATH = "Recordings"  # Default Recordings Path
DEF_DURATION = 1  # Default Duration (Chirps, Recordings, etc)
DEF_SR = 44100  # Default Sample Rate (see above)
sd.default.samplerate = DEF_SR  # Currently unused, as we know each file's sample rate
sd.default.channels = 2  # Set sound-device's default to Stereo


class Return(Enum):
    SUCCESS = 1  # The function performed successfully.
    CANCEL = 0  # The function was cancelled.
    UNDEF_ERROR = -1  # All errors not mentioned below
    INDEX_ERROR = -2  # Index (out of range) Error
    VALUE_ERROR = -3  # Value (conversion) Error
    AUDIO_ERROR = -4  # Audio (playback) Error


class Track(object):
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
        if path == "" or ("/" not in self.path[0] and "\\" not in self.path[0]):
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

    def play(self: class_name, wait: bool = True) -> None:
        """
        Play the Track.

        :param wait: If true, the program halts until the Track stops playing once
        :type wait: bool
        :return: None
        :rtype: NoneType
        """
        sd.play(self.y, self.sr)
        if wait:
            sd.wait()

    def plot(self: class_name, ax: plt.Axes = None, fig: plt.Figure = None) -> (plt.Figure, plt.Axes):
        """
        Plot the waveform of self to the given axis if available, else plot to new axis.

        :param ax: The axis of the figure on which to plot the waveform
        :type ax: plt.Axes
        :param fig: The figure on which to plot the waveform
        :type fig: plt.Figure
        :return: Figure, Axis on which the waveform was plotted
        :rtype: (plt.Figure, plt.Axes)
        """
        if ax is None:
            fig, ax = plt.subplots()  # {fig} is only used as tuple filler.
        if fig is None:
            fig = ax.get_figure()
        librosa.display.waveshow(y=self.y, sr=self.sr, ax=ax)
        ax.set(title=f"Waveform: {self.title}")
        ax.label_outer()
        return fig, ax

    def get_stft(self: class_name, n_fft=2048, hop_length=None, win_length=None) -> np.ndarray:
        """
        Get the Short-Time Fourier Transform (STFT) of the Track.

        :return: Track's STFT.
        :rtype: np.ndarray
        """
        win_length = win_length if win_length is not None else n_fft
        hop_length = hop_length if hop_length is not None else win_length // 4
        return librosa.stft(self.y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    def plot_stft(self: class_name, ax: plt.Axes = None, fig: plt.Figure = None) \
            -> tuple[Union[Figure, Any], Union[Axes, Any]]:
        """
        Plot the STFT of the track in a new window.

        :param ax: The axis of the figure on which to plot the STFT
        :type ax: plt.Axes
        :param fig: The figure on which to plot the STFT
        :type fig: plt.Figure
        :return: Figure, Axis on which the STFT was plotted
        :rtype: (plt.Figure, plt.Axes)
        """
        return plot_stft_spectrogram(self.get_stft(), sr=self.sr, ax=ax, fig=fig)

    def get_constellation_map(self: class_name, threshold=0.5, distance=10):
        """Get a copy of the track's STFT where all non-peaks are set to 0."""
        stft = np.abs(self.get_stft())  # Usually only magnitude is relevant for peak detection.
        peaks = np.zeros(stft.shape, dtype=bool)  # Boolean array: x[i,j] = true -> x[i,j] is a peak.
        for time_idx in range(stft.shape[1]):  # Loop through time axis
            time_slice_maxima = scipy.signal.find_peaks(stft[:, time_idx], threshold=threshold, distance=distance)[0]
            if len(time_slice_maxima) > 0:  # if any number of peaks was found
                for freq_idx in time_slice_maxima:
                    peaks[freq_idx, time_idx] = True
        maxima_stft = np.where(peaks, 1, 0)  # A 2D array where only maxima retain their value; else 0.
        return maxima_stft

    def plot_constellation_map(self: class_name,
                               threshold: float = 0.5,
                               distance: int = 10,
                               ax: plt.Axes = None, fig: plt.Figure = None):
        """
        Plot the Constellation Map of the track.

        :param threshold: Required threshold value for determining peaks.
        :type threshold: float
        :param distance: Required distance (between peaks) value for determining peaks.
        :type distance: int
        :param ax: The axis of the figure on which to plot the STFT
        :type ax: plt.Axes
        :param fig: The figure on which to plot the STFT
        :type fig: plt.Figure
        :return: Figure, Axis on which the STFT was plotted
        :rtype: (plt.Figure, plt.Axes)
        """
        return plot_stft_spectrogram(self.get_constellation_map(threshold=threshold, distance=distance), sr=self.sr,
                                     ax=ax, fig=fig)

    def plot_stft_and_constellation(self: class_name,
                                    threshold: float = 0.5, distance: int = 10) \
            -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the STFT and the Constellation Map of the Track, side-by-side on the same Figure.

        :param threshold: Required threshold value for determining peaks.
        :type threshold: float
        :param distance: Required distance (between peaks) value for determining peaks.
        :type distance: int
        :return: Figure, Axes on which the STFT & Constellation Map were plotted.
        :rtype: (plt.Figure, plt.Axes)
        """
        fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
        fig, axs[0] = self.plot_stft(fig=fig, ax=axs[0])
        fig, axs[1] = self.plot_constellation_map(fig=fig, ax=axs[1], threshold=threshold, distance=distance)
        return fig, axs

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

    def plot_beat_times(self: class_name, ax: plt.Axes = None, fig: plt.Figure = None) \
            -> tuple[Figure, Axes]:
        """
        Plot the waveform (with beat indicators) of self to the given axis if available, else plot to new axis.

        :param fig: The figure on which to plot the waveform
        :type fig: plt.Figure
        :param ax: The axis of the figure on which to plot the waveform
        :type ax: plt.Axes
        :return: Figure, Axis on which the waveform was plotted
        :rtype: (plt.Figure, plt.Axes)
        """
        if ax is None:
            fig, ax = plt.subplots()
        if fig is None:
            fig = ax.get_figure()
        fig, ax = self.plot(ax=ax, fig=fig)
        beats = self.get_beat_times()
        [ax.axvline(x=_x, color='k', lw=0.5, linestyle='dashed') for _x in beats]  # NOTE: Plot v-lines @ beat x-values.
        return fig, ax


# Utilities
def plot_stft_spectrogram(data: np.ndarray, sr: int = DEF_SR,
                          ax: plt.Axes = None, fig: plt.Figure = None,
                          xlim: Optional[float] = None,
                          ylim: Optional[float] = None) \
        -> (plt.Figure, plt.Axes):
    """
    Plot an STFT spectrogram.

    :param data: STFT to plot.
    :type data: np.ndarray
    :param sr: Sample Rate of STFT to plot.
    :type sr: int
    :param ax: Axes on which to plot the STFT.
    :type ax: plt.Axes
    :param fig: Figure on which to plot the STFT.
    :type fig: plt.Figure
    :param xlim: Limits of the X-axis (Time) to plot.
    :type xlim: Optional[float]
    :param ylim: Limits of the Y-axis (Frequency) to plot.
    :type xlim: Optional[float]
    :return: Figure, Axes on which the STFT was plotted.
    :rtype: (plt.Figure, plt.Axes)
    """
    if ax is None:
        fig, ax = plt.subplots()
    if fig is None:
        fig = ax.get_figure()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    db = librosa.amplitude_to_db(np.abs(data), ref=np.max)  # NOTE: Generally speaking, only the magnitude is relevant.
    img = librosa.display.specshow(db, sr=sr, y_axis='log', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig, ax


def chirp_track(sr: int = DEF_SR, duration: float = DEF_DURATION) -> Track:
    """
    Creates a new Track instance with a synthetic chirp, going from C3 to C5.

    :return: Track
    :rtype: Track
    """
    track = Track(f"Chirp(sr={sr}, duration={duration})", load=False)  # Create an empty Track to record on.
    track.sr = sr
    track.y = librosa.chirp(fmin=librosa.note_to_hz('C3'),
                            fmax=librosa.note_to_hz('C5'),
                            sr=track.sr,
                            duration=duration)
    return track


def example_load() -> None:
    """Loads the 'choice' and 'nutcracker' example file from the librosa example library as Track instances."""
    track1 = Track(librosa.ex('choice', hq=True))
    track2 = Track(librosa.example('nutcracker'))
    track1.plot_beat_times()  # TODO: Seems to miss beats near the end of the file. Intentional?
    plt.show()
    track2.plot_stft()
    plt.show()


# Main Program
def main() -> Return:
    """
    The Main() function. Provides a user interface for the program.

    :return: Did the function complete successfully?
    :rtype: bool
    """
    # Variables
    print("Hello! Welcome to the ACR Project!")
    running = True
    tracks = []

    # Commands
    def init() -> Return:
        """Initialize a demo Track list."""
        nonlocal tracks
        tracks = [chirp_track(sr=sr) for sr in [44100, 48000]]  # Initialize Track list to Chirps of each SR in list.
        return Return.SUCCESS

    def terminate() -> Return:
        """Terminate the program."""
        nonlocal running
        stop()
        running = False
        return Return.SUCCESS

    def play() -> Return:
        """Play a Track by ID from user."""
        print("Available Tracks:")
        for i in range(len(tracks)):
            print(f"{i}: {tracks[i].title}")
        track_id = input("Please enter Track ID:")
        try:
            track_id = int(track_id)
            if track_id == "":
                print("Cancelling...")
                return Return.CANCEL
            track: Track = tracks[track_id]
            track.play(wait=True)
            return Return.SUCCESS
        except ValueError:
            print("Not a valid Track ID number. Please try again.")
            return Return.VALUE_ERROR
        except IndexError:
            print("Track ID out of range. Please try again.")
            return Return.INDEX_ERROR
        except sounddevice.PortAudioError:
            print("Error playing Track. Please try again.")
            return Return.AUDIO_ERROR
        pass

    def stop() -> Return:
        """Stop playing all currently playing audios on sound-device."""
        sd.stop()
        return Return.SUCCESS

    def record(wait: bool = True, save: bool = True, ext: str = "wav") -> Return:
        """Record from sound-device."""
        try:
            duration = float(
                input(f"Please enter recording Duration (seconds, Default={DEF_DURATION}):") or DEF_DURATION)
        except ValueError:
            print("Duration invalid. Please try again.")
            return Return.VALUE_ERROR
        try:
            sr = int(input(f"Please enter Sample Rate (Hz, Default={DEF_SR}):") or DEF_SR)
        except ValueError:
            print("Sample Rate invalid. Please try again.")
            return Return.VALUE_ERROR
        recording = sd.rec(int(duration * sr))
        if wait:
            sd.wait()  # Wait for recording to finish.
        track = Track(f'{DEF_REC_PATH}\\Recording - {datetime.now().strftime("%Y-%m-%d  %H-%M-%S")}', load=False)
        track.sr = sr
        track.y = recording
        if save:
            if not os.path.exists(DEF_REC_PATH):
                os.mkdir(DEF_REC_PATH)  # Create "Recordings" folder if one does not exist. (SoundFile can't mkdir)
            sf.write(f"{track.title}.{ext}", track.y, track.sr)  # Save recording to file.
        # TODO: Soundfile takes a while to save the recording - is it possible to block Python in the meantime?
        tracks.append(track)
        return Return.SUCCESS

    def generate() -> Return:
        """Generate a new Chirp with input from user."""
        nonlocal tracks
        try:
            sr = int(input("Please enter Chirp Sample Rate:"))
        except ValueError:
            print("Invalid Sample Rate. Please try again.")
            return Return.VALUE_ERROR
        try:
            duration = float(input("Please enter Chirp Duration:"))
        except ValueError:
            print("Invalid Duration. Please try again.")
            return Return.VALUE_ERROR
        tracks.append(chirp_track(sr=sr, duration=duration))
        return Return.SUCCESS

    # Command List
    cmds = dict()
    for cmd in [init, terminate, play, stop, record, generate]:  # TODO: Add all commands here!
        cmds[cmd.__name__] = cmd

    # Main Loop
    while running:
        print("\nAvailable Commands:")
        print(list(cmds.keys()))
        inp = input("Please enter a command:").lower()  # Case-insensitive.
        if inp not in cmds.keys():
            print("Command not found. Please try again.")
            continue
        cmds[inp]()
    print("Goodbye!")
    return Return.SUCCESS


if __name__ == "__main__":
    pass  # main()

# TODO:
#  COMPLETE: Change Sample Rate -> Higher SR = shorter higher pitch
#  COMPLETE: Experiment with cross-referencing librosa, soundfile, python-sounddevice -> All are based on NumPy.
#  COMPLETE: Basic Menu -> Currently good-faith programmer-only.
#  COMPLETE: Playback -> Not tested with relative/absolute path.
#  COMPLETE: Recording (python-sounddevice = NumPy / pyaudio = byte) -> Used python-sounddevice, as it's based on NumPy.
#  IN PROGRESS: Change STFT window size
#  IN PROGRESS: Initial database -> Ripped Home CDs, will rip friends' CDs in the future.
#  IN PROGRESS: Play with peak_find parameters to affect Constellation Map.
#  NOTE: show patch as window, i.e: 8x8, 3x3. Use np.max. Control target area / k parameters. (5 sec, 1024 bins)

_track = Track(librosa.ex('choice', hq=True))
_fig, _axs = _track.plot_stft_and_constellation(threshold=0.75, distance=15)
