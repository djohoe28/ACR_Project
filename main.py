#  Copyright (c) 2022. Code by DJohoe28.
import argparse
import os
from datetime import datetime
from enum import Enum
from typing import Optional, Union, Iterable

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice as sd
import soundfile as sf

# Initializations
OPTIONS = {
    "DatabasePath": "C:\\Users\\DJoho\\Downloads\\_Database ACR Shazam",
    "RecordingPath": "Recordings",  # Default Recordings Path
    "RecordingExtension": "wav",
    "RecordingDuration": 1,  # Default Duration (Chirps, Recordings, etc)
    "SampleRate": 44100,  # Default Sample Rate (see above)
    "Channels": 2,
    "NFFT": 2048,
    "HopLength": None,
    "WindowLength": None,
    "Threshold": 0.5,
    "Distance": 10,
    "TimeRadius": 20,
    "FreqRadius": 20,
    "FanOutFactor": 10,
    "TFRangeMin": 10,
    "TFRangeMax": 160,
    "TFRangeStep": 10
}
OPTIONS["TFRange"] = range(OPTIONS["TFRangeMin"],
                           OPTIONS["TFRangeMax"] + OPTIONS["TFRangeStep"],
                           OPTIONS["TFRangeStep"])

parser = argparse.ArgumentParser(
    prog="ACR Project",
    description="Automatic Content Recognition of songs in Hebrew.",
    epilog="By Jonathan Eddie Amir for Tel Hai College."
)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-p', '--path', default=OPTIONS["RecordingPath"])
parser.add_argument('-e', '--extension', default=OPTIONS["RecordingExtension"])
parser.add_argument('-sr', '--sample-rate', default=OPTIONS["SampleRate"], type=int)
parser.add_argument('-c', '--channels', default=OPTIONS["Channels"], type=int)
parser.add_argument('-n', '--num-fft', default=OPTIONS["NFFT"], type=int)
parser.add_argument('-l', '--hop-length', default=OPTIONS["HopLength"], type=int)
parser.add_argument('-w', '--window-length', default=OPTIONS["WindowLength"], type=int)
parser.add_argument('-th', '--threshold', default=OPTIONS["Threshold"], type=float)
parser.add_argument('-d', '--distance', default=OPTIONS["Distance"], type=int)
parser.add_argument('-tr', '--time-radius', default=OPTIONS["TimeRadius"], type=int)
parser.add_argument('-fr', '--freq-radius', default=OPTIONS["FreqRadius"], type=int)
parser.add_argument('-fo', '--fan-out', default=OPTIONS["FanOutFactor"], type=int)
args = parser.parse_args()

sd.default.samplerate = OPTIONS["SampleRate"]  # Currently unused, as we know each file's sample rate
sd.default.channels = OPTIONS["Channels"]  # Set sound-device's default to Stereo


def tuple_to_array(arr: Union[Iterable, tuple]) -> np.array:
    return np.column_stack((arr[0], arr[1]))


def nonzero_as_array(arr: np.array) -> np.array:
    return tuple_to_array(arr.nonzero())


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
    :type y: np.array
    :ivar sr: The Sample Rate of the track - the timespan used by y
    :type sr: int
    """
    class_name: str = "Track"

    def __init__(self: class_name, path: str = "", load: bool = True) -> None:
        """
        Track Constructor.

        :param path: Path to file
        :type path: str
        :param load: If true, loads file - else, leaves values uninitialized.
        :type load: bool
        """
        self.path: str = path
        """Path to Track's self."""
        self.filename: str = os.path.basename(path)
        """File name of self."""
        self.title: str = os.path.basename(os.path.splitext(path)[0])
        """Title of self."""
        self.y: np.array = np.zeros(1)
        """Audio time series of self (y values over samples)."""
        self.sr: int = 0
        """Sample rate of self."""
        # NOTE: Lazy Evaluation = If property is accessed for the first time: evaluate it; Else: recall last evaluation.
        # Lazy Evaluation can be used when a property is too expensive to evaluate "by default" when unnecessary.
        self._stft: Optional[np.array] = None
        """Private Lazy Evaluation of Short-Time Fourier Transform of self."""
        self._cmap: Optional[np.array] = None
        """Private Lazy Evaluation of Constellation Map of self"""
        self._anchors: Optional[np.array] = None
        """Private Lazy Evaluation of Anchors of self."""
        self._hashes: Optional[dict[str, dict[str, int]]] = None
        """Private Lazy Evaluation of Hashes of self. { Hash -> {title -> offset} }"""

        # Load file (if load=true).
        if load:
            self.y, self.sr = librosa.load(self.path)

        # Load Metadata (if available)
        if sf.check_format(self.path):
            _metadata = sf.info(self.path)
            if hasattr(_metadata, "title"):
                self.title = _metadata.title

    def __str__(self: class_name):
        return f"Track: {self.title}"

    def __repr__(self: class_name):
        return f"Track: {self.path}"

    def play(self: class_name, wait: bool = True) -> None:
        """
        Play self.

        :param wait: If true, the program halts until self stops playing once
        :type wait: bool
        :return: None
        :rtype: NoneType
        """
        sd.play(self.y, self.sr)
        if wait:
            sd.wait()

    @property
    def stft(self) -> np.array:
        """Short-Time Fourier Transform of self."""
        if self._stft is None:
            self._stft = self._evaluate_stft()
        return self._stft

    @property
    def constellation_map(self) -> np.array:
        """Constellation Map of self."""
        if self._cmap is None:
            self._cmap = self._evaluate_constellation_map()
        return self._cmap

    @property
    def anchors(self) -> np.array:
        """Anchors of self."""
        if self._anchors is None:
            self._anchors = self._evaluate_anchors()
        return self._anchors

    @property
    def hashes(self) -> dict[str, dict[str, int]]:
        """Hashes of self. { Hash -> { Title -> Offset } }"""
        if self._hashes is None:
            self._hashes = self._evaluate_hashes()
        return self._hashes

    def _evaluate_stft(self: class_name, n_fft=OPTIONS["NFFT"], hop_length=OPTIONS["HopLength"],
                       win_length=OPTIONS["WindowLength"]) -> np.array:
        """
        Get the Short-Time Fourier Transform (STFT) of self.

        :return: STFT of self.
        :rtype: np.array
        """
        win_length = win_length if win_length is not None else n_fft
        hop_length = hop_length if hop_length is not None else win_length // 4
        _stft = librosa.stft(self.y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        return _stft
        # TODO: To add labels to the axes, use the following line;
        # np.rec.array(np.ascontiguousarray(_stft),
        #              dtype=[('frequency', np.dtype('complex64')), ('time', np.dtype('complex64'))])

    def _evaluate_constellation_map(self: class_name,
                                    time_radius=OPTIONS["TimeRadius"],
                                    freq_radius=OPTIONS["FreqRadius"]) -> np.array:
        """
        Get a copy of self's STFT where all non-peaks are set to 0.

        :param time_radius: Width of Target Zone
        :type time_radius: int
        :param freq_radius: Height of Target Zone
        :type freq_radius: int
        :return: Constellation Map of self
        :rtype: np.array
        """
        # TODO: THIS, STUPID
        _stft = np.abs(self.stft)  # Only magnitude is relevant for peak detection.
        _maxima_idx = np.argmax(_stft, axis=0)  # Find maxima along axis 0 - Frequency. ("Time Maxima" is irrelevant.)
        _maxima_val = _stft[_maxima_idx, :]  # Get the values of the maxima indices found above.
        return tuple_to_array((_maxima_idx, _maxima_val))  # Reshapes the (idx[], val[]) tuple into a (idx, val)[].

    def _evaluate_anchors(self: class_name,
                          time_radius=OPTIONS["TimeRadius"],
                          freq_radius=OPTIONS["FreqRadius"]) -> np.array:
        """
        Get a list of anchors (points) of self's Constellation Map.

        :param time_radius: Width of Target Zone
        :type time_radius: int
        :param freq_radius: Height of Target Zone
        :type freq_radius: int
        :return: Anchors of self
        :rtype: np.array
        """
        return nonzero_as_array(self.constellation_map)

    def _evaluate_hashes(self: class_name,
                         time_radius=OPTIONS["TimeRadius"],
                         freq_radius=OPTIONS["FreqRadius"]) -> dict[str, dict[str, int]]:
        """
        Get a Hash Set dictionary of self's Constellation Map hashes.

        :param time_radius: Width of Target Zone
        :type time_radius: int
        :param freq_radius: Height of Target Zone
        :type freq_radius: int
        :return: Hash Set of self's hashes { Hash -> {title -> offset} }
        :rtype: dict[str, dict[str, int]]
        """
        _hashes = {}
        _anchors = self.anchors
        for _anchor in _anchors:
            _target_zone: np.array = self.constellation_map[
                                     (_anchor[0] - time_radius):(_anchor[0] + time_radius),
                                     (_anchor[1] - freq_radius):(_anchor[1] + freq_radius)]
            _targets = nonzero_as_array(_target_zone)
            for _target in _targets:
                _target_hash = f"[{_anchor[0]}:{_target[0]}:{_anchor[1] - _target[1]}]"
                _hashes[_target_hash] = {self.title: _anchor[1]}
        return _hashes

    def plot_waveform(self: class_name, ax: plt.Axes = None, fig: plt.Figure = None) -> (plt.Figure, plt.Axes):
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

    def plot_spectrogram(self: class_name,
                         ax: plt.Axes = None, fig: plt.Figure = None,
                         xlim: Optional[float] = None,
                         ylim: Optional[float] = None) -> (plt.Figure, plt.Axes):
        """
        Plot a spectrogram of STFT of self.

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
        # Generate new subplots if not defined.
        if (fig is None) and (ax is not None):
            fig = ax.figure
        elif (fig is not None) and (ax is None):
            ax = fig.axes[0]
        elif (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        _db = librosa.amplitude_to_db(np.abs(self.stft), ref=np.max)  # NOTE: Mostly magnitude is relevant for peaks.
        _img = librosa.display.specshow(_db, sr=self.sr, y_axis='log', x_axis='time', ax=ax)
        fig.colorbar(_img, ax=ax, format="%+2.0f dB")
        return fig, ax

    def plot_constellation_map(self: class_name,
                               ax: plt.Axes = None, fig: plt.Figure = None,
                               xlim: Optional[float] = None,
                               ylim: Optional[float] = None) -> (plt.Figure, plt.Axes):
        """
        Plot a spectrogram of Constellation Map of self.

        :param ax: Axes on which to plot the Constellation Map.
        :type ax: plt.Axes
        :param fig: Figure on which to plot the Constellation Map.
        :type fig: plt.Figure
        :param xlim: Limits of the X-axis (Time) to plot.
        :type xlim: Optional[float]
        :param ylim: Limits of the Y-axis (Frequency) to plot.
        :type xlim: Optional[float]
        :return: Figure, Axes on which the Constellation Map was plotted.
        :rtype: (plt.Figure, plt.Axes)
        """
        # Generate new subplots if not defined.
        if (fig is None) and (ax is not None):
            fig = ax.figure
        elif (fig is not None) and (ax is None):
            ax = fig.axes[0]
        elif (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        _db = librosa.amplitude_to_db(np.abs(self.constellation_map), ref=np.max)  # NOTE: Phase irrelevant for peaks.
        _img = librosa.display.specshow(_db, sr=self.sr, y_axis='log', x_axis='time', ax=ax)
        fig.colorbar(_img, ax=ax, format="%+2.0f dB")
        return fig, ax

    def plot_stft_and_constellation(self: class_name,
                                    axs: plt.Axes = None, fig: plt.Figure = None,
                                    xlim: Optional[float] = None,
                                    ylim: Optional[float] = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the STFT and the Constellation Map of self, side-by-side on the same Figure.

        :param axs: Axes on which to plot the STFT & Constellation Map.
        :type axs: plt.Axes
        :param fig: Figure on which to plot the STFT & Constellation Map.
        :type fig: plt.Figure
        :param xlim: Limits of the X-axis (Time) to plot.
        :type xlim: Optional[float]
        :param ylim: Limits of the Y-axis (Frequency) to plot.
        :type xlim: Optional[float]
        :return: Figure, Axes on which the STFT  & Constellation Map were plotted.
        :rtype: (plt.Figure, plt.Axes)
        """
        if (fig is None) and (axs is not None):
            fig = axs.figure
        elif (fig is not None) and (axs is None):
            axs = fig.axes
        elif (fig is None) and (axs is None):
            fig, axs = plt.subplots(1, 2, sharex="all", sharey="all")
        if xlim is not None:
            axs.set_xlim(xlim)
        if ylim is not None:
            axs.set_ylim(ylim)
        fig, axs[0] = self.plot_spectrogram(fig=fig, ax=axs[0])
        fig, axs[1] = self.plot_constellation_map(fig=fig, ax=axs[1])
        return fig, axs


class TrackList(dict):
    class_name: str = "TrackList"
    Hashes: dict = {}  # TODO: Change to Hash Dictionary

    def __setitem__(self: class_name, key, value: Track):
        super().__setitem__(key, value)
        if value is Track:
            _hashes = value.hashes
            for key in _hashes:
                if key not in self.Hashes:
                    self.Hashes[key] = {}
                for _title in _hashes[key]:
                    self.Hashes[key][_title] = _hashes[key][_title]
            # self.Hashes += value.get_hashes()  # TODO: Check that this works

    def append(self: class_name, value: Track):
        self.__setitem__(value.title, value)


# Utilities
def chirp_track(sr: int = OPTIONS["SampleRate"], duration: float = OPTIONS["RecordingDuration"]) -> Track:
    """
    Creates a new Track instance with a synthetic chirp, going from C3 to C5.

    :return: Track
    :rtype: Track
    """
    _track = Track(f"Chirp(sr={sr}, duration={duration})", load=False)  # Create an empty Track to record on.
    _track.sr = sr
    _track.y = librosa.chirp(fmin=librosa.note_to_hz('C3'),
                             fmax=librosa.note_to_hz('C5'),
                             sr=_track.sr,
                             duration=duration)
    return _track


def example_load() -> (plt.Figure, plt.Axes):
    """Loads the 'choice' and 'nutcracker' example file from the librosa example library as Track instances."""
    _track1 = Track(librosa.ex('choice', hq=True))
    _track2 = Track(librosa.example('nutcracker'))
    _fig, _axs = plt.subplots(2, 2)
    _track1.plot_stft_and_constellation(axs=_axs[0], fig=_fig)
    _track2.plot_stft_and_constellation(axs=_axs[1], fig=_fig)
    return _fig, _axs


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
    database = TrackList()

    # Commands
    def init() -> Return:
        """Initialize a demo Track list."""
        nonlocal database
        database = [chirp_track(sr=sr) for sr in [44100, 48000]]  # Initialize Track list to Chirps of each SR in list.
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
        for i in range(len(database)):
            print(f"{i}: {database[i].title}")
        track_id = input("Please enter Track ID:")
        try:
            track_id = int(track_id)
            if track_id == "":
                print("Cancelling...")
                return Return.CANCEL
            track: Track = database[track_id]
            track.play(wait=True)
            return Return.SUCCESS
        except ValueError:
            print("Not a valid Track ID number. Please try again.")
            return Return.VALUE_ERROR
        except IndexError:
            print("Track ID out of range. Please try again.")
            return Return.INDEX_ERROR
        except sd.PortAudioError:
            print("Error playing Track. Please try again.")
            return Return.AUDIO_ERROR
        pass

    def stop() -> Return:
        """Stop playing all currently playing audios on sound-device."""
        sd.stop()
        return Return.SUCCESS

    def record(wait: bool = True, save: bool = True, add: bool = False, ext: str = OPTIONS["FileType"]) -> Return:
        """Record from sound-device."""
        try:
            duration = float(
                input(f"Please enter recording Duration (seconds, Default={OPTIONS['DefaultDuration']}):") or OPTIONS[
                    "RecordingDuration"])
        except ValueError:
            print("Duration invalid. Please try again.")
            return Return.VALUE_ERROR
        try:
            sr = int(input(f"Please enter Sample Rate (Hz, Default={OPTIONS['DefaultSampleRate']}):") or OPTIONS[
                "SampleRate"])
        except ValueError:
            print("Sample Rate invalid. Please try again.")
            return Return.VALUE_ERROR
        recording = sd.rec(int(duration * sr))
        if wait:
            sd.wait()  # Wait for recording to finish.
        track = Track(f'{OPTIONS["RecordingPath"]}\\Recording - {datetime.now().strftime("%Y-%m-%d  %H-%M-%S")}',
                      load=False)
        track.sr = sr
        track.y = recording
        if save:
            if not os.path.exists(OPTIONS["RecordingPath"]):
                # Create "Recordings" folder if one does not exist. (SoundFile can't mkdir)
                os.mkdir(OPTIONS["RecordingPath"])
            sf.write(f"{track.title}.{ext}", track.y, track.sr)  # Save recording to file.
        # TODO: Soundfile takes a while to save the recording - is it possible to block Python in the meantime?
        if add:
            database.append(track)  # TODO: Make sure this works.
        return Return.SUCCESS

    def generate() -> Return:
        """Generate a new Chirp with input from user."""
        nonlocal database
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
        database.append(chirp_track(sr=sr, duration=duration))
        return Return.SUCCESS

    def append() -> Return:
        # TODO: Do this.
        return Return.SUCCESS

    # Command List
    cmds = dict()
    for cmd in [init, terminate, play, stop, record, generate, append]:  # TODO: Add all commands here! SEARCH
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


class FP_Results:
    def __init__(self, track: Track,
                 t_range: range = OPTIONS["TFRange"],
                 f_range: range = OPTIONS["TFRange"]):
        self.t_range: range = t_range
        self.f_range: range = f_range
        # TODO: np.meshgrid
        self.title: str = track.title
        self.stft: np.array = track.stft
        self.results: np.array = self.load()
        self._max_val: int = self._get_max_val()

    def _get_max_val(self) -> int:
        _max_val = 0
        for _time_idx in range(len(self.t_range)):
            _time_val = self.t_range[_time_idx]
            for _freq_idx in range(len(self.f_range)):
                _freq_val = self.f_range[_freq_idx]
                _results_val = self.results[_time_idx, _freq_idx]
                _val = _time_val * _freq_val  # * results_val
                print(f"{_time_val} * {_freq_val} * {_results_val} = {_val} "
                      f"{'<' if (_val < _max_val) else ('>' if (_val > _max_val) else '=')} "
                      f"{_max_val}")
                if _val > _max_val and 1000 < _results_val < 2000:
                    _max_val = _val
        np.save(f"{self.title}.npy", self.results)
        return _max_val

    def load(self) -> np.array:
        _results = np.zeros((len(self.t_range), len(self.f_range)), dtype=int)
        try:
            _load = np.load(f"{self.title}.npy")
            if _load.shape != _results.shape:
                raise FileNotFoundError
            _results = _load
        except FileNotFoundError:
            for _time_idx in range(len(self.t_range)):
                _time_radius = self.t_range[_time_idx]
                for _freq_idx in range(len(self.f_range)):
                    _freq_radius = self.f_range[_freq_idx]
                    _shape = (_time_radius, _freq_radius)
                    _stft = np.abs(self.stft)
                    _cmap = scipy.ndimage.maximum_filter(_stft, footprint=np.ones(_shape), mode='constant') == _stft
                    _count = nonzero_as_array(_cmap).shape[0]
                    _results[_time_idx, _freq_idx] = _count
                    print(f"({_time_radius}, {_freq_radius}) = {_count}")
        return _results

    def plot(self) -> (plt.Figure, plt.Axes):
        _max_val: int = self._max_val
        _fig: plt.Figure
        _ax: plt.Axes
        _fig, _ax = plt.subplots()
        _ax.imshow(self.results, cmap='viridis')
        for _time_idx in range(len(self.t_range)):
            for _freq_idx in range(len(self.f_range)):
                val = self.results[_time_idx, _freq_idx]
                color = 'white' if (not (1000 < val < 10000)) else (
                    'green' if (self.t_range[_time_idx] * self.f_range[_freq_idx] == _max_val) else 'yellow')
                _fig.text(_freq_idx, _time_idx, f"{val}", ha='center', va='center', color=color, fontsize=8)
        _ax.set_xticks(range(self.results.shape[1]), self.t_range)
        _ax.set_yticks(range(self.results.shape[0]), self.f_range)
        _ax.set_xlabel("Time")
        _ax.set_ylabel("Frequency")
        _ax.set_title("# Anchors by (Time, Frequency) Footprint")
        _ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        _fig.savefig(f"{self.title}.png")
        return _fig, _ax


if __name__ == "__main__":
    pass  # main()


# TODO:
#  COMPLETE: Change Sample Rate -> Higher SR = shorter higher pitch
#  COMPLETE: Experiment with cross-referencing librosa, soundfile, python-sounddevice -> All are based on NumPy.
#  COMPLETE: Basic Menu -> Currently good-faith programmer-only.
#  COMPLETE: Playback -> Not tested with relative/absolute path.
#  COMPLETE: Recording (python-sounddevice = NumPy / pyaudio = byte) -> Used python-sounddevice, as it's based on NumPy.
#  COMPLETE: Change STFT window size
#  COMPLETE: Initial database -> Ripped Home CDs, will rip friends' CDs in the future.
#  COMPLETE: Options struct
#  IN PROGRESS: Play with peak_find parameters to affect Constellation Map.
#  NOTE: show patch as window, i.e: 8x8, 3x3. Use np.max. Control target area / k parameters. (5 sec, 1024 bins)
#  NOTE: Window - 140ms ~ 500ms, ~10 in range, histogram points in window, threshold,
#  NOTE: Mode5, MDA256, CRC checksum
#  NOTE: matlab find, SIMD, parameter sweep, scipy research


def tracklist_from_directory(path: str = OPTIONS["DatabasePath"],
                             ext: (Union[str, tuple[str, ...]]) = OPTIONS["RecordingExtension"]):
    _tracklist = TrackList()
    for _filename in os.listdir(path):
        if _filename.endswith(ext):
            print(f"#{len(_tracklist) + 1} = {_filename}...")
            _track = Track(os.path.join(path, _filename))
            _tracklist.append(_track)
    _tracklist.append(Track(librosa.ex('choice')))
    return _tracklist


def find_match(track: Track, tracklist: TrackList):
    # TODO: THIS, STUPID
    _hashes = t.hashes


hashes = {}
results = {}
p = "C:\\Users\\DJoho\\Downloads\\_Database ACR Shazam\\אביגייל רוז - הפרעות - לא טוב לי.wav"
t = Track(p)
example_load()
"""
# Constellation Map - ver. Maximum Filter
time_radius=OPTIONS["TimeRadius"]
freq_radius=OPTIONS["FreqRadius"]
return np.array(scipy.ndimage.maximum_filter(stft, size=(time_radius, freq_radius), mode='constant') == stft)
# Constellation Map - ver. Find Peaks
height = np.mean(stft) + 2 * np.std(stft)  # A dynamically-calculated threshold to consider a peak.
peaks, properties = scipy.signal.find_peaks(stft, height=height)
"""
"""
tl = tracklist_from_directory()
for key in tl:
    results[key] = FP_Results(tl[key])
    fig, ax = results[key].plot()
    plt.close(fig=fig)
"""
print("Done!")
