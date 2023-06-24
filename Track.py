from Utilities import *

from typing import Optional

import os
import copy
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

if not OPTIONS["COLAB"]:
    import sounddevice as sd

from scipy.ndimage import maximum_filter, binary_erosion


class Track(object):
    """
    A Track loaded into the program.
    # TODO: >type lines in documentation cause conflict with @property decorator

    :cvar class_name: The name of the class
    :type class_name: AnyStr
    :ivar path: The path (relative/absolute) to the file represented by self
    :type path: AnyStr
    :ivar filename: Filename of self (name + extension, no path)
    :type filename: AnyStr
    :ivar title: The title of the track  # NOTE: Currently the file name, could be the track title
    :type title: AnyStr
    :ivar is_compressed: Disables lazy evaluation of non-cached sizable elements.
    :type is_compressed: Boolean
    :ivar is_verbose: Prints additional data.
    :type is_verbose: Boolean
    :ivar _y: Audio time series of self (y values over samples). (Masked, lazily evaluated)
    :type _y: Optional[np.ndarray]
    :ivar _sr: Sample Rate of self. (Masked by .sr)
    :type _sr: Integer
    :ivar _duration: Duration of self (floating point, in seconds) (Masked, lazily evaluated)
    :type _duration: Float
    :ivar _shape: Shape of maps of self
    >type _shape: Optional[Tuple]
    :ivar _stft: Short-Time Fourier Transform representation of self. (Masked, lazily evaluated)
    >type _stft: Optional[np.ndarray]
    :ivar _constellation: Constellation of self. (Masked, lazily evaluated)
    >type _constellation: Optional[np.ndarray]
    :ivar _asterism: Asterism list of self. (Masked, lazily evaluated)
    >type _asterism: Optional[np.ndarray]
    :ivar _hashes: HashSet of self. (Masked, lazily evaluated)
    >type _hashes: Optional[np.ndarray]
    """
    class_name: AnyStr = "Track"

    def __init__(self: class_name, path: AnyStr = "") -> None:
        """
        Track Constructor.

        :param path: Path to file
        :type path: AnyStr
        """
        self.path: AnyStr = path
        """Path to Track's self."""
        self.filename: AnyStr = os.path.basename(path)
        """File name of self."""
        self.title: AnyStr = os.path.basename(os.path.splitext(path)[0])
        """Title of self."""
        self.is_compressed: Boolean = False
        self.is_verbose: Boolean = OPTIONS["DEBUG"]
        # NOTE: Lazy Evaluation = If property is accessed for the first time: evaluate it; Else: recall last evaluation.
        # Lazy Evaluation can be used when a property is too expensive to evaluate "by default" when unnecessary.
        self._y: Optional[np.ndarray] = None
        """Audio time series of self (y values over samples)."""
        self._sr: Optional[int] = OPTIONS["SampleRate"]
        """Sample rate of self."""
        self._duration: Optional[float] = None  # len(self.y) / float(self.sr)
        """Duration of self (floating point, in seconds.)"""
        self._shape: Optional[Tuple] = None
        """Shape of self's STFT array (evaluated later). Used in debugging."""
        self._stft: Optional[np.ndarray] = None
        """Private Lazy Evaluation of Short-Time Fourier Transform of self."""
        self._constellation: Optional[np.ndarray] = None
        """Private Lazy Evaluation of Constellation of self"""
        self._asterism: Optional[np.ndarray] = None
        """Private Lazy Evaluation of Asterism of self."""
        self._hashes: Optional[Hashes] = None
        """Private Lazy Evaluation of Hashes of self. { Hash -> {title -> offset} }"""
        # Load Metadata (if available)
        if sf.check_format(self.path):
            _metadata = sf.info(self.path)
            if hasattr(_metadata, "title"):
                self.title = _metadata.title
            if hasattr(_metadata, "duration"):  # Guaranteed by SoundFileInfo class.
                self._duration = _metadata.duration  # NOTE: Overwrites (len(y)/sr), just in case.

    def __str__(self: class_name):
        return f"Track: {self.title}"

    def __repr__(self: class_name):
        return f"Track: {self.path}"

    def play(self: class_name, wait: Boolean = OPTIONS["Synchronous"]) -> None:
        """Play self using sounddevice."""
        if OPTIONS["COLAB"]:
            return
        sd.play(self.y, self.sr)
        if wait:
            sd.wait()

    @property
    def y(self) -> Optional[np.ndarray]:
        """Audio time series of self (y values over samples)."""
        if self.is_verbose or True:  # TODO: PyCharm tends to crash here.
            print("y", self._y is None, self.title)
        if self._y is None and not self.is_compressed:
            self._y, self._sr = librosa.load(self.path, sr=OPTIONS["SampleRate"])
        return self._y

    @property
    def sr(self) -> int:
        """Sample Rate of self."""
        if self.is_verbose:
            print("SR", self.title)
        return self._sr

    @property
    def duration(self) -> float:
        """Duration of self (floating point, in seconds.)"""
        if self.is_verbose:
            print("Duration", self._duration is None, self.title)
        if self._duration is None and not self.is_compressed:
            self._duration = len(self.y) / float(self.sr)
        return self._duration

    @property
    def stft(self) -> np.ndarray:
        """Short-Time Fourier Transform of self."""
        if self.is_verbose:
            print("STFT", self._stft is None, self.title)
        if self._stft is None and not self.is_compressed:
            self._stft = self._evaluate_stft()
        return self._stft

    @property
    def shape(self) -> tuple:
        """Shape of self's STFT array (evaluated later). Used in debugging."""
        if self.is_verbose:
            print("Shape", self._shape is None, self.title)
        if self._shape is None and not self.is_compressed:
            win_length = OPTIONS["WindowLength"] if OPTIONS["WindowLength"] is not None else OPTIONS["NFFT"]
            hop_length = OPTIONS["HopLength"] if OPTIONS["HopLength"] is not None else win_length // 4
            self._shape = (win_length // 2 + 1, (len(self.y) - win_length) // hop_length + 1)
        return self._shape

    @property
    def constellation(self) -> np.ndarray:
        """Constellation of self."""
        if self.is_verbose:
            print("Constellation", self._constellation is None, self.title)
        if self._constellation is None and not self.is_compressed:
            self._constellation = self._evaluate_constellation()
        return self._constellation

    @property
    def asterism(self) -> np.ndarray:
        """Asterism of self."""
        if self.is_verbose:
            print("Asterism", self._asterism is None, self.title)
        if self._asterism is None and not self.is_compressed:
            self._asterism = self._evaluate_asterism()
        return self._asterism

    @property
    def hashes(self) -> Hashes:
        """Hashes of self. { Hash -> { Title -> Offset } }"""
        if self.is_verbose:
            print("Hashes", self._hashes is None, self.title)
        if self._hashes is None and not self.is_compressed:
            self._hashes = self._evaluate_hashes()
        return self._hashes

    def _evaluate_stft(self: class_name, n_fft=OPTIONS["NFFT"], hop_length=OPTIONS["HopLength"],
                       win_length=OPTIONS["WindowLength"]) -> np.ndarray:
        """
        Get the Short-Time Fourier Transform (STFT) of self.

        :return: STFT of self.
        :rtype: np.ndarray
        """
        if self.is_verbose:
            print("_evaluate_stft", self.title)
        win_length = win_length if win_length is not None else n_fft
        hop_length = hop_length if hop_length is not None else win_length // 4
        _stft = librosa.stft(self.y, sr=self.sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        return _stft

    def _evaluate_constellation(self: class_name,
                                maxima_width=OPTIONS["MaximaFilterWidth"],
                                maxima_height=OPTIONS["MaximaFilterHeight"]) -> np.ndarray:
        """
        Get a copy of self's STFT where all non-peaks are set to 0.
        Adapted from https://stackoverflow.com/a/3689710

        :param maxima_width: Width of Target Zone
        :type maxima_width: int
        :param maxima_height: Height of Target Zone
        :type maxima_height: int
        :return: Constellation of self
        :rtype: np.ndarray
        """
        if self.is_verbose:
            print("_evaluate_constellation()", self.title)
        _stft = np.abs(self.stft)
        _size = (maxima_width, maxima_height)
        _structure = np.ones(_size)
        # Apply a boolean filter to each point => (new_point = True if point is maxima else False).
        _maxima_mask = maximum_filter(_stft, size=_size, mode='constant') == _stft
        # Create an "image mask" of the background
        _background = (_stft == 0)
        # Erode background to subtract from _maxima_mask in order to ignore the border points (technically extrema)
        _eroded = binary_erosion(_background, structure=_structure, border_value=1)
        # Apply XOR to remove eroded background (i.e: border points) from result
        return _maxima_mask ^ _eroded

    def _evaluate_asterism(self: class_name) -> np.ndarray:
        """
        Get a list of asterism (local maxima) of self's Constellation.

        :return: Asterism of self
        :rtype: np.ndarray
        """
        if self.is_verbose:
            print("_evaluate_asterism()", self.title)
        return nonzero_to_array(self.constellation)

    def _evaluate_hashes(self: class_name,
                         target_width: Integer = OPTIONS["TargetZoneWidth"],
                         target_height: Integer = OPTIONS["TargetZoneHeight"],
                         time_offset: Integer = OPTIONS["TargetZoneTimeOffset"]) -> Hashes:
        """
        Get a Hash Set dictionary of self's Constellation hashes.

        :param target_width: Width of Target Zone
        :type target_width: Integer
        :param target_height: Height of Target Zone
        :type target_height: Integer
        :param time_offset: Time (X-axis) Offset of Target Zone
        :type time_offset: Integer
        :return: Hash Set of self's hashes { Hash -> {title -> offset} }
        :rtype: Dict[AnyStr, Dict[AnyStr, Integer]]
        """
        if self.is_verbose:
            print("_evaluate_hashes()", self.title)
        _hashes = {}
        _asterism = self.asterism
        _constellation = self.constellation
        for _anchor in _asterism:
            # Find Boundaries of Target Zone
            _t_min = _anchor[0] + time_offset  # Left
            _t_max = _t_min + target_width  # Right
            _f_min = _anchor[1] - target_height // 2  # Top
            _f_max = _f_min + target_height // 2  # Bottom
            # Clip appropriately (do not leave confines of array, nor go backwards in time.)
            _t_min, _t_max = np.clip(a=(_t_min, _t_max), a_min=_anchor[0] + 1, a_max=self.shape[0] - 1)
            _f_min, _f_max = np.clip(a=(_f_min, _f_max), a_min=0, a_max=self.shape[1] - 1)
            # Create Target Zone from Constellation; Where maxima in Boundaries: True, Else: False
            _target_zone_bounds = (slice(_t_min, _t_max), slice(_f_min, _f_max))
            _target_zone = np.zeros_like(_constellation)
            _target_zone[_target_zone_bounds] = _constellation[_target_zone_bounds]
            _targets = nonzero_to_array(_target_zone)
            # Create Hashes from anchor-target pairs
            for _target in _targets:
                _target_hash = f"[{_anchor[1]}:{_target[1]}:{_anchor[0] - _target[0]}]"
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
        if self.is_verbose:
            print("plot_waveform()", self.title)
        if ax is None:
            fig, ax = plt.subplots()  # {fig} is only used as tuple filler.
        if fig is None:
            fig = ax.get_figure()
        librosa.display.waveshow(y=self.y, sr=self.sr, ax=ax)
        ax.set(title=f"Waveform: {self.title}")
        ax.label_outer()
        return fig, ax

    def plot_stft(self: class_name,
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
        if self.is_verbose:
            print("plot_spectrogram()", self.title)
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

    def plot_constellation(self: class_name,
                           ax: plt.Axes = None, fig: plt.Figure = None,
                           xlim: Optional[float] = None,
                           ylim: Optional[float] = None) -> (plt.Figure, plt.Axes):
        """
        Plot a spectrogram of Constellation of self.

        :param ax: Axes on which to plot the Constellation.
        :type ax: plt.Axes
        :param fig: Figure on which to plot the Constellation.
        :type fig: plt.Figure
        :param xlim: Limits of the X-axis (Time) to plot.
        :type xlim: Optional[float]
        :param ylim: Limits of the Y-axis (Frequency) to plot.
        :type xlim: Optional[float]
        :return: Figure, Axes on which the Constellation was plotted.
        :rtype: (plt.Figure, plt.Axes)
        """
        if self.is_verbose:
            print("plot_constellation()", self.title)
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
        x, y = self.constellation.nonzero()
        ax.scatter(x, y)
        return fig, ax

    def plot_stft_and_constellation(self: class_name,
                                    axs: plt.Axes = None, fig: plt.Figure = None,
                                    xlim: Optional[float] = None,
                                    ylim: Optional[float] = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the STFT and the Constellation of self, side-by-side on the same Figure.

        :param axs: Axes on which to plot the STFT & Constellation.
        :type axs: plt.Axes
        :param fig: Figure on which to plot the STFT & Constellation.
        :type fig: plt.Figure
        :param xlim: Limits of the X-axis (Time) to plot.
        :type xlim: Optional[float]
        :param ylim: Limits of the Y-axis (Frequency) to plot.
        :type xlim: Optional[float]
        :return: Figure, Axes on which the STFT  & Constellation were plotted.
        :rtype: (plt.Figure, plt.Axes)
        """
        if self.is_verbose:
            print("plot_stft_constellation", self.title)
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
        fig, axs[0] = self.plot_stft(fig=fig, ax=axs[0])
        fig, axs[1] = self.plot_constellation(fig=fig, ax=axs[1])
        return fig, axs

    def get_compressed(self: class_name) -> class_name:
        """
        :return: Compressed version of self, eliminating sizable elements.
        :rtype: np.ndarray
        """
        if self.is_verbose:
            print("Compressed", self.title)
        _temp = self.is_compressed
        self.is_compressed = True  # Prevent unnecessary computations
        _copy = copy.deepcopy(self)
        self.is_compressed = _temp  # Prevent unnecessary computations
        _copy.is_compressed = True  # Prevent unnecessary computations
        _copy._duration = self.duration  # Run the duration evaluation (if somehow didn't trigger.)
        _copy._shape = self.shape  # Run the shape evaluation.
        _copy._y = None
        # _copy._sr = None
        _copy._stft = None
        _copy._constellation = None
        _copy._asterism = None
        _copy._hashes = None
        return _copy

    def decompress(self: class_name) -> None:
        """Enables lazy evaluation of sizable elements.."""
        # TODO: Anything else?
        if self.is_verbose:
            print("Decompress", self.is_compressed, self.title)
        self.is_compressed = False

    def cache_property(self: class_name, prop: Properties, plot_figures: Boolean = False,
                       verbose: Boolean = True) -> None:
        if verbose:
            print(f"Caching {prop.name} of {self.title}")
        if prop == Properties.Pickle:
            save_as_pickle(os.path.join(OPTIONS["CachePath"], prop.name, f'{self.title}.pkl'), self)
            return
        _data: Any = getattr(self, prop.value)
        if isinstance(_data, np.ndarray):
            np.save(os.path.join(OPTIONS["CachePath"], prop.name, f'{self.title}.npy'), _data)
            if plot_figures:
                _fig: Optional[plt.Figure] = None
                _axs: Optional[plt.Axes] = None
                _fig_path: AnyStr = os.path.join(OPTIONS["FigurePath"], prop.name, f'{self.title}.png')
                if prop == Properties.Y:
                    _fig, _axs = self.plot_waveform()
                elif prop == Properties.STFT:
                    _fig, _axs = self.plot_stft()
                elif prop == Properties.Constellation:
                    _fig, _axs = self.plot_constellation()
                if _fig is not None:
                    _fig.savefig(_fig_path)
                    plt.close()


def save_as_pickle(path: AnyStr, data) -> None:
    """
    Save an object using the pickle library. Compresses Track objects.

    :param path: Filepath to save to data
    :param data: Data to save to filepath
    """
    with open(path, 'wb') as fp:
        if isinstance(data, Track):
            pickle.dump(data.get_compressed(), fp, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(path: AnyStr, decompress: Boolean = False) -> Any:
    """
    Load object from pickle file.

    :param path: Filepath to file
    :param decompress: Allows Track objects to decompress (lazy evaluation)
    :return: Loaded object
    :rtype: Any
    """
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    if isinstance(data, Track) and decompress:
        data.decompress()  # Re-allow Lazy Evaluation.
    return data
