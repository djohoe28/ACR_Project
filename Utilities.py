from enum import Enum
from typing import Union, Iterable, Dict, AnyStr, Tuple, Any

import sys
import datetime
import numpy as np

# Documentation
Integer = int
Float = float
Boolean = bool
Hashes = Dict[AnyStr, Dict[AnyStr, Integer]]


def is_google_colab() -> Boolean:
    """Returns whether this program is running on the Google Colab service."""
    return 'google.colab' in sys.modules


# Global Parameters
OPTIONS: Dict[AnyStr, Any] = {
    "DEBUG": False,
    "CACHED": True,
    "COLAB": is_google_colab(),
    "Synchronous": True,
    "TimestampFormat": "%Y-%m-%d %H:%M:%S",
    "CacheFile": "hash.pkl",
    "SampleFile": "SAMPLE.wav",
    "FullFile": "FULL.wav",
    # Database
    "MakeDirMode": 0o666,
    "DatabasePath": "C:/Users/DJoho/Downloads/_Database ACR Shazam",
    "CachePath": "Cache",
    "FigurePath": "Figures",
    # Recording Properties
    "RecordingPath": "Recordings",  # Path for Recordings
    "RecordingExtension": "wav",  # File Extension for Recordings
    "RecordingDuration": 1,  # Duration of Recordings
    "SampleRate": 44100,  # Sample Rate
    "Channels": 2,  # Channels
    # Training Properties
    "ParameterSamples": 16,
    "MaximaMinSide": 36,
    "MaximaMaxSide": 256,
    "MaximaMaxPerimeter": 256,
    "AverageHashCount": 2500,
    "CacherFile": "training.npy",
    # Short-Time Fourier Transform (STFT) Parameters
    "NFFT": 2048,  # Number of FFTs (Fast Fourier Transforms)
    "HopLength": None,  # Hop Length - how many units should the STFT window move.
    "WindowLength": None,  # Window Length - what size is the STFT window.
    # Constellation - Maxima Filter Hyperparameters
    "MaximaFilterWidth": 64,  # Width of "window" used for finding maxima.
    "MaximaFilterHeight": 36,  # Height of "window" used for finding maxima.
    # Target Zone - Window Hyperparameters
    "TargetZoneWidth": 90,  # Width (X-axis, Time) |
    "TargetZoneHeight": 160,  # Height (Y-axis, Frequency) | 0 ~ Nyquist / 2
    "TargetZoneTimeOffset": 64,  # Time (X-axis) offset to move target zone "forwards". |
    # Searching and Scoring
    "Threshold": 0.5,
    "Distance": 10
}


class Return(Enum):
    """Status Code Enum"""
    SUCCESS = 1  # The function performed successfully.
    CANCEL = 0  # The function was cancelled.
    UNDEF_ERROR = -1  # All errors not mentioned below
    INDEX_ERROR = -2  # Index (out of range) Error
    VALUE_ERROR = -3  # Value (conversion) Error
    AUDIO_ERROR = -4  # Audio (playback) Error
    FILES_ERROR = -5  # File (soundfile) Error
    COLAB_ERROR = -6  # Google Colab Error


class Properties(Enum):
    Pickle = "get_compressed"
    Y = "y"
    STFT = "stft"
    Constellation = "constellation"
    Asterism = "asterism"
    Hash = "hashes"
    Duration = "Durations.pkl"
    Shape = "Shapes.pkl"


def timestamp(strftime: AnyStr = OPTIONS["TimestampFormat"]):
    """
    Timestamp of current time in given format.

    :param strftime:
    :type strftime: AnyStr
    :return: Timestamp string
    :rtype: AnyStr
    """
    return datetime.datetime.now().strftime(strftime)


def tuple_to_array(arr: Union[Iterable, Tuple]) -> np.ndarray:
    """
    Converts a tuple of N D-axis coordinates into an array of shape (N, D)

    :param arr: Array of shape (D, N)
    :type arr: np.ndarray
    :return: Array of shape (N, D)
    :rtype: np.ndarray
    """
    return np.column_stack((arr[0], arr[1]))


def nonzero_to_array(arr: np.ndarray) -> np.ndarray:
    """
    Returns an array of all N non-zero coordinates of the given D-dimensional array as an array of shape (N, D)

    :param arr: Array to find non-zero coordinates of
    :type arr: np.ndarray
    :return: Array of non-zero coordinates of given array
    :rtype: np.ndarray
    """
    return tuple_to_array(arr.nonzero())


def filter_coordinates(coordinates, x_min, x_max, y_min, y_max):
    """
    Returns a masked copy of given array within given bounds

    :param coordinates: Matrix of coordinates
    :type coordinates: Any
    :param x_min: Left boundary
    :type x_min: Any
    :param x_max: Right boundary
    :type x_max: Any
    :param y_min: Bottom boundary
    :type y_min: Any
    :param y_max: Upper boundary
    :type y_max: Any
    :return:
    """
    mask = (coordinates[:, 0] >= x_min) & (coordinates[:, 0] <= x_max) & (coordinates[:, 1] >= y_min) & (
            coordinates[:, 1] <= y_max)
    return coordinates[mask]
