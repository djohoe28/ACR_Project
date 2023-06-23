from Track import *

from typing import List


def get_constellation_map(title: AnyStr,
                          maxima_width=OPTIONS["MaximaFilterWidth"],
                          maxima_height=OPTIONS["MaximaFilterHeight"]) -> np.ndarray:
    """
    Get a copy of self's STFT where all non-peaks are set to 0.
    Adapted from https://stackoverflow.com/a/3689710

    :param title: Title of track (used for caching)
    :type title: AnyStr
    :param maxima_width: Width of Maxima Filter Window
    :type maxima_width: Integer
    :param maxima_height: Height of Maxima Filter Window
    :type maxima_height: Integer
    :return: Constellation Map of self
    :rtype: np.ndarray
    """
    _stft = np.abs(np.load(os.path.join(OPTIONS["CachePath"], "STFTs", f'{title}.npy')))
    _structure = np.ones((maxima_width, maxima_height))
    # Apply a boolean filter to each point => (new_point = True if point is maxima else False).
    _is_maxima = maximum_filter(_stft, footprint=_structure, mode='constant') == _stft
    # Create an "image mask" of the background
    _background = (_stft == 0)
    # Erode background to subtract from maxima mask in order to ignore the border points (they're technically extrema.)
    _eroded = binary_erosion(_background, structure=_structure, border_value=1)
    # Apply XOR to remove eroded background (i.e: border points) from result
    return _is_maxima ^ _eroded


def get_anchors(title: AnyStr) -> np.ndarray:
    """
    Get a list of anchors (points) of self's Constellation Map.

    :return: Anchors of self
    :rtype: np.ndarray
    """
    return nonzero_to_array(np.load(os.path.join(OPTIONS["CachePath"], "CMaps", f'{title}.npy')))


def get_titles() -> List[AnyStr]:
    """
    Return list of titles of tracks available in database (not necessarily in cache).

    :return: List of track available in database
    :rtype: List[AnyStr]
    """
    _db_path: AnyStr = OPTIONS["DatabasePath"]
    _ext: (Union[AnyStr, Tuple[AnyStr, ...]]) = OPTIONS["RecordingExtension"]
    _titles = list()
    for _filename in os.listdir(_db_path):  # TODO: Change to .pkl files when using Google Colab
        if _filename.endswith(_ext):
            _title = os.path.basename(os.path.splitext(os.path.join(_db_path, _filename))[0])
            _titles.append(_title)
    return _titles


def get_titles_from_cache(load: Boolean = False, verbose: Boolean = True) -> List[AnyStr]:
    """
    Generate cache from files available in database.

    :param load: if true: load .wav files; else: load .pkl files
    :type load: Boolean
    :param verbose: Print additional data.
    :type verbose: Boolean
    :return: List of titles available in cache.
    :rtype: List[AnyStr]
    """
    _durations: Dict[AnyStr, Float] = {}
    _titles = get_titles()
    _shapes = {}
    if load:
        for _i in range(len(_titles)):
            _title = _titles[_i]
            print(f'{_i + 1} / {len(_titles)} = {_title}')
            _track = Track(os.path.join(OPTIONS["DatabasePath"], f'{_title}.{OPTIONS["RecordingExtension"]}'))
            # NOTE: Pickle *sometimes* runs the _evaluate functions!
            save_as_pickle(os.path.join(OPTIONS["CachePath"], "Pickles", f'{_title}.pkl'), _track)
            # np.save(os.path.join(OPTIONS["CachePath"], "STFTs", f'{_title}.npy'), _track.stft)
            # np.save(os.path.join(OPTIONS["CachePath"], "CMaps", f'{_title}.npy'), _track.constellation_map)
            # np.save(os.path.join(OPTIONS["CachePath"], "Anchors", f'{_title}.npy'), _track.anchors)
        # save_as_pickle(os.path.join(OPTIONS["CachePath"], "Durations.pkl"), _durations)
        save_as_pickle(os.path.join(OPTIONS["CachePath"], "Shapes.plk"), _shapes)
    return _titles
