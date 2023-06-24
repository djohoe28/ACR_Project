from Track import *

from typing import List


def cache_stfts(sr: Integer = OPTIONS["SampleRate"], n_fft: Integer = OPTIONS["NFFT"],
                hop_length: Integer = OPTIONS["HopLength"], win_length: Integer = OPTIONS["WindowLength"],
                verbose: Boolean = OPTIONS["DEBUG"]) -> np.ndarray:
    """Cache the STFTs of all Tracks in the database."""
    for title in get_titles(False):
        if verbose:
            print(f"Caching {title}...")
        win_length = win_length if win_length is not None else n_fft
        hop_length = hop_length if hop_length is not None else win_length // 4
        y, sr = librosa.load(os.path.join(OPTIONS["DatabasePath"], f"{title}.wav"), sr=sr)
        _stft = librosa.stft(y, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        np.save(os.path.join(OPTIONS["CachePath"], Properties.STFT.name, f"{title}.npy"))
        return _stft


def cache_constellations_and_asterisms(maxima_width: Integer = OPTIONS["MaximaFilterWidth"],
                                       maxima_height: Integer = OPTIONS["MaximaFilterHeight"]):
    """Cache the Constellations of all Tracks in the database."""
    for _title in get_titles(False):
        _constellation = constellation_from_cached_stft(_title, maxima_width, maxima_height)
        np.save(os.path.join(OPTIONS["CachePath"], Properties.Constellation.name, f"{_title}.npy"), _constellation)
        np.save(os.path.join(OPTIONS["CachePath"], Properties.Asterism.name, f"{_title}.npy"),
                nonzero_to_array(_constellation))  # Would be a waste not to do this.


def cache_asterisms():
    """Cache the Asterisms of all Tracks in the database."""
    for _title in get_titles(False):
        _asterism = asterism_from_cached_constellation(_title)
        np.save(os.path.join(OPTIONS["CachePath"], Properties.Asterism.name, f"{_title}.npy"), _asterism)


def cache_hashes():
    for _title in get_titles(False):
        _asterism = asterism_from_cached_constellation(_title)
        np.save(os.path.join(OPTIONS["CachePath"], Properties.Asterism.name, f"{_title}.npy"), _asterism)
    pass


def cache_database_by_step():
    cache_stfts()
    cache_constellations_and_asterisms()
    # cache_asterisms()  # NOTE: Done above
    cache_hashes()


def constellation_from_cached_stft(title: AnyStr,
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
    :return: Constellation of self
    :rtype: np.ndarray
    """
    _stft = np.abs(np.load(os.path.join(OPTIONS["CachePath"], Properties.STFT.name, f'{title}.npy')))
    _structure = np.ones((maxima_width, maxima_height))
    # Apply a boolean filter to each point => (new_point = True if point is maxima else False).
    _is_maxima = maximum_filter(_stft, footprint=_structure, mode='constant') == _stft
    # Create an "image mask" of the background
    _background = (_stft == 0)
    # Erode background to subtract from maxima mask in order to ignore the border points (they're technically extrema.)
    _eroded = binary_erosion(_background, structure=_structure, border_value=1)
    # Apply XOR to remove eroded background (i.e: border points) from result
    return _is_maxima ^ _eroded


def asterism_from_cached_constellation(title: AnyStr) -> np.ndarray:
    """
    Load an Asterism (local maxima) list from cached Constellation.

    :return: Asterism of self
    :rtype: np.ndarray
    """
    return nonzero_to_array(np.load(os.path.join(OPTIONS["CachePath"], Properties.Constellation.name, f'{title}.npy')))


def hashes_from_cached_constellation(title: AnyStr, target_width: Integer = OPTIONS["TargetZoneWidth"],
                                     target_height: Integer = OPTIONS["TargetZoneHeight"],
                                     time_offset: Integer = OPTIONS["TargetZoneTimeOffset"]) -> Hashes:
    """
    Get a Hash Set dictionary of self's Constellation hashes.

    :param title: Title of song to cache
    :type title: AnyStr
    :param target_width: Width of Target Zone
    :type target_width: Integer
    :param target_height: Height of Target Zone
    :type target_height: Integer
    :param time_offset: Time (X-axis) Offset of Target Zone
    :type time_offset: Integer
    :return: Hash Set of self's hashes { Hash -> {title -> offset} }
    :rtype: Dict[AnyStr, Dict[AnyStr, Integer]]
    """
    _constellation = np.load(os.path.join(OPTIONS["CachePath"], Properties.Constellation.name, f'{title}.npy'))
    _asterism = nonzero_to_array(
        _constellation)  # np.load(os.path.join(OPTIONS["CachePath"], Properties.Asterism.name, f'{title}.npy'))  #
    """
    # Load Metadata (if available)
    _path = os.path.join(OPTIONS["DatabasePath"], f'{title}.wav')
    _duration = 0
    if sf.check_format(_path):
        _metadata = sf.info(_path)
        if hasattr(_metadata, "duration"):  # Guaranteed by SoundFileInfo class.
            _duration = _metadata.duration  # NOTE: Overwrites (len(y)/sr), just in case.
    if _duration == 0:
        np.save("ERROR.npy", np.array([-1]))
    win_length = OPTIONS["WindowLength"] if OPTIONS["WindowLength"] is not None else OPTIONS["NFFT"]
    hop_length = OPTIONS["HopLength"] if OPTIONS["HopLength"] is not None else win_length // 4
    """
    _shape = _constellation.shape  # (win_length // 2 + 1, (_duration - win_length) // hop_length + 1)
    _hashes = {}
    for _anchor in _asterism:
        # Find Boundaries of Target Zone
        _t_min = _anchor[0] + time_offset  # Left
        _t_max = _t_min + target_width  # Right
        _f_min = _anchor[1] - target_height // 2  # Top
        _f_max = _f_min + target_height // 2  # Bottom
        # Clip appropriately (do not leave confines of array, nor go backwards in time.)
        _t_min, _t_max = np.clip(a=(_t_min, _t_max), a_min=_anchor[0] + 1, a_max=_shape[0] - 1)
        _f_min, _f_max = np.clip(a=(_f_min, _f_max), a_min=0, a_max=_shape[1] - 1)
        # Create Target Zone from Constellation; Where maxima in Boundaries: True, Else: False
        _target_zone_bounds = (slice(_t_min, _t_max), slice(_f_min, _f_max))
        _target_zone = np.zeros_like(_constellation)
        _target_zone[_target_zone_bounds] = _constellation[_target_zone_bounds]
        _targets = nonzero_to_array(_target_zone)
        # Create Hashes from anchor-target pairs
        for _target in _targets:
            _target_hash = f"[{_anchor[1]}:{_target[1]}:{_anchor[0] - _target[0]}]"
            _hashes[_target_hash] = {title: _anchor[1]}
    return _hashes


def get_titles(cached: Boolean = False, path: Optional[AnyStr] = None) -> List[AnyStr]:
    """
    Load a list of titles from cache if cached: else load from database.

    :param path: Path to database/cache directory
    :type path: Optional[AnyStr]
    :param cached: Load titles from a cache?
    :type cached: Boolean
    :return: List of titles available.
    :rtype: List[AnyStr]
    """
    if path is None:
        path = OPTIONS["CachePath"] if cached else OPTIONS["DatabasePath"]
    _titles: List[AnyStr] = []
    if cached:
        # Get titles from cached/pickled Tracks
        for _filename in os.listdir(os.path.join(path, Properties.Pickle.name)):
            if _filename.endswith('.pkl'):
                _titles.append(os.path.basename(os.path.splitext(
                    os.path.join(path, Properties.Pickle.name, _filename))[0]))
    else:
        # Get titles from sound files
        _ext: (Union[AnyStr, Tuple[AnyStr, ...]]) = OPTIONS["RecordingExtension"]
        for _filename in os.listdir(path):
            if _filename.endswith(_ext):
                _title = os.path.basename(os.path.splitext(os.path.join(path, _filename))[0])
                _titles.append(_title)
    return _titles


def make_directories() -> None:
    """Creates the Cache directories if they don't already exist."""
    _cache_path = OPTIONS["CachePath"]
    _figure_path = OPTIONS["FigurePath"]
    if not os.path.exists(_cache_path):
        os.mkdir(_cache_path)
    if not os.path.exists(_figure_path):
        os.mkdir(_figure_path)
    for _prop in Properties:
        if _prop in [Properties.Pickle, Properties.Duration, Properties.Shape]:
            continue
        _cache_path_folder = os.path.join(_cache_path, _prop.name)
        _figure_path_folder = os.path.join(_figure_path, _prop.name)
        if not os.path.exists(_cache_path_folder):
            os.mkdir(_cache_path_folder)
        if not os.path.exists(_figure_path_folder):
            os.mkdir(_figure_path_folder)


def cache_database_by_track(path: AnyStr = OPTIONS["DatabasePath"],
                            plot_figures: Boolean = False, verbose: Boolean = True) -> List[AnyStr]:
    """
    Generate cache from files available in database.

    :param path: Path to database
    :param path: AnyStr
    :param plot_figures:
    :param verbose: Print additional data.
    :type verbose: Boolean
    :return: List of titles available in cache.
    :rtype: List[AnyStr]
    """
    _titles: List[AnyStr] = get_titles(False)[:1]
    _shapes: Dict[AnyStr, Tuple[Integer, Integer]] = {}
    _durations: Dict[AnyStr, Float] = {}
    make_directories()
    for _i in range(len(_titles)):
        _title: AnyStr = _titles[_i]
        print(f'{_i + 1} / {len(_titles)} = {_title}')
        # NOTE: Pickle *sometimes* runs the _evaluate functions if not track.is_compressed!
        _track: Track = Track(os.path.join(path, f'{_title}.{OPTIONS["RecordingExtension"]}'))
        _track.cache_property(Properties.Pickle, plot_figures)
        _track.cache_property(Properties.Y, plot_figures)
        _track.cache_property(Properties.STFT, plot_figures)
        _track.cache_property(Properties.Constellation, plot_figures)
        _track.cache_property(Properties.Asterism, plot_figures)
        _durations[_title] = _track.duration
        _shapes[_title] = _track.shape
    save_as_pickle(os.path.join(OPTIONS["CachePath"], Properties.Duration.name), _durations)
    save_as_pickle(os.path.join(OPTIONS["CachePath"], Properties.Shape.name), _shapes)
    return _titles


def optimize_maxima_window(coords: np.ndarray = np.array([(64, 36), (80, 45), (96, 54), (112, 63), (128, 72), (144, 81),
                                                          (160, 90), (176, 99), (192, 108), (208, 117), (224, 126),
                                                          (240, 135), (256, 144)][:4])):
    # TODO: Assumes cached memory
    stfts = {}
    for title in os.listdir("Cache/STFT"):
        print("Reading", title, "...")
        stfts[title.removesuffix('.npy')] = np.abs(np.load(f"Cache/STFT/{title}"))
    _results = np.zeros((len(stfts), coords.shape[0]), dtype=int)
    ctr = 0
    for i, (width, height) in enumerate(coords):
        ctr += 1
        if not (OPTIONS["MaximaMinSide"] <= width <= OPTIONS["MaximaMaxSide"]):
            break
        if not (OPTIONS["MaximaMinSide"] <= height <= OPTIONS["MaximaMaxSide"]):
            break
        if not (width + height) <= OPTIONS["MaximaMaxPerimeter"]:
            break
        _structure = np.ones((width, height))
        for t, title in enumerate(get_titles(False)):
            stft = stfts[title]
            # Apply a boolean filter to each point => (new_point = True if point is maxima else False).
            _is_maxima = maximum_filter(stft, footprint=_structure, mode='constant') == stft
            # Create an "image mask" of the background
            _background = (stft == 0)
            # Erode background to subtract from maxima mask in order to ignore the border points (extrema.)
            _eroded = binary_erosion(_background, structure=_structure, border_value=1)
            # Apply XOR to remove eroded background (i.e: border points) from result
            _cmap = _is_maxima ^ _eroded
            _cmap_len = len(_cmap.nonzero()[0])
            _results[t, i] += _cmap_len if (1000 < _cmap_len < 10000) else 0
            print(f"[{i}] = ({width},{height})\t[{t}] = {_cmap_len}\t"
                  f"{ctr} / {_results.size} = {round(100.0 * ctr / _results.size, 2)}%")
            if _cmap_len < 1000:
                break
    _results = _results.mean(axis=0)
    return _results
