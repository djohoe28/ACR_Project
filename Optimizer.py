from HashDatabase import *

from typing import Sequence

from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


class Cacher(object):
    """
    Handles caching of data (mostly used to expedite optimization training)

    :ivar expected: Expected value for score
    :type expected: Union[Integer, Float]
    :ivar param_grid:
    :type param_grid: Dict[AnyStr, Sequence]
    :ivar path:
    :type path: AnyStr
    :ivar verbose: Print additional data.
    :type verbose: Boolean
    :ivar is_cached:
    :type is_cached: Boolean
    :ivar titles: List of titles of Tracks available in database.
    :type titles: List[AnyStr]
    :ivar shape: Shape of training results array.
    :type shape: Tuple[Integer, Integer, Integer, Integer]
    :ivar get_t: Reverse Lookup Table of titles.
    :type get_t: Dict[AnyStr, Integer]
    :ivar get_w: Reverse Lookup Table of widths.
    :type get_w: Dict[Integer, Integer]
    :ivar get_h: Reverse Lookup Table of heights.
    :type get_h: Dict[Integer, Integer]
    :ivar get_o: Reverse Lookup Table of offsets.
    :type get_o: Dict[Integer, Integer]
    :ivar tracks: Dictionary of Track instances available in cache by Track title.
    :type tracks: Optional[Dict[AnyStr, Track]]
    :ivar asterism: Dictionary of Asterism arrays of Track instances available in cache by Track title
    :type asterism: Optional[Dict[AnyStr, np.ndarray]]
    :ivar cache: Cached memory
    :type cache: np.ndarray
    """
    class_name: AnyStr = "Cacher"

    def __init__(self: class_name, param_grid: Dict[AnyStr, Sequence], filepath: Optional[AnyStr] = None,
                 verbose: Boolean = True):
        """
        Cacher Constructor

        :param param_grid: Ranges of parameters to optimize.
        :type param_grid: Dict[AnyStr, Sequence]
        :param filepath: Filepath to cache file.
        :type filepath: AnyStr
        :param verbose: Print additional data.
        :type verbose: Boolean
        """
        self.expected: Union[Integer, Float] = OPTIONS["AverageHashCount"]
        self.param_grid: Dict[AnyStr, Sequence] = param_grid
        self.path: AnyStr = filepath
        self.verbose: Boolean = verbose
        self.is_cached: Boolean = self.path is not None
        self.titles: List[AnyStr] = list()
        for _filename in os.listdir(os.path.join(OPTIONS["CachePath"], Properties.Pickle.name)):
            if _filename.endswith('.pkl'):
                self.titles.append(
                    os.path.basename(
                        os.path.splitext(os.path.join(OPTIONS["CachePath"], Properties.Pickle.name, _filename))[0]))
        self.shape: Tuple[Integer, Integer, Integer, Integer] = (
            len(self.titles), len(param_grid['width']), len(param_grid['height']), len(param_grid['offset']))
        self.get_t: Dict[AnyStr, Integer] = {self.titles[_i]: _i for _i in range(self.shape[0])}
        self.get_w: Dict[Integer, Integer] = {self.param_grid['width'][_i]: _i for _i in range(self.shape[1])}
        self.get_h: Dict[Integer, Integer] = {self.param_grid['height'][_i]: _i for _i in range(self.shape[2])}
        self.get_o: Dict[Integer, Integer] = {self.param_grid['offset'][_i]: _i for _i in range(self.shape[3])}
        self.tracks: Optional[Dict[AnyStr, Track]] = None if self.is_cached else {}
        self.asterism: Optional[Dict[AnyStr, np.ndarray]] = None if self.is_cached else {}
        self.cache: np.ndarray = np.zeros(self.shape)
        if not self.is_cached:
            self.generate_cache()
        self.cache = np.load(filepath)  # NOTE: Loads file even if is_cached to validate reproducibility.

    def generate_cache(self: class_name) -> None:
        """Caches amount of hashes generated from parameters in param_grid."""
        print(f'Caching! {timestamp()}')
        _ctr: Integer = 0
        for title in self.titles:
            self.tracks[title] = load_from_pickle(
                os.path.join(OPTIONS["CachePath"], Properties.Pickle.name, f'{title}.pkl'))
            self.asterism[title] = np.load(os.path.join(OPTIONS["CachePath"], Properties.Asterism.name, f"{title}.npy"))
        for _t, _w, _h, _o in np.ndindex(*self.shape):  # np.ndindex raises warning it doesn't expect tuple; oversight?
            _title: AnyStr = self.titles[_t]
            _width: Integer = self.param_grid['width'][_w]
            _height: Integer = self.param_grid['height'][_h]
            _offset: Integer = self.param_grid['offset'][_o]
            _ctr += 1
            if self.verbose:
                print(f"T {_t + 1}/{self.shape[0]} | "
                      f"W {_w + 1}/{self.shape[1]} | "
                      f"H {_h + 1}/{self.shape[2]} | "
                      f"O {_o + 1}/{self.shape[3]} | "
                      f"= {_ctr}/{self.cache.size}")
            self.cache[_t, _w, _h, _o] = self.get_len_hashes(_title, _width, _height, _offset)
        self.path = os.path.join(OPTIONS["CachePath"], OPTIONS["CacherFile"])
        np.save(self.path, self.cache)

    def get_len_hashes(self: class_name, title: AnyStr, width: Integer, height: Integer, offset: Integer) -> Integer:
        """
        Get a HashSet dictionary of self's Constellation hashes.

        :param title: Title of Track to evaluate
        :param width: Width of Target Zone to evaluate
        :param height: Height of Target Zone to evaluate
        :param offset: Offset of Target Zone to evaluate
        :return: Amount of Constellation hashes generated from the parameters.
        :rtype: Integer
        """
        if self.is_cached:
            return self.cache[self.get_t[title], self.get_w[width], self.get_h[height], self.get_o[offset]]
        _hashes_len: Integer = 0
        _track: Track = self.tracks[title]
        _asterism: np.ndarray = self.asterism[title]  # TODO: Use get_asterism() if window width/height are not default.
        for _anchor in _asterism:
            # Find Boundaries of Target Zone
            _t_min: Integer = _anchor[0] + offset  # Left
            _t_max: Integer = _t_min + width  # Right
            _f_min: Integer = _anchor[1] - height // 2  # Top
            _f_max: Integer = _f_min + height // 2  # Bottom
            # Clip appropriately (do not leave confines of array, nor go backwards in time.)
            _t_min, _t_max = np.clip(a=(_t_min, _t_max), a_min=_anchor[0] + 1, a_max=_track.shape[0] - 1)
            _f_min, _f_max = np.clip(a=(_f_min, _f_max), a_min=0, a_max=_track.shape[1] - 1)
            # Create Target Zone from Constellation; Where maxima in Boundaries: True, Else: False
            _hashes_len += len(filter_coordinates(_asterism, x_min=_t_min, x_max=_t_max, y_min=_f_min, y_max=_f_max))
        return _hashes_len

    def get_score(self: class_name, y_true, y_pred) -> Float:
        """
        Applies a scoring arithmetic to y_pred; 0 <= proximity to expected value <= 1

        :param y_true:
        :param y_pred:
        :return: Normalize score based on proximity to expected value
        :rtype: Float
        """
        average_pred = np.mean(y_pred)
        score = 1 - abs(average_pred - self.expected) / max(abs(average_pred - self.expected), abs(self.expected))
        return score


def train(param_grid: Dict[AnyStr, Sequence],
          filepath: Optional[AnyStr] = os.path.join(OPTIONS["CachePath"], OPTIONS["CacherFile"]),
          verbose: Boolean = True) -> (Any, Any):
    """
    Optimizes the given parameters using GridSearchCV training.

    :param param_grid: Sequence of values for each parameter's range to use when optimizing.
    :type param_grid: List[AnyStr, Sequence]
    :param filepath: Path to cache file.
    :type filepath: Optional[AnyStr]
    :param verbose: Print more data.
    :type verbose: Boolean
    :return: (Any, Any)
    """
    cacher = Cacher(param_grid, filepath, verbose)

    class ValueEstimator(BaseEstimator):
        """
        Estimator for hyperparameter training optimization.

        :cvar class_name: Name of class.
        :type class_name: AnyStr
        :ivar width: Width of Target Zone to estimate.
        :type width: Any
        :ivar height: Height of Target Zone to estimate.
        :type height: Any
        :ivar offset: Offset of Target Zone to estimate.
        :type offset: Any
        """
        class_name: AnyStr = "ValueEstimator"

        def __init__(self: class_name, width: Any = None, height: Any = None, offset: Any = None):
            """
            ValueEstimator Constructor

            :param width: Width of Target Zone to estimate.
            :type width: Any
            :param height: Height of Target Zone to estimate.
            :type height: Any
            :param offset: Offset of Target Zone to estimate.
            :type offset: Any
            """
            self.width = width
            self.height = height
            self.offset = offset

        def fit(self: class_name, X: Any, y: Any):
            """
            Apply fitting to model.

            :param X:
            :type X: any
            :param y:
            :type y: any
            :return: None
            :rtype: None
            """
            return self

        def predict(self, X: Iterable):
            """
            Predict using model.

            :param X: Input array
            :type X: Iterable
            :return: List of hash amount "predictions"
            :rtype: List[Integer]
            """
            return [cacher.get_len_hashes(_title, self.width, self.height, self.offset) for _title in X]

    print(f'Training! {timestamp()}')
    scoring: Any = make_scorer(lambda y_true, y_pred: cacher.get_score(y_true, y_pred))
    grid_search = GridSearchCV(ValueEstimator(), param_grid, cv=5, n_jobs=-1, scoring=scoring, verbose=3)
    grid_search.fit(cacher.titles, [2500] * len(cacher.titles))
    best_params_: Any = grid_search.best_params_
    best_score_: Any = grid_search.best_score_
    print("Best Score:", best_score_)
    print("Best Parameters:", best_params_)
    print(f'Trained! {timestamp()}')
    return best_params_, best_score_


def optimize_parameters() -> Tuple[Any, Any]:
    """Returns the training results of default parameters."""
    best_params: Any
    best_score: Any
    param_samples: Integer = OPTIONS["ParameterSamples"]
    param_grid: Dict[AnyStr, Sequence] = {
        'width': np.linspace(1, OPTIONS["SampleRate"] // 2 + 1, param_samples, dtype=Integer),  # 140ms ~ 500ms
        'height': np.linspace(1, OPTIONS["SampleRate"] // 2 + 1, param_samples, dtype=Integer),  # Nyquist Frequency
        'offset': np.linspace(1, OPTIONS["SampleRate"] // 2 + 1, param_samples, dtype=Integer),
    }
    best_params, best_score = train(param_grid, filepath=None)
    return best_params, best_score
