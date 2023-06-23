import datetime

from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from TrackList import *

# Load Cache
print(f'Starting! {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S + 3h")}')
titles: list[str] = []
for _filename in os.listdir(os.path.join(OPTIONS["CachePath"], "Pickles")):
    if _filename.endswith('.pkl'):
        titles.append(os.path.basename(os.path.splitext(os.path.join(OPTIONS["CachePath"], "Pickles", _filename))[0]))
print(f'Loading! {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S + 3h")}')
db_tracks: dict[str, Track] = {}
db_stft: dict[str, np.ndarray] = {}
db_anchors: dict[str, np.ndarray] = {}
for _title in titles:
    db_tracks[_title] = load_from_pickle(os.path.join(OPTIONS["CachePath"], "Pickles", f'{_title}.pkl'))
    db_anchors[_title] = np.load(os.path.join(OPTIONS["CachePath"], "Anchors", f"{_title}.npy"))
    # db_stft[_title] = np.load(os.path.join(OPTIONS["CachePath"], "STFTs", f"{_title}.npy"))


def get_anchors(title: str, maxima_width, maxima_height):
    _stft = np.abs(db_stft[title])
    _structure = np.ones((maxima_width, maxima_height))
    # Apply a boolean filter to each point => (new_point = True if point is maxima else False).
    _maxima_bool_mask = scipy.ndimage.maximum_filter(_stft, footprint=_structure, mode='constant') == _stft
    # Create an "image mask" of the background
    _background = (_stft == 0)
    # Erode background to subtract from _maxima_bool_mask in order to ignore the border points (technically extrema)
    _eroded = scipy.ndimage.binary_erosion(_background, structure=_structure, border_value=1)
    # Apply XOR to remove eroded background (i.e: border points) from result
    return _maxima_bool_mask ^ _eroded


def get_len_hashes(title: str, maxima_width: int, maxima_height: int,
                   target_width: int, target_height: int, target_offset: int) -> int:
    """Get a Hash Set dictionary of self's Constellation Map hashes."""
    _hashes_len = 0
    _track = db_tracks[title]
    _anchors = db_anchors[title]  # TODO: Use get_anchors if maxima_width and maxima_height are not default.
    for _anchor in _anchors:
        # Find Boundaries of Target Zone
        _t_min = _anchor[0] + target_offset  # Left
        _t_max = _t_min + target_width  # Right
        _f_min = _anchor[1] - target_height // 2  # Top
        _f_max = _f_min + target_height // 2  # Bottom
        # Clip appropriately (do not leave confines of array, nor go backwards in time.)
        _t_min, _t_max = np.clip(a=(_t_min, _t_max), a_min=_anchor[0] + 1, a_max=_track.shape[0] - 1)
        _f_min, _f_max = np.clip(a=(_f_min, _f_max), a_min=0, a_max=_track.shape[1] - 1)
        # Create Target Zone from Constellation Map; Where maxima in Boundaries: True, Else: False
        _hashes_len += len(filter_coordinates(_anchors, x_min=_t_min, x_max=_t_max, y_min=_f_min, y_max=_f_max))
    return _hashes_len


def get_value(title: str, maxima_width: int, maxima_height: int,
              target_width: int, target_height: int, target_offset: int) -> int:
    _hashes_len: int = get_len_hashes(title=title, maxima_width=maxima_width, maxima_height=maxima_height,
                                      target_width=target_width, target_height=target_height,
                                      target_offset=target_offset)
    return _hashes_len


class ValueEstimator(BaseEstimator):
    def __init__(self, maxima_width=None, maxima_height=None,
                 target_width=None, target_height=None, target_offset=None):
        self.maxima_width = maxima_width
        self.maxima_height = maxima_height
        self.target_width = target_width
        self.target_height = target_height
        self.target_offset = target_offset

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [get_value(title, self.maxima_width, self.maxima_height,
                          self.target_width, self.target_height, self.target_offset) for title in X]


print(f'Training! {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S + 3h")}')
param_grid = {
    'maxima_width': [OPTIONS["MaximaFilterWidth"]],  # 5 Seconds
    'maxima_height': [OPTIONS["MaximaFilterWidth"]],  # 1024 bins
    'target_width': np.linspace(6174, 22050, 10, dtype=int),  # 140ms ~ 500ms
    'target_height': np.linspace(1, OPTIONS["SampleRate"] // 2, 10, dtype=int),  # Nyquist Frequency
    'target_offset': np.linspace(1, 50, 10, dtype=int)
}
scoring = make_scorer(lambda y_true, y_pred: np.mean(np.abs(np.array(y_pred) - 2500)))
grid_search = GridSearchCV(ValueEstimator(), param_grid, cv=2, n_jobs=-1, scoring=scoring, verbose=3)
grid_search.fit(titles, [2500] * len(titles))
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print(f'Done! {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S + 3h")}')
