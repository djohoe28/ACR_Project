from Cache import *


class TrackList(Dict[AnyStr, Track]):
    class_name: AnyStr = "TrackList"
    hashes: Hashes = {}  # TODO: Change to HashSet

    def __setitem__(self: class_name, key, value: Track):
        super().__setitem__(key, value)
        if value is Track:
            print(f"Hash generation START | {value.title}")
            _hashes = value.hashes
            print(f"Hash generation END | {value.title}")
            for key in _hashes:
                if key not in self.hashes:
                    self.hashes[key] = {}
                for _title in _hashes[key]:
                    self.hashes[key][_title] = _hashes[key][_title]

    def append(self: class_name, value: Track):
        self.__setitem__(value.title, value)

    def match_from_sample(self: class_name, sample: Hashes) -> Optional[AnyStr]:
        """
        Returns the Track from self that most closely matches the given sample.

        :param sample: Hashes of sample to match with.
        :type sample: Hashes
        :return: Title of matched Track, if found.
        :rtype Optional[AnyStr]:
        """
        _best_match: Optional[AnyStr] = None
        _best_score: Float = 0
        for _hash_key in sample:
            if _hash_key in self.hashes:
                _track_info = self.hashes[_hash_key]
                for _track_title, _time_offset in _track_info.items():
                    _score = self.calculate_similarity_score(_time_offset, sample[_hash_key][_track_title])
                    if _score > _best_score:
                        _best_match = _track_title
                        _best_score = _score
        return _best_match

    @staticmethod
    def calculate_similarity_score(track_offset: int, sample_offset: int) -> int:
        """
        Calculates the similarity score between two time offsets.

        :param track_offset: Time offset of the track's anchor.
        :type track_offset: int
        :param sample_offset: Time offset of the sample's anchor.
        :type sample_offset: int
        :return: Similarity score.
        :rtype: int
        """
        # TODO: Implement the similarity score calculation logic
        # You can use the methodology described in the Shazam article (Section 2.3)
        # Calculate a score based on the time offset difference
        _score = abs(track_offset - sample_offset)

        return _score


def tracklist_from_database(path: AnyStr = OPTIONS["DatabasePath"],
                            ext: (Union[AnyStr, Tuple[AnyStr, ...]]) = OPTIONS["RecordingExtension"]) -> TrackList:
    """
    Load all files in the database directory into a new TrackList.

    :param path: Path to directory
    :param ext: Supported file extension(s)
    :return: Generated TrackList
    :rtype: TrackList
    """
    _tracklist = TrackList()
    for _filename in os.listdir(path):
        if _filename.endswith(ext):
            print(f"#{len(_tracklist) + 1} = {_filename}...")
            _track = Track(os.path.join(path, _filename))
            _tracklist.append(_track)
    # _tracklist.append(Track(librosa.ex('choice')))
    return _tracklist
