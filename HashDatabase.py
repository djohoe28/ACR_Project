from Cache import *


class HashDatabase(HashSet):
    class_name: AnyStr = "HashDatabase"

    # is_cached: Boolean = True
    # hashes: Hashes = {}  # TODO: Change to HashSet
    #
    # def __setitem__(self: class_name, key, value: Track):
    #     super().__setitem__(key, value)
    #     if value is Track:
    #         print(f"Hash generation START | {value.title}")
    #         _hashes = value.hashes
    #         print(f"Hash generation END | {value.title}")
    #         for key in _hashes:
    #             if key not in self.hashes:
    #                 self.hashes[key] = {}
    #             for _title in _hashes[key]:
    #                 self.hashes[key][_title] = _hashes[key][_title]

    def append_hashes(self: class_name, hashes: HashSet):
        for _hash in hashes:
            if _hash not in self:
                self[_hash] = {}
            for _title in hashes[_hash]:
                self[_hash][_title] = hashes[_hash][_title]

    def append_track(self: class_name, track: Track):
        self.append_hashes(track.hashes)

    def match_from_sample(self: class_name, sample: HashSet) -> Optional[AnyStr]:
        """
        Returns the Track from self that most closely matches the given sample.

        :param sample: Hashes of sample to match with.
        :type sample: Hashes
        :return: Title of matched Track, if found.
        :rtype Optional[AnyStr]:
        """
        _best_match: Optional[AnyStr] = None
        _best_score: Float = 0
        _track_titles = {}
        for _hash_key in sample:
            if _hash_key in self:
                _track_info = self[_hash_key]
                for _track_title, _time_offset in _track_info.items():
                    if _track_title not in _track_titles:
                        _track_titles[_track_title] = {}
                    for _sample_title in sample[_hash_key]:
                        _offset = self.find_delta_offset(_time_offset, sample[_hash_key][_sample_title])
                        if _offset not in _track_titles[_track_title]:
                            _track_titles[_track_title][_offset] = 0
                        _track_titles[_track_title][_offset] += 1
        for _track_title in _track_titles:
            for _offset in _track_titles[_track_title]:
                if _track_titles[_track_title][_offset] > _best_score:
                    _best_score = _track_titles[_track_title][_offset]
                    _best_match = _track_title
        return _best_match

    @staticmethod
    def find_delta_offset(track_offset: int, sample_offset: int) -> int:
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
        _score = int(abs(track_offset - sample_offset))

        return _score


def tracklist_from_database(path: AnyStr = OPTIONS["DatabasePath"],
                            ext: (Union[AnyStr, Tuple[AnyStr, ...]]) = OPTIONS["RecordingExtension"]) -> HashDatabase:
    """
    Load all files in the database directory into a new TrackList.

    :param path: Path to directory
    :param ext: Supported file extension(s)
    :return: Generated TrackList
    :rtype: HashDatabase
    """
    _tracklist = HashDatabase()
    for _filename in os.listdir(path):
        if _filename.endswith(ext):
            print(f"#{len(_tracklist) + 1} = {_filename}...")
            _track = Track(os.path.join(path, _filename))
            _tracklist.append(_track)
    # _tracklist.append(Track(librosa.ex('choice')))
    return _tracklist
