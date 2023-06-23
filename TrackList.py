from Cache import *


class TrackList(Dict[AnyStr, Track]):
    class_name: AnyStr = "TrackList"
    hashes: Hashes = {}  # TODO: Change to HashSet

    def __setitem__(self: class_name, key, value: Track):
        super().__setitem__(key, value)
        if value is Track:
            _hashes = value.hashes
            for key in _hashes:
                if key not in self.hashes:
                    self.hashes[key] = {}
                for _title in _hashes[key]:
                    self.hashes[key][_title] = _hashes[key][_title]
            # self.Hashes += value.get_hashes()  # TODO: Check that this works

    def append(self: class_name, value: Track):
        self.__setitem__(value.title, value)

    def get_track_by_sample(self: class_name, sample: Hashes) -> Optional[AnyStr]:
        """
        Returns the Track from self that most closely matches the given sample.

        :param sample: Hashes of sample to match with.
        :type sample: Hashes
        :return: Title of matched Track, if found.
        :rtype Optional[AnyStr]:
        """
        _best_match = None
        _best_score = 0
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
    def calculate_similarity_score(time_offset1: int, time_offset2: int) -> int:
        """
        Calculates the similarity score between two time offsets.

        :param time_offset1: Time offset of the first anchor.
        :type time_offset1: int
        :param time_offset2: Time offset of the second anchor.
        :type time_offset2: int
        :return: Similarity score.
        :rtype: int
        """
        # TODO: Implement the similarity score calculation logic
        # You can use the methodology described in the Shazam article (Section 2.3)
        # Calculate a score based on the time offset difference
        _score = abs(time_offset1 - time_offset2)

        return _score
