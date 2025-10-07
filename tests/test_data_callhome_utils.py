from dr_sad.data import callhome_utils


class TestRemoveOverlap:
    def test_no_overlap(self):
        segments = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == segments

    def test_one_overlap(self):
        segments = [(0.0, 1.0), (0.5, 2.0), (3.0, 4.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == [(0.0, 2.0), (3.0, 4.0)]

    def test_multiple_overlaps(self):
        segments = [(0.0, 1.0), (0.5, 2.0), (1.5, 3.0), (4.0, 5.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == [(0.0, 3.0), (4.0, 5.0)]

    def test_touching_segments(self):
        segments = [(0.0, 1.0), (1.0, 2.0), (3.0, 4.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == [(0.0, 2.0), (3.0, 4.0)]

    def test_inside_segment(self):
        segments = [(0.0, 3.0), (1.0, 2.0), (4.0, 5.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == [(0.0, 3.0), (4.0, 5.0)]

    def test_type_output(self):
        segments = [(0.0, 1.0), (0.5, 2.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert all(isinstance(seg, tuple) for seg in merged)

    def test_single_segment(self):
        segments = [(0.0, 1.0)]
        merged = callhome_utils.remove_overlap(segments)
        assert merged == [(0.0, 1.0)]


class TestRoundRobin:
    def test_equal_length(self):
        list1 = [1, 2, 3]
        list2 = ["a", "b", "c"]
        result = list(callhome_utils.roundrobin(list1, list2))
        assert result == [1, "a", 2, "b", 3, "c"]

    def test_unequal_length(self):
        list1 = [1, 2, 3]
        list2 = ["a", "b", "c", "d"]
        result = list(callhome_utils.roundrobin(list1, list2))
        assert result == [1, "a", 2, "b", 3, "c", "d"]

    def test_empty_input(self):
        list1: list[None] = []
        list2 = ["a", "b"]
        result = list(callhome_utils.roundrobin(list1, list2))
        assert result == ["a", "b"]

    def test_single_input(self):
        list1 = [1, 2]
        result = list(callhome_utils.roundrobin(list1))
        assert result == [1, 2]

    def test_multiple_lists(self):
        list1 = [1, 2, 3]
        list2 = ["a"]
        list3 = [True, False]
        list4 = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = list(callhome_utils.roundrobin(list1, list2, list3, list4))
        assert result == [1, "a", True, 0.1, 2, False, 0.2, 3, 0.3, 0.4, 0.5]

    def test_no_input(self):
        result = list(callhome_utils.roundrobin())
        assert result == []


class TestFillGaps:
    def test_fill_gaps_basic(self):
        sample = {
            "timestamps_start": [0.0, 1.0, 3.0],
            "timestamps_end": [1.0, 2.0, 4.0],
            "speakers": ["A", "None", "B"],
            "audio": {"array": [0] * 16000 * 5, "sampling_rate": 16000},
        }
        filled = callhome_utils.fill_gaps(sample)
        expected_starts = [0.0, 1.0, 2.0, 3.0, 4.0]
        expected_ends = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_speakers = ["A", "None", "None", "B", "None"]
        assert filled["timestamps_start"] == expected_starts
        assert filled["timestamps_end"] == expected_ends
        assert filled["speakers"] == expected_speakers

    def test_fill_gaps_no_silence(self):
        sample = {
            "timestamps_start": [0.0, 2.0],
            "timestamps_end": [2.0, 4.0],
            "speakers": ["A", "B"],
            "audio": {"array": [0] * 16000 * 5, "sampling_rate": 16000},
        }
        filled = callhome_utils.fill_gaps(sample)
        expected_starts = [0.0, 2.0, 4.0]
        expected_ends = [2.0, 4.0, 5.0]
        expected_speakers = ["A", "B", "None"]
        assert filled["timestamps_start"] == expected_starts
        assert filled["timestamps_end"] == expected_ends
        assert filled["speakers"] == expected_speakers
