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
        assert (
            all(isinstance(seg[0], float) and isinstance(seg[1], float)
            for seg in merged)
        )
