import os
import tempfile
from pathlib import Path

from dr_sad.data.utils import generate_rttm

valid_sample = {
    "timestamps_start": [0.0, 1.0, 3.0],
    "timestamps_end": [1.0, 2.0, 4.0],
    "speakers": ["A", "None", "B"],
    "audio": {"array": [0] * 16000 * 5, "sampling_rate": 16000},
}

invalid_sample = {
    "timestamps_start": [0.0, 2.0],
    "timestamps_end": [1.0, 1.5],
    "speakers": ["A", "B"],
    "audio": {"array": [0] * 16000 * 5, "sampling_rate": 16000},
}


class TestGenerateRTTM:
    def test_generate_rttm_negative_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_dir = Path(tmpdir)
            file_id = "testfile"
            result = generate_rttm(invalid_sample, rttm_dir, file_id)
            rttm_path = rttm_dir / f"{file_id}.rttm"
            # Expect function to return False and not create RTTM file
            assert result is False
            assert not rttm_path.exists()

    def test_generate_rttm_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rttm_dir = Path(tmpdir)
            file_id = "testfile"
            # Use the correctly formatted sample
            result = generate_rttm(valid_sample, rttm_dir, file_id)
            rttm_path = rttm_dir / f"{file_id}.rttm"
            assert result is True
            assert os.path.exists(rttm_path)
            with open(rttm_path) as f:
                content = f.read()
            assert "SPEAKER" in content
