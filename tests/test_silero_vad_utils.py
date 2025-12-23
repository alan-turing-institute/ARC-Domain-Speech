import torch

from dr_sad.models.silero_vad_utils import get_probs


class DummyModel:
    def __call__(self, _chunk, _sr):
        # Always return a fixed probability
        return torch.tensor(0.5)

    def reset_states(self):
        pass


def test_get_probs_length():
    sample_rate = 16000
    window_size_samples = 512
    duration_sec = 2
    num_samples = sample_rate * duration_sec
    waveform = torch.randn(num_samples)

    model = DummyModel()

    probs = get_probs(model, waveform, window_size_samples, sample_rate)

    expected_num_windows = (
        num_samples + window_size_samples - 1
    ) // window_size_samples
    assert len(probs) == expected_num_windows
    assert all(isinstance(p, float) for p in probs)
