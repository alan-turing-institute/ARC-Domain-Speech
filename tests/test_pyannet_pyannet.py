import numpy as np
import torch

from dr_sad.pyannet import PyanNet


class TestPyanNet:
    def test_instantiation_and_basic_props(self):
        p = PyanNet()
        # basic properties
        assert isinstance(p.sample_rate, int)
        assert p.sample_rate == 16000

        num_frames = p.num_frames(16000)
        assert isinstance(num_frames, int)
        assert num_frames > 0

        rf_size = p.receptive_field_size(1)
        assert isinstance(rf_size, int)
        assert rf_size > 0

        center = p.receptive_field_center(0)
        assert isinstance(center, int)

    def test_forward_and_output_shape(self):
        p = PyanNet()
        batch = 2
        samples = 16000
        x = torch.randn(batch, 1, samples)
        # run forward on CPU without grad
        with torch.no_grad():
            out = p(x)

        # Expect shape (batch, frames, classes)
        assert out.ndim == 3
        assert out.shape[0] == batch
        assert out.shape[2] == p.num_classes

    def test_frame_centers_and_start_step(self):
        p = PyanNet()
        start, step = p.frame_centers_start_step()
        assert isinstance(start, float)
        assert isinstance(step, float)

        centers = p.frame_centers(16000, as_numpy=True)
        assert isinstance(centers, np.ndarray)
        assert centers.shape[0] == p.num_frames(16000)
        # centers should be strictly increasing
        assert np.all(np.diff(centers) > 0)

    def test_prepare_annotation_shape(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        assert isinstance(speaker_truth, torch.Tensor)
        # (batch, 1, frames)
        assert speaker_truth.shape[0] == 1
        assert speaker_truth.shape[1] == 1
        assert speaker_truth.shape[2] == p.num_frames(waveforms.shape[-1])
        assert speaker_truth[0, 0, 2] == 1.0
        assert speaker_truth[0, 0, -2] == 0.0

    def test_loss_function(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)
        loss = p.loss_function(speaker_truth, [0], outputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_accuracy_function(self):
        p = PyanNet()
        waveforms = torch.zeros(1, 1, 16000)
        ann = [[(0.0, 0.1)]]
        speaker_truth = p.prepare_annotation(waveforms, ann)
        outputs = torch.rand_like(speaker_truth)
        acc = p.accuracy_function(speaker_truth, [0], outputs, threshold=0.5)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
