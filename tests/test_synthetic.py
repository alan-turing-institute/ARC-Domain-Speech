"""Tests for dr_sad.data.synthetic module."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dr_sad.data.synthetic import (
    ResampleNoiseBuilder,
    ReverbNoiseBuilder,
    VolumeNoiseBuilder,
)


class TestResampleNoiseBuilder:
    """Test the ResampleNoiseBuilder class."""

    def test_resample_noise_builder_initialization(self):
        """Test that ResampleNoiseBuilder initializes correctly."""
        downsample_factor = 4
        builder = ResampleNoiseBuilder(downsample_factor=downsample_factor)
        assert builder.downsample_factor == downsample_factor

    def test_add_noise_changes_signal(self):
        """Test that add_noise produces a different signal."""
        # Create a simple test signal
        signal = np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, 16000)
        )  # 1 second 440Hz tone
        builder = ResampleNoiseBuilder(downsample_factor=4)

        noisy_signal = builder.add_noise(signal)

        # Check that output has same length
        assert len(noisy_signal) == len(signal)
        # Check that signal was actually modified (downsampling should change it)
        assert not np.array_equal(signal, noisy_signal)

    def test_add_noise_preserves_shape(self):
        """Test that add_noise preserves signal shape."""
        signal = np.random.randn(8000)
        builder = ResampleNoiseBuilder(downsample_factor=2)

        noisy_signal = builder.add_noise(signal)

        assert noisy_signal.shape == signal.shape
        assert isinstance(noisy_signal, np.ndarray)

    def test_add_noise_diminishes_high_frequencies(self):
        """Test that resampling diminishes high-frequency content."""
        # Create a signal with high-frequency content
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        # Mix low (440Hz) and high (4000Hz) frequency components
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)

        builder = ResampleNoiseBuilder(downsample_factor=4)
        noisy_signal = builder.add_noise(signal)

        # Compute FFT to analyze frequency content
        fft_original = np.abs(np.fft.fft(signal))
        fft_noisy = np.abs(np.fft.fft(noisy_signal))

        # High frequencies (around 4000Hz) should be diminished
        # frequency after downsampling is sample_rate / (2 * downsample_factor) = 2000Hz
        # So frequencies above 2000Hz should be attenuated
        high_freq_idx = len(signal) // 4  # Index corresponding to ~4000Hz
        low_freq_idx = len(signal) // 18  # Index corresponding to ~440Hz

        # High frequency energy should be reduced more than low frequency energy
        high_freq_ratio = fft_noisy[high_freq_idx] / (
            fft_original[high_freq_idx] + 1e-10
        )
        low_freq_ratio = fft_noisy[low_freq_idx] / (fft_original[low_freq_idx] + 1e-10)

        assert high_freq_ratio < low_freq_ratio


class TestVolumeNoiseBuilder:
    """Test the VolumeNoiseBuilder class."""

    def test_volume_noise_builder_initialization(self):
        """Test that VolumeNoiseBuilder initializes correctly."""
        volume_range = (0.5, 2.0)
        volume_change_time = 2
        sample_rate = 16000
        seed = 42

        builder = VolumeNoiseBuilder(
            volume_range=volume_range,
            volume_change_time=volume_change_time,
            sample_rate=sample_rate,
            seed=seed,
        )

        assert builder.volume_range == volume_range
        assert builder.volume_change_time == volume_change_time
        assert builder.sample_rate == sample_rate

    def test_add_noise_with_seed_reproducible(self):
        """Test that add_noise is reproducible with same seed."""
        signal = np.ones(16000)  # 1 second of constant signal
        builder1 = VolumeNoiseBuilder(
            volume_range=(0.5, 2.0), volume_change_time=1, sample_rate=16000, seed=42
        )
        builder2 = VolumeNoiseBuilder(
            volume_range=(0.5, 2.0), volume_change_time=1, sample_rate=16000, seed=42
        )

        result1 = builder1.add_noise(signal)
        result2 = builder2.add_noise(signal)

        np.testing.assert_array_equal(result1, result2)

    def test_add_noise_changes_signal(self):
        """Test that add_noise produces volume variations."""
        # Create a constant signal to clearly see volume changes
        signal = np.ones(32000)  # 2 seconds at 16kHz
        builder = VolumeNoiseBuilder(
            volume_range=(0.1, 3.0), volume_change_time=1, sample_rate=16000, seed=123
        )

        noisy_signal = builder.add_noise(signal)

        # Check that output has same length
        assert len(noisy_signal) == len(signal)
        # Check that signal was actually modified
        assert not np.array_equal(signal, noisy_signal)
        # Volume should vary, so variance should be > 0
        assert np.var(noisy_signal) > 0


class TestReverbNoiseBuilder:
    """Test the ReverbNoiseBuilder class."""

    @pytest.fixture()  # type: ignore[misc]
    def temp_reverb_file(self, tmp_path: Path) -> Path:
        """Create a temporary reverb impulse response file."""
        # Generate a simple impulse response (exponential decay)
        sample_rate = 16000
        duration = 0.5  # 500ms reverb
        t = np.linspace(0, duration, int(sample_rate * duration))
        reverb_signal = np.exp(-5 * t) * np.random.randn(len(t)) * 0.1

        # Convert to stereo format as expected by the class
        reverb_stereo = np.column_stack([reverb_signal, reverb_signal])

        reverb_path = tmp_path / "test_reverb.wav"
        sf.write(reverb_path, reverb_stereo, sample_rate)
        return reverb_path

    def test_reverb_noise_builder_initialization(self, temp_reverb_file):
        """Test that ReverbNoiseBuilder initializes correctly."""
        sample_rate = 16000
        builder = ReverbNoiseBuilder(
            reverb_sample_filepath=temp_reverb_file, sample_rate=sample_rate
        )

        assert hasattr(builder, "reverb_sound_match")
        assert isinstance(builder.reverb_sound_match, np.ndarray)

    def test_add_noise_changes_signal(self, temp_reverb_file):
        """Test that add_noise adds reverberation to the signal."""
        # Create a short impulse signal
        signal = np.zeros(16000)
        signal[1000] = 1.0  # Sharp impulse at sample 1000

        builder = ReverbNoiseBuilder(
            reverb_sample_filepath=temp_reverb_file, sample_rate=16000
        )

        reverb_signal = builder.add_noise(signal)

        # Check that output has same length
        assert len(reverb_signal) == len(signal)
        # Check that signal was actually modified
        assert not np.array_equal(signal, reverb_signal)
        # After adding reverb, the signal should have energy at more time points
        assert np.sum(np.abs(reverb_signal) > 1e-6) > np.sum(np.abs(signal) > 1e-6)

    def test_add_noise_preserves_shape_and_type(self, temp_reverb_file):
        """Test that add_noise preserves signal properties."""
        signal = np.random.randn(8000) * 0.1
        builder = ReverbNoiseBuilder(
            reverb_sample_filepath=temp_reverb_file, sample_rate=16000
        )

        reverb_signal = builder.add_noise(signal)

        assert reverb_signal.shape == signal.shape
        assert isinstance(reverb_signal, np.ndarray)
        assert reverb_signal.dtype == signal.dtype
