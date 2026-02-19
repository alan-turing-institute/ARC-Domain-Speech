import numpy as np
import pytest
import soundfile as sf

from dr_sad.data.noise import (
    BaseNoiseBuilder,
    NoiseBuilder,
    _clipping_and_typecast,
    _get_root_mean_square,
    _signal_to_noise_scale,
    generate_noise_kwargs_list,
)


class TestBaseNoiseBuilder:
    def test_base_noise_builder_initialization_fails(self):
        """Test that BaseNoiseBuilder initializes correctly."""
        with pytest.raises(
            TypeError,
            match=(
                r"Can't instantiate abstract class BaseNoiseBuilder without an "
                r"implementation for abstract method 'add_noise'"
            ),
        ):
            _ = BaseNoiseBuilder()


class TestHelperTools:
    def test_get_root_mean_square(self):
        array = np.array([1.0, -1.0, 1.0, -1.0])
        rms = _get_root_mean_square(array)
        assert np.isclose(rms, 1.0)

    def test_signal_to_noise_scale(self):
        signal = np.array([2.0, 1.5, -2.0, 1.0])
        noise = np.array([-0.5, 0.5, -0.5, 0.5])
        snr_db = 0.0  # Desired SNR in dB
        scale = _signal_to_noise_scale(signal, noise, snr_db)
        expected_scale = (
            np.sqrt(np.mean(np.square(signal)))
            / np.sqrt(np.mean(np.square(noise)))
            / (10 ** (snr_db / 20))
        )
        assert np.isclose(scale, expected_scale)
        # Test with different SNR and int arrays
        signal = np.array([33124, -32341, 32121, -34421])
        noise = np.array([1124, -1421, 1421, -1212])
        snr_db = 10.0
        scale = _signal_to_noise_scale(signal, noise, snr_db)
        expected_scale = (
            np.sqrt(np.mean(np.square(signal)))
            / np.sqrt(np.mean(np.square(noise)))
            / (10 ** (snr_db / 20))
        )
        assert np.isclose(scale, expected_scale)

    def test_clipping_and_typecast(self):
        # Test with int16
        original_dtype = np.int16
        signal = np.array([40000, -40000, 20000, -20000], dtype=np.int32)
        clipped_signal = _clipping_and_typecast(signal, original_dtype)
        assert clipped_signal.dtype == original_dtype
        assert np.all(clipped_signal <= 32767)
        assert np.all(clipped_signal >= -32768)

        # Test with float32
        original_dtype = np.float32
        signal = np.array([1.5, -1.5, 0.5, -0.5], dtype=np.float64)
        clipped_signal = _clipping_and_typecast(signal, original_dtype)
        assert clipped_signal.dtype == original_dtype
        assert np.all(clipped_signal <= 1.0)
        assert np.all(clipped_signal >= -1.0)

        # Test with float32 to int16 conversion
        original_dtype = np.int16
        signal = np.array([1.1, -1.05, 0.8, -0.4], dtype=np.float32) * 32767
        clipped_signal = _clipping_and_typecast(signal, original_dtype)
        assert clipped_signal.dtype == original_dtype
        assert np.all(clipped_signal <= 32767)
        assert np.all(clipped_signal >= -32768)
        assert clipped_signal.max() == 32767

        # Test with larger negative value
        original_dtype = np.float32
        signal = np.array([1.1, -1.5, 0.8, -0.4], dtype=np.float32)
        clipped_signal = _clipping_and_typecast(signal, original_dtype)
        assert clipped_signal.dtype == original_dtype
        assert np.all(clipped_signal <= 1.0)
        assert np.all(clipped_signal >= -1.0)

        # Test with unsupported dtype
        original_dtype = np.complex64
        signal = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        with pytest.raises(TypeError, match=r"^Unsupported data type:"):
            _clipping_and_typecast(signal, original_dtype)


class TestNoiseBuilder:
    def test_valid_directory(self, noise_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, 0)
        assert builder.num_noises == 3  # There are 3 .wav files
        assert builder.noise_dir == noise_dir
        assert all(f.suffix == ".wav" for f in builder.noise_files)
        assert all(f.exists() for f in builder.noise_files)

    def test_single_directory(self, noise_dataset):
        noise_dir = noise_dataset / "noise" / "noise_folder1"
        builder = NoiseBuilder(noise_dir, 0)
        assert builder.num_noises == 2  # There are 2 .wav files
        assert builder.noise_dir == noise_dir

    def test_err_invalid_directory(self, tmp_path):
        noise_dir = tmp_path / "invalid_noise"
        with pytest.raises(FileNotFoundError):
            NoiseBuilder(noise_dir, 0)

    def test_get_one_noise(self, noise_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, 0, simultaneous=1, seed=42)
        noise = builder.get_noise()
        assert isinstance(noise, np.ndarray)
        assert noise.ndim == 1  # Mono audio
        assert noise.size > 0  # Non-empty array

    def test_get_multiple_noises(self, noise_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, 0, simultaneous=2, seed=42)
        noise = builder.get_noise()
        assert isinstance(noise, np.ndarray)
        assert noise.ndim == 1  # Mono audio
        assert noise.size > 0  # Non-empty array

    def test_err_get_more_noises_than_available(self, noise_dataset):
        noise_dir = noise_dataset / "noise"
        with pytest.raises(ValueError, match=r"^The number of simultaneous noise"):
            NoiseBuilder(noise_dir, 0, simultaneous=5, seed=42)

    def test_add_noise(self, noise_dataset, example_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, snr_db=10, simultaneous=2, seed=42)
        speak_clip_1 = sf.read(example_dataset / "flac" / "TEST_0001.flac")[0]
        assert isinstance(speak_clip_1, np.ndarray)
        assert speak_clip_1.ndim == 1  # Mono audio
        assert np.abs(speak_clip_1).max() <= 1.0  # Ensure it's in float32 range
        noisy_signal = builder.add_noise(speak_clip_1)
        assert isinstance(noisy_signal, np.ndarray)
        assert noisy_signal.ndim == 1  # Mono audio
        assert noisy_signal.size == speak_clip_1.size  # Same length as input signal
        assert noisy_signal.dtype == speak_clip_1.dtype  # Ensure same dtype as input

    def test_add_noise_long_and_short(self, noise_dataset, example_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, snr_db=10, simultaneous=2, seed=42)
        # Test with a longer signal
        speak_1 = sf.read(example_dataset / "flac" / "TEST_0001.flac")[0]
        speak_clip_long = np.tile(speak_1, 5)  # Make it longer by repeating
        noisy_signal_long = builder.add_noise(speak_clip_long)
        assert noisy_signal_long.size == speak_clip_long.size

        # Test with a shorter signal
        speak_2 = sf.read(example_dataset / "flac" / "TEST_0002.flac")[0]
        speak_clip_short = speak_2[:8000]  # Make it shorter by slicing
        noisy_signal_short = builder.add_noise(speak_clip_short)
        assert noisy_signal_short.size == speak_clip_short.size

    def test_add_noise_start_choice(self, noise_dataset, example_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(
            noise_dir, snr_db=10, simultaneous=2, seed=42, start_choice=False
        )
        speak_clip_1 = sf.read(example_dataset / "flac" / "TEST_0001.flac")[0]
        noisy_signal = builder.add_noise(speak_clip_1)
        assert isinstance(noisy_signal, np.ndarray)
        assert noisy_signal.ndim == 1  # Mono audio
        assert noisy_signal.size == speak_clip_1.size  # Same length as input signal
        assert noisy_signal.dtype == speak_clip_1.dtype  # Ensure same dtype as input

    def test_add_noise_snr_range(self, noise_dataset, example_dataset):
        noise_dir = noise_dataset / "noise"
        builder = NoiseBuilder(noise_dir, snr_db=(-5, 5), simultaneous=2, seed=42)
        speak_clip_1 = sf.read(example_dataset / "flac" / "TEST_0001.flac")[0]
        noisy_signal = builder.add_noise(speak_clip_1)
        assert isinstance(noisy_signal, np.ndarray)
        assert noisy_signal.ndim == 1  # Mono audio
        assert noisy_signal.size == speak_clip_1.size  # Same length as input signal
        assert noisy_signal.dtype == speak_clip_1.dtype  # Ensure same dtype as input

    def test_noise_files_list(self, noise_dataset):
        noise_dir = noise_dataset / "noise"
        # Test with specific noise files
        noise_files_list = [
            "noise_folder1/noise-0001.wav",
            "noise_folder2/noise-0003.wav",
        ]
        builder = NoiseBuilder(noise_dir, 0, noise_files_list=noise_files_list)
        assert builder.num_noises == 2
        assert len(builder.noise_files) == 2
        # Check that the correct files are selected
        expected_files = [noise_dir / f for f in noise_files_list]
        assert all(f in builder.noise_files for f in expected_files)

        # Test with empty list (should raise error)
        msg = r"^noise_files_list cannot be an empty list"
        with pytest.raises(ValueError, match=msg):
            NoiseBuilder(noise_dir, 0, noise_files_list=[])

        # Test with non-existent file (should raise error)
        msg = r"^Noise file .* does not exist"
        with pytest.raises(FileNotFoundError, match=msg):
            NoiseBuilder(noise_dir, 0, noise_files_list=["nonexistent.wav"])

        # Test with non-wav file (should raise error)
        with (noise_dir / "invalid_file.txt").open("w") as f:
            f.write("This is not a wav file.")
        msg = r"^Noise file .* is not a .wav file"
        with pytest.raises(ValueError, match=msg):
            NoiseBuilder(noise_dir, 0, noise_files_list=["invalid_file.txt"])

        # Test with invalid type (should raise error)
        msg = r"^noise_files_list must be a list of strings"
        with pytest.raises(ValueError, match=msg):
            NoiseBuilder(
                noise_dir,
                0,
                noise_files_list="not_a_list",  # type: ignore[arg-type]
            )


class TestGenerateNoiseKwargsList:
    def test_none_input(self):
        """Test that None input returns a list of None values."""
        result = generate_noise_kwargs_list(None, 3)
        assert result == [None, None, None]

    def test_with_seed(self):
        """Test that seed is incremented for each dict."""
        noise_kwargs = {
            "noise_dir": "/path/to/noise",
            "snr_db": 10.0,
            "seed": 42,
        }
        result = generate_noise_kwargs_list(noise_kwargs, 3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(d, dict) for d in result)
        assert result[0]["seed"] == 43  # type: ignore[index]
        assert result[1]["seed"] == 44  # type: ignore[index]
        assert result[2]["seed"] == 45  # type: ignore[index]
        # Verify other keys are preserved
        assert result[0]["snr_db"] == 10.0  # type: ignore[index]
        assert result[1]["noise_dir"] == "/path/to/noise"  # type: ignore[index]

    def test_without_seed(self):
        """Test that dicts without seed are shallow copied."""
        noise_kwargs = {
            "noise_dir": "/path/to/noise",
            "snr_db": 10.0,
        }
        result = generate_noise_kwargs_list(noise_kwargs, 2)
        assert len(result) == 2
        assert result[0] == noise_kwargs
        assert result[1] == noise_kwargs
        # Verify they are shallow copies (not the same object)
        assert result[0] is not result[1]
