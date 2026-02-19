from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, resample

from dr_sad.data.noise import BaseNoiseBuilder


class ResampleNoiseBuilder(BaseNoiseBuilder):
    def __init__(self, downsample_factor: int):
        """
        Noise builder that adds noise by downsampling and upsampling the signal,
        which can simulate the effect of low-quality audio.

        Args:
            downsample_factor: The factor by which to downsample the signal before
                upsampling it back to the original rate.
        """
        if downsample_factor < 2:
            err_msg = f"downsample_factor must be >= 2, got {downsample_factor}."
            raise ValueError(err_msg)
        self.downsample_factor = downsample_factor

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        len_signal = len(signal)
        new_length = len_signal // self.downsample_factor
        if new_length < 1:
            err_msg = (
                "Signal is too short for the chosen downsample_factor: "
                f"len(signal)={len_signal}, downsample_factor={self.downsample_factor} "
                "would result in a downsampled length of 0."
            )
            raise ValueError(err_msg)
        downsample = resample(signal, new_length)
        return resample(downsample, len_signal)


class VolumeNoiseBuilder(BaseNoiseBuilder):
    def __init__(
        self,
        volume_range: tuple[float, float],
        volume_change_time: int,
        sample_rate: int,
        seed: int | None,
    ):
        """
        Noise builder that adds volume variation to the signal. The volume changes
        linearly between random values in the specified range every volume_change_time
        seconds.

        Args:
            volume_range: A tuple specifying the range (min, max) of volume scaling
                factors.
            volume_change_time: The time interval (in seconds) at which the volume
                scaling factor changes.
            sample_rate: The sample rate of the audio signal, used to determine how many
                samples correspond to volume_change_time.
            seed: Random seed for reproducibility of the volume changes.
                If None, a random seed will be generated.
        """
        self.volume_range = volume_range
        self.volume_change_time = volume_change_time
        self.sample_rate = sample_rate
        seed = seed if seed is not None else np.random.randint(0, 1e6)
        self.rng = np.random.Generator(np.random.PCG64(seed))

        if volume_change_time <= 0:
            err_msg = f"volume_change_time must be positive, got {volume_change_time}."
            raise ValueError(err_msg)

        if not (0 <= self.volume_range[0] <= self.volume_range[1]):
            err_msg = (
                f"volume_range must satisfy 0 <= min <= max, got {self.volume_range}."
            )
            raise ValueError(err_msg)

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        num_seconds = (len(signal) // self.sample_rate) + 1
        volume_update_time = self.volume_change_time

        time_points = np.arange(0, num_seconds, volume_update_time)

        volume_factors = np.power(
            2,
            self.rng.uniform(
                np.log2(self.volume_range[0]),
                np.log2(self.volume_range[1]),
                time_points.shape[0],
            ),
        )

        interp_func = interp1d(
            time_points, volume_factors, kind="linear", fill_value="extrapolate"
        )
        sample_indices = np.linspace(0, len(signal) / self.sample_rate, num=len(signal))

        interpolated_factors = interp_func(sample_indices)
        return signal * interpolated_factors


class ReverbNoiseBuilder(BaseNoiseBuilder):
    def __init__(
        self,
        reverb_sample_filepath: str | Path,
        sample_rate: int = 16000,
    ):
        """
        Noise builder that adds reverberation to the signal by convolving it with a
        reverb impulse response.

        Args:
            reverb_sample_filepath: File path to the audio sample that will be used as
            the reverb impulse response.
            sample_rate: The sample rate of the audio signal, used to resample the
                reverb impulse response to match the signal's sample rate.
        """
        reverb_sound, rsr = sf.read(Path(reverb_sample_filepath))
        reverb_mono = reverb_sound if reverb_sound.ndim == 1 else reverb_sound[:, 0]
        self.reverb_sound_match = resample(
            reverb_mono, reverb_mono.shape[0] * sample_rate // rsr
        )

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        reverb_noise = fftconvolve(signal, self.reverb_sound_match, mode="full")[
            : len(signal)
        ]
        signal_mean = np.mean(np.abs(signal))
        noise_mean = np.mean(np.abs(reverb_noise))

        # Guard against division by zero or extremely small values
        if np.issubdtype(reverb_noise.dtype, np.floating):
            eps = np.finfo(reverb_noise.dtype).eps
        else:
            eps = 1e-12

        reverb_vol = 0.0 if noise_mean < eps else signal_mean / noise_mean

        noisy_signal = signal + reverb_vol * reverb_noise
        return noisy_signal.astype(signal.dtype, copy=False)
