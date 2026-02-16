from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import convolve, resample


class BaseNoiseBuilder(ABC):
    @abstractmethod
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        pass


class ResampleNoiseBuilder(BaseNoiseBuilder):
    def __init__(self, downsample_factor: int):
        self.downsample_factor = downsample_factor

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        len_signal = len(signal)
        downsample = resample(signal, len_signal // self.downsample_factor)
        return resample(downsample, len_signal)


class VolumeNoiseBuilder(BaseNoiseBuilder):
    def __init__(self, volume_range: tuple[float, float], n_volume_changes: int = 10):
        self.volume_range = volume_range
        self.n_volume_changes = n_volume_changes

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        volume_factor_shape = len(signal) // self.n_volume_changes
        volume_factor = np.random.uniform(
            *self.volume_range, size=signal.shape[0] // volume_factor_shape
        )
        volume_factor = np.repeat(volume_factor, volume_factor_shape + 1)[: len(signal)]
        return signal * volume_factor


class ReverbNoiseBuilder(BaseNoiseBuilder):
    def __init__(self, ratio: float, reverb_volume: float = 0.3):
        self.ratio = ratio  # a higher power makes the reverb die out more quickly
        self.reverb_vol = reverb_volume

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        # Create impulse response from the signal itself
        verb = np.abs(signal)
        if np.max(verb) > 0:
            verb = verb / np.max(verb)  # normalize
        verb = np.power(verb, self.ratio)  # shape the decay

        # Make the impulse response shorter to avoid excessive reverb
        max_reverb_len = len(signal) // 4  # limit reverb length
        verb = verb[:max_reverb_len]

        convolved = convolve(signal, verb, mode="same")

        # Mix: mostly dry signal with a little wet reverb
        dry_mix = 1.0 - self.reverb_vol
        result = (signal * dry_mix) + (convolved * self.reverb_vol)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result
