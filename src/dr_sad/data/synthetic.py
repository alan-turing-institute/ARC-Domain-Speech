from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import resample


class BaseNoiseBuilder(ABC):
    @abstractmethod
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        pass


class IdentityNoiseBuilder(BaseNoiseBuilder):
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        return signal


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
