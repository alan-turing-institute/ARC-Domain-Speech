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
