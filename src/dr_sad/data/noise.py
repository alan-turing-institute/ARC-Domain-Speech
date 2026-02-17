from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

__all__ = ("BaseNoiseBuilder", "NoiseBuilder", "generate_noise_kwargs_list")


class BaseNoiseBuilder(ABC):
    @abstractmethod
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Abstract method to add noise to the input signal. This should be implemented by
        subclasses to define how the noise is generated and added to the signal.

        Args:
            signal: A 1D numpy array representing the clean audio signal.

        Returns:
            A 1D numpy array representing the noisy audio signal.
        """
        err_msg = "BaseNoiseBuilder is an abstract class and cannot be instantiated."
        raise NotImplementedError(err_msg)


def _get_root_mean_square(noise_array: np.ndarray) -> float:
    """This computes the root mean square of a 1D numpy array.

    Args:
        noise_array (np.ndarray): The input 1D numpy array.

    Returns:
        rms (float): The root mean square of the input array.
    """
    return float(np.sqrt(np.mean(np.square(noise_array))))


def _signal_to_noise_scale(
    signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> float:
    """This computes the scaling factor to apply to the noise to achieve the
    desired signal-to-noise ratio (SNR) in decibels (dB).

    Args:
        signal (np.ndarray): The clean signal as a 1D numpy array.
        noise (np.ndarray): The noise signal as a 1D numpy array.
        snr_db (float): The desired SNR in decibels.

    Returns:
        scale (float): The scaling factor to apply to the noise.
    """
    rms_signal = _get_root_mean_square(signal)
    rms_noise = _get_root_mean_square(noise)
    snr_linear = 10 ** (snr_db / 20)  # Convert dB to linear scale
    return rms_signal / (snr_linear * rms_noise)


def _clipping_and_typecast(signal: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """This helper function ensures that the signal does not exceed the limits of
    the original data type and casts it back to that type.

    Args:
        signal (np.ndarray): The input signal as a numpy array.
        original_dtype (np.dtype): The original data type of the signal.

    Returns:
        clipped_signal (np.ndarray): The clipped and typecast signal.
    """
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        v_min = info.min
        v_max = info.max
    elif np.issubdtype(original_dtype, np.floating):
        info = np.finfo(original_dtype)
        v_min = -1.0
        v_max = 1.0
    else:
        msg = f"Unsupported data type: {original_dtype}"
        raise TypeError(msg)

    if signal.max() > v_max:
        signal = signal / signal.max() * v_max
    if signal.min() < v_min:
        signal = signal / abs(signal.min()) * abs(v_min)

    return signal.astype(original_dtype)


class NoiseBuilder(BaseNoiseBuilder):
    def __init__(
        self,
        noise_dir: Path | str,
        snr_db: float | tuple[float, float],
        simultaneous: int = 1,
        seed: int | None = None,
        start_choice: bool = True,
        noise_files_list: list[str] | None = None,
    ) -> None:
        """This builds a temporary dataset of noise augmented audio files.

        This class assumes that the noise files are in .wav format and are stored
        in the specified directory.

        Args:
            noise_dir (Path | str): The directory containing noise audio files.]
                This looks for the directory in the following order:
                1) If noise_dir is a Path object, it uses that.
                1) If noise_dir is a valid path, it uses that.
                2) If not, it looks for the directory in the default data/noise
                directory relative to the project root.
            snr_db (float | tuple[float, float]): The desired signal-to-noise ratio
                in decibels (dB), can be a value or a range.
            simultaneous (int, optional): The number of noise files to use
                simultaneously. Defaults to 1, which means only one noise file
                will be used.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
                If seed is None, the seed will be randomly set.
            start_choice (bool, optional): If True, the starting point for cropping
                noise files will be randomly chosen. If False, the start will always
                be at the beginning of the noise file. Defaults to True.
            noise_files_list (list[str] | None, optional): If provided, this is a
                list of specific noise file names (with .wav extension) to use from
                the noise_dir. If None, all .wav files in noise_dir will be used.
                Defaults to None.
        """
        # Validate and set the noise directory
        if isinstance(noise_dir, str):
            # Checks if the string is a valid path
            if Path(noise_dir).exists():
                noise_dir = Path(noise_dir)
            else:
                noise_dir = Path(__file__).parents[3] / "data" / noise_dir
        if not noise_dir.exists():
            msg = f"Noise directory {noise_dir} does not exist."
            raise FileNotFoundError(msg)
        self.noise_dir = noise_dir
        if noise_files_list is None:
            self.noise_files = sorted(noise_dir.rglob("*.wav"))
            if not self.noise_files:
                msg = f"No .wav files found in noise directory {noise_dir}."
                raise ValueError(msg)
        else:
            if not isinstance(noise_files_list, list):
                msg = (  # type: ignore[unreachable]
                    "noise_files_list must be a list of strings."
                )
                raise ValueError(msg)
            if len(noise_files_list) == 0:
                msg = "noise_files_list cannot be an empty list."
                raise ValueError(msg)

            self.noise_files = []
            for file in noise_files_list:
                p = Path(self.noise_dir / file)
                if not p.exists():
                    msg = f"Noise file {p} does not exist."
                    raise FileNotFoundError(msg)
                if p.suffix.lower() != ".wav":
                    msg = f"Noise file {p} is not a .wav file."
                    raise ValueError(msg)
                self.noise_files.append(p)
            self.noise_files = sorted(self.noise_files)

        # Set the SNR
        self.snr_db: float | tuple[float, float] = 0.0
        if isinstance(snr_db, tuple):
            if len(snr_db) == 1:
                print("Warning: snr_db tuple has length 1")  # type: ignore[unreachable]
                self.snr_db = snr_db[0]
            elif len(snr_db) == 2:
                self.snr_db = (min(snr_db), max(snr_db))
            else:
                msg = (  # type: ignore[unreachable]
                    "If snr_db is a tuple, it must have length 2."
                )
                raise ValueError(msg)
        elif isinstance(snr_db, int | float | np.floating):
            self.snr_db = float(snr_db)
        else:
            msg = (  # type: ignore[unreachable]
                "snr_db must be a float or a tuple of two floats."
            )
            raise ValueError(msg)

        # Set the number of simultaneous noise files
        self.simultaneous = simultaneous
        if self.simultaneous < 1:
            msg = "The number of simultaneous noise files must be at least 1."
            raise ValueError(msg)
        if self.simultaneous > len(self.noise_files):
            msg = (
                f"The number of simultaneous noise files ({self.simultaneous}) cannot "
                f"exceed the total number of noise files ({len(self.noise_files)})."
            )
            raise ValueError(msg)

        # Set the random seed
        if seed is None:
            seed = np.random.randint(0, 1_000_000)
        self.random_state = np.random.RandomState(seed)

        # Set the start choice behaviour
        self.start_choice = start_choice

    @property
    def num_noises(self) -> int:
        return len(self.noise_files)

    def get_noise(self) -> np.ndarray:
        """This returns a single noise array. If multiple noise files are to be
        used simultaneously, they are averaged together after being cropped to the
        same length.

        This is assuming the noise files are all mono audio with a sample rate of
        16 kHz and int16 format.

        Returns:
            noise_array (np.ndarray): The noise as a 1D numpy array as float32, but
            with values that go from -32768 to 32767 (i.e., not normalized int16).
        """
        choices = self.random_state.choice(
            self.noise_files, self.simultaneous, replace=False
        )

        sr_noises = [wavfile.read(noise_file) for noise_file in choices]
        for sr, _ in sr_noises:
            if sr != 16_000:
                msg = (
                    f"All noise files must have a sample rate of 16 kHz, but got {sr}."
                )
                raise ValueError(msg)

        noises = [noise.astype(np.float32) for _, noise in sr_noises]

        if self.simultaneous == 1:
            return noises[0]

        min_length = min(noise.shape[0] for noise in noises)

        same_length = []
        for noise in noises:
            if noise.shape[0] > min_length:
                if self.start_choice:
                    start_idx = self.random_state.randint(
                        0, noise.shape[0] - min_length
                    )
                else:
                    start_idx = 0
                same_length.append(noise[start_idx : start_idx + min_length])
            else:
                same_length.append(noise)

        return np.mean(np.stack(same_length), axis=0)

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """This adds noise to the input signal at the desired SNR.

        Args:
            signal (np.ndarray): The clean signal as a 1D numpy array.

        Returns:
            noisy_signal (np.ndarray): The noisy signal as a 1D numpy array, this is
                the same length as the input signal and always cast to float32.
                The noise is scaled and added to the signal to achieve the desired SNR.
        """

        # Check the input signal
        if signal.ndim != 1:
            msg = "Input signal must be a 1D numpy array."
            raise ValueError(msg)

        if not np.issubdtype(signal.dtype, np.floating):
            try:
                signal = signal.astype(np.float32)
            except Exception as err:
                msg = f"Could not convert input signal to float: {err}"
                raise ValueError(msg) from err

        full_noise = self.get_noise()

        # Make signal and noise the same length
        if full_noise.shape[0] < signal.shape[0]:
            # Repeat the noise to make it longer than the signal
            repeats = signal.shape[0] // full_noise.shape[0] + 1
            full_noise = np.tile(full_noise, repeats)

        if full_noise.shape[0] <= signal.shape[0]:
            msg = "Unreachable code: full_noise should be longer than signal."
            raise RuntimeError(msg)

        if self.start_choice:
            start_idx = self.random_state.randint(
                0, full_noise.shape[0] - signal.shape[0]
            )
        else:
            start_idx = 0
        noise_array = full_noise[start_idx : start_idx + signal.shape[0]]

        # Get scaling factor for noise
        if isinstance(self.snr_db, tuple):
            snr_db = self.random_state.uniform(self.snr_db[0], self.snr_db[1])
        else:
            snr_db = self.snr_db

        linear_scale = _signal_to_noise_scale(signal, noise_array, snr_db)
        return signal + linear_scale * noise_array


def generate_noise_kwargs_list(
    noise_kwargs: dict[str, Any] | None, count: int
) -> list[dict[str, Any] | None]:
    """Return a list of `count` noise_kwargs dicts.

    If noise_kwargs is None -> returns [None]*count.
    If noise_kwargs contains a numeric 'seed' -> returns copies with seeds
    incremented by +1 .. +count (preserving other keys).
    If 'seed' is missing or None -> returns shallow copies of the dict repeated.

    Args:
        noise_kwargs (dict | None): The base noise kwargs dictionary.
        count (int): The number of dicts to generate.

    Returns:
        list[dict | None]: A list of noise kwargs dictionaries or None.
    """
    if count < 1:
        msg = "count must be at least 1"
        raise ValueError(msg)
    if noise_kwargs is None:
        return [None] * count
    if not isinstance(noise_kwargs, dict):
        msg = "noise_kwargs must be a dict or None"  # type: ignore[unreachable]
        raise TypeError(msg)
    base_seed = noise_kwargs.get("seed", None)
    if base_seed is not None:
        base_seed = int(base_seed)
        return [{**noise_kwargs, "seed": base_seed + i} for i in range(1, count + 1)]
    # no seed provided: return independent shallow copies
    return [noise_kwargs.copy() for _ in range(count)]
