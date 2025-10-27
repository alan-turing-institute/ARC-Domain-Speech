# The MIT License (MIT)
#
# Copyright (c) 2019 CNRS
#
# Adapted from
# Hervé Bredin - http://herve.niderb.fr
# Updated by ARC

__all__ = (
    "Abs",
    "SincNet",
)

import math
from functools import cached_property

import numpy as np
import torch
import torch.nn as nn
from asteroid_filterbanks import Encoder, ParamSincFB

import dr_sad.pyannet.receptive_field as r_f


def _pool_stride(m: nn.MaxPool1d) -> int:
    # PyTorch allows stride=None -> defaults to kernel_size
    return int(m.stride if m.stride is not None else m.kernel_size)


def _append_1d_spec(
    K: list[int],
    S: list[int],
    P: list[int],
    D: list[int],
    k: int,
    s: int,
    p: int,
    d: int,
):
    K.append(int(k))
    S.append(int(s))
    P.append(int(p))
    D.append(int(d))


def _extract_time_spec(
    seq: nn.Sequential,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Extracts the time-domain specifications (kernel size, stride, padding, dilation)
    for all time-related operations within a given `nn.Sequential` module.

    This function iterates through the layers of the provided `seq` and collects
    the time-domain parameters for convolutional, pooling, and other relevant
    operations in the order they appear. It supports both standard PyTorch layers
    (e.g., `nn.Conv1d`, `nn.MaxPool1d`) and custom layers like the Asteroid Sinc
    Encoder.

    Args:
        seq (nn.Sequential): A sequential container of PyTorch layers.

    Returns:
        tuple[list[int], list[int], list[int], list[int]]:
            - K: List of kernel sizes for each time-related operation.
            - S: List of strides for each time-related operation.
            - P: List of paddings for each time-related operation.
            - D: List of dilations for each time-related operation.
    """
    K: list[int] = []
    S: list[int] = []
    P: list[int] = []
    D: list[int] = []
    for m in seq.modules():
        # Asteroid Sinc Encoder acts like a Conv1d
        if isinstance(m, Encoder):
            fb = m.filterbank
            k = int(fb.kernel_size)  # asteroid docs expose kernel_size
            s = int(fb.stride)  # and stride (hop) on the filterbank
            _append_1d_spec(K, S, P, D, k, s, 0, 1)
        elif isinstance(m, nn.Conv1d):
            _append_1d_spec(
                K, S, P, D, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0]
            )
        elif isinstance(m, nn.MaxPool1d):
            _append_1d_spec(
                K, S, P, D, m.kernel_size, _pool_stride(m), m.padding, m.dilation
            )
        else:
            # Abs / InstanceNorm1d / activations don't affect time geometry
            continue
    return K, S, P, D


class Abs(nn.Module):  # type: ignore[misc]
    """Absolute value layer"""

    def forward(self, x):
        return x.abs()


class SincNet(nn.Module):  # type: ignore[misc]
    def __init__(self, sample_rate: int = 16000, stride: int = 10):
        """SincNet feature extractor.
        This is adapted from pyannote.audio some attempt has been made to restructure
        and document their code.

        Args:
            sample_rate (int): Sample rate of the input waveform.
                Only 16kHz was supported. (default: 16000)
            stride (int): Stride (in samples) of the first convolutional layer.
                (default: 10) The default was 1 but then changed in the child class.

        Attributes:
            sample_rate (int): Sample rate of the input waveform.
            stride (int): Stride (in samples) of the first convolutional layer.
            self.block0 (nn.Sequential): First block of the network. Takes raw audio
                waveform as input and outputs 80-dimensional features.
                Consists of:
                - waveform norm: InstanceNorm1d
                - sinc encoder: ParamSincFB
                - |·|: Absolute value layer
                - pool: MaxPool1d
                - norm: InstanceNorm1d
                - lrelu: LeakyReLU
            self.block1 (nn.Sequential): Second block of the network. Takes
                80-dimensional features as input and outputs 60-dimensional features.
                Consists of:
                - conv: Conv1d
                - pool: MaxPool1d
                - norm: InstanceNorm1d
                - lrelu: LeakyReLU
            self.block2 (nn.Sequential): Third block of the network. Takes
                60-dimensional features as input and outputs 60-dimensional features.
                Consists of:
                - conv: Conv1d
                - pool: MaxPool1d
                - norm: InstanceNorm1d
                - lrelu: LeakyReLU
            self.features (nn.Sequential): All three blocks combined in one container.
                This takes raw audio waveform as input and outputs 60-dimensional
                features.
        """
        super().__init__()
        if sample_rate != 16000:
            msg = "SincNet only supports 16kHz audio for now."
            raise NotImplementedError(msg)

        # Store parameters as instance attributes
        self.sample_rate = sample_rate
        self.stride = stride

        self.out_features = 60

        # block 0: waveform norm → sinc encoder → |·| → pool → norm → lrelu
        self.block0 = nn.Sequential(
            nn.InstanceNorm1d(1, affine=True),
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            ),
            Abs(),
            nn.MaxPool1d(3, stride=3),
            nn.InstanceNorm1d(80, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # block 1: conv → pool → norm → lrelu
        self.block1 = nn.Sequential(
            nn.Conv1d(80, 60, kernel_size=5, stride=1),
            nn.MaxPool1d(3, stride=3),
            nn.InstanceNorm1d(60, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # block 2: conv → pool → norm → lrelu
        self.block2 = nn.Sequential(
            nn.Conv1d(60, self.out_features, kernel_size=5, stride=1),
            nn.MaxPool1d(3, stride=3),
            nn.InstanceNorm1d(self.out_features, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # or, if you prefer one container:
        self.features = nn.Sequential(self.block0, self.block1, self.block2)

        self._K, self._S, self._P, self._D = _extract_time_spec(self.features)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # (B, 1, T) → (B, 60, L)
        return self.features(waveforms)

    @property
    def frame_hop_samples(self) -> int:
        return math.prod(self._S)

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.frame_hop_samples

    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Args:
            num_samples (int): Number of input samples.

        Returns:
            num_frames (int): Number of output frames.
        """

        return r_f.multi_conv_num_frames(
            num_samples,
            kernel_size=self._K,
            stride=self._S,
            padding=self._P,
            dilation=self._D,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Args:
            num_frames (int, optional): Number of frames in the output signal

        Returns:
            receptive_field_size (int): Receptive field size.
        """

        return r_f.multi_conv_receptive_field_size(
            num_frames,
            kernel_size=self._K,
            stride=self._S,
            padding=self._P,
            dilation=self._D,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Args:
            frame (int, optional): Frame index

        Returns:
            receptive_field_center (int): Index of receptive field center.
        """

        return r_f.multi_conv_receptive_field_center(
            frame,
            kernel_size=self._K,
            stride=self._S,
            padding=self._P,
            dilation=self._D,
        )

    @cached_property
    def frame_center_start_step(self) -> tuple[int, int]:
        """Compute step (in samples) between the start of two consecutive frames

        Returns:
            start (int): Start (in samples) of the first frame
            step (int): Step (in samples) between the start of two consecutive frames
        """
        return r_f.multi_conv_start_step(
            kernel_size=self._K, stride=self._S, padding=self._P, dilation=self._D
        )

    def frame_centers(
        self, num_samples: int, as_numpy: bool = False
    ) -> list[int] | np.ndarray:
        """Compute centers (in samples) of all output frames

        Args:
            num_samples (int): Number of input samples

        Returns:
            centers (list[int]): List of input-sample indices of receptive field centers
        """
        return r_f.frame_centers_samples(
            num_samples,
            kernel_size=self._K,
            stride=self._S,
            padding=self._P,
            dilation=self._D,
            as_numpy=as_numpy,
        )
