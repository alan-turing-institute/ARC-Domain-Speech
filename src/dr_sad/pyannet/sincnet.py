# The MIT License (MIT)
#
# Copyright (c) 2019 CNRS
#
# Adapted from
# Hervé Bredin - http://herve.niderb.fr
# Updated by ARC

__all__ = ("SincNet",)

import math

import torch
import torch.nn as nn
from asteroid_filterbanks import Encoder, ParamSincFB

import dr_sad.pyannet.receptive_field as r_f


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

        self._K = [251, 3, 5, 3, 5, 3]
        self._S = [self.stride, 3, 1, 3, 1, 3]
        self._P = [0, 0, 0, 0, 0, 0]
        self._D = [1, 1, 1, 1, 1, 1]

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
