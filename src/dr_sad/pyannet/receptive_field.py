# The MIT License (MIT)
#
# Copyright (c) 2023 CNRS
#
# Adapted from
# Hervé Bredin - http://herve.niderb.fr
# Updated by ARC

__all__ = (
    "conv1d_num_frames",
    "conv1d_receptive_field_center",
    "conv1d_receptive_field_size",
    "multi_conv_num_frames",
    "multi_conv_receptive_field_center",
    "multi_conv_receptive_field_size",
)

from collections.abc import Sequence



def _check_int(name: str, x: int) -> None:
    """Check that x is an integer, print meaningful error message otherwise"""
    if not isinstance(x, int):
        msg = f"{name}: expected int, got {type(x)}"  # type: ignore[unreachable]
        raise TypeError(msg)


def _check_lists(
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> None:
    """Check that kernel_size, stride, padding, dilation are all provided
    and have equal length, return meaningful error message otherwise.

    Args:
        kernel_size (Sequence[int]): List of kernel sizes
        stride (Sequence[int]): List of strides
        padding (Sequence[int]): List of paddings
        dilation (Sequence[int]): List of dilations
    """
    if any(x is None for x in (kernel_size, stride, padding, dilation)):
        msg = "kernel_size, stride, padding, dilation must all be provided"
        raise ValueError(msg)
    n = len(kernel_size)
    if not (len(stride) == len(padding) == len(dilation) == n):
        msg = "kernel_size, stride, padding, dilation must have equal length"
        raise ValueError(msg)


def conv1d_num_frames(
    num_samples: int,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Compute expected number of frames after 1D convolution

    Args:
        num_samples (int): Number of samples in the input signal
        kernel_size (int): Kernel size (default: 5)
        stride (int): Stride (default: 1)
        padding (int): Padding (default: 0)
        dilation (int): Dilation (default: 1)

    Returns:
        num_frames (int): Number of frames in the output signal

    Source:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
    """
    for name, input in [
        ("num_samples", num_samples),
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ]:
        _check_int(name, input)

    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def multi_conv_num_frames(
    num_samples: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> int:
    """Compute expected number of frames after multiple 1D convolutions.

    This is done by iteratively applying `conv1d_num_frames`.

    Args:
        num_samples (int): Number of samples in the input signal
        kernel_size (Sequence[int]): List of kernel sizes
        stride (Sequence[int]): List of strides
        padding (Sequence[int]): List of paddings
        dilation (Sequence[int]): List of dilations

    Returns:
        num_frames (int): Number of frames in the output signal
    """
    _check_lists(kernel_size, stride, padding, dilation)
    num_frames = num_samples
    for k, s, p, d in zip(kernel_size, stride, padding, dilation, strict=True):
        num_frames = conv1d_num_frames(
            num_frames, kernel_size=k, stride=s, padding=p, dilation=d
        )

    return num_frames


def conv1d_receptive_field_size(
    num_frames: int = 1,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    """Compute size of receptive field, this is the number of input samples
    that affect a given output frame from the forward pass of a 1D convolution.

    Args:
        num_frames (int): Number of frames in the output signal (default: 1)
        kernel_size (int): Kernel size (default: 5)
        stride (int): Stride (default: 1)
        padding (int): Padding (default: 0)
        dilation (int): Dilation (default: 1)

    Returns:
        size (int): Receptive field size
    """
    for name, input in [
        ("num_frames", num_frames),
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ]:
        _check_int(name, input)

    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return effective_kernel_size + (num_frames - 1) * stride - 2 * padding


def multi_conv_receptive_field_size(
    num_frames: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> int:
    """Compute size of receptive field after multiple 1D convolutions.

    This is done by iteratively applying `conv1d_receptive_field_size`.

    Args:
        num_frames (int): Number of frames in the output signal
        kernel_size (Sequence[int]): List of kernel sizes
        stride (Sequence[int]): List of strides
        padding (Sequence[int]): List of paddings
        dilation (Sequence[int]): List of dilations

    Returns:
        receptive_field_size (int): Receptive field size
    """
    _check_lists(kernel_size, stride, padding, dilation)
    receptive_field_size = num_frames

    for k, s, p, d in reversed(
        list(zip(kernel_size, stride, padding, dilation, strict=True))
    ):
        receptive_field_size = conv1d_receptive_field_size(
            num_frames=receptive_field_size,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )
    return receptive_field_size


def conv1d_receptive_field_center(
    frame: int = 0,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Compute center of receptive field

    Args:
        frame (int): Frame index (default: 0)
        kernel_size (int): Kernel size (default: 5)
        stride (int): Stride (default: 1)
        padding (int): Padding (default: 0)
        dilation (int): Dilation (default: 1)

    Returns:
        center (int): Index of receptive field center
    """
    for name, input in [
        ("frame", frame),
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ]:
        _check_int(name, input)

    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return frame * stride + (effective_kernel_size - 1) // 2 - padding


def multi_conv_receptive_field_center(
    frame: int,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> int:
    """Compute center of receptive field after multiple 1D convolutions.

    This is done by iteratively applying `conv1d_receptive_field_center`.

    Args:
        frame (int): Frame index
        kernel_size (Sequence[int]): List of kernel sizes
        stride (Sequence[int]): List of strides
        padding (Sequence[int]): List of paddings
        dilation (Sequence[int]): List of dilations

    Returns:
        receptive_field_center (int): Index of receptive field center
    """
    _check_lists(kernel_size, stride, padding, dilation)
    receptive_field_center = frame
    for k, s, p, d in reversed(
        list(zip(kernel_size, stride, padding, dilation, strict=True))
    ):
        receptive_field_center = conv1d_receptive_field_center(
            frame=receptive_field_center,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
        )

    return receptive_field_center
