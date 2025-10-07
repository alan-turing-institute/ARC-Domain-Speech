# Made by ARC - Adapted from pyannote.audio

import torch
import torch.nn as nn
from einops import rearrange


class LSTMNet(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        input_size: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        """Initializes the LSTM network.

        This class wraps around `torch.nn.LSTM` to provide additional functionality
        to match that of the original Pyannote implementation.

        Args:
            input_size (int, optional): The number of expected features in the input
                `x`. Defaults to 60.
            hidden_size (int, optional): The number of features in the hidden state
                `h`. Defaults to 128.
            num_layers (int, optional): Number of recurrent layers.
                Defaults to 2.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM.
                Defaults to True.
            dropout (float, optional): If non-zero, introduces a `Dropout` layer on
                the outputs of each LSTM layer except the last layer, with dropout
                probability equal to `dropout`. Defaults to 0.0. (no dropout).

        Attributes:
            lstm (nn.LSTM): The underlying LSTM module.
            input_size (int): The number of expected features in the input `x`.
            out_features (int): The number of features in the output `y`.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.input_size = input_size
        self.out_features = hidden_size * (2 if bidirectional else 1)

    def forward(
        self,
        x: torch.Tensor,
        hn_cn: tuple[torch.Tensor, torch.Tensor] | None = None,
        keep_order: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Performs a forward pass through the LSTM network.

        This class that wraps around `torch.nn.LSTM` to handle input tensor
        rearrangement based on the `keep_order` flag.
        This rearrangement is done in the original Pyannote implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, feature, frame)
                if `keep_order` is False, or (batch, frame, feature)
                if `keep_order` is True.
            hn_cn (tuple[torch.Tensor, torch.Tensor] | None, optional): A tuple
                containing the hidden state (hn) and cell state (cn) of the LSTM.
                If None, the LSTM will initialize its states. Defaults to None.
            keep_order (bool, optional): If False, the input tensor `x` will be
                rearranged from (batch, feature, frame) to (batch, frame, feature).
                Defaults to False.

        Returns:
            output (torch.Tensor): The output tensor from the LSTM.
            (hn, cn) tuple[torch.Tensor, torch.Tensor]: A tuple of the updated
            hidden state (hn) and cell state (cn), or None if not applicable.
        """
        if not keep_order:
            # (batch, feature, frame) -> (batch, frame, feature)
            x = rearrange(x, "batch feature frame -> batch frame feature")

        return self.lstm(x, hn_cn)  # type: ignore[no-any-return]
