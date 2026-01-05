import torch

from dr_sad.pyannet.lstmnet import LSTMNet


class TestLSTMNet:
    """Test suite for LSTMNet class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        model = LSTMNet()

        assert model.input_size == 60
        assert model.out_features == 256  # 128 * 2 (bidirectional)
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert model.lstm.input_size == 60
        assert model.lstm.hidden_size == 128
        assert model.lstm.num_layers == 2
        assert model.lstm.bidirectional is True

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = LSTMNet(
            input_size=80,
            hidden_size=64,
            num_layers=3,
            bidirectional=False,
            dropout=0.2,
        )

        assert model.input_size == 80
        assert model.out_features == 64  # 64 * 1 (unidirectional)
        assert model.lstm.input_size == 80
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 3
        assert model.lstm.bidirectional is False

    def test_forward_default_keep_order_false(self):
        """Test forward pass with keep_order=False (default)."""
        model = LSTMNet(input_size=10, hidden_size=32, num_layers=1)

        # Input shape: (batch, feature, frame)
        batch_size, feature_size, frame_size = 2, 10, 20
        x = torch.randn(batch_size, feature_size, frame_size)

        output, (hn, cn) = model.forward(x)

        # Output should have shape (batch, frame, hidden_size * directions)
        expected_output_shape = (batch_size, frame_size, 64)  # 32 * 2 (bidirectional)
        assert output.shape == expected_output_shape

        # Hidden states should have shape (num_layers * directions, batch, hidden_size)
        expected_hidden_shape = (2, batch_size, 32)  # 1 * 2 (bidirectional)
        assert hn.shape == expected_hidden_shape
        assert cn.shape == expected_hidden_shape

    def test_forward_keep_order_true(self):
        """Test forward pass with keep_order=True."""
        model = LSTMNet(input_size=10, hidden_size=32, num_layers=1)

        # Input shape: (batch, frame, feature) - already in correct order
        batch_size, frame_size, feature_size = 2, 20, 10
        x = torch.randn(batch_size, frame_size, feature_size)

        output, (hn, cn) = model.forward(x, keep_order=True)

        # Output should have shape (batch, frame, hidden_size * directions)
        expected_output_shape = (batch_size, frame_size, 64)  # 32 * 2 (bidirectional)
        assert output.shape == expected_output_shape

        # Hidden states should have shape (num_layers * directions, batch, hidden_size)
        expected_hidden_shape = (2, batch_size, 32)  # 1 * 2 (bidirectional)
        assert hn.shape == expected_hidden_shape
        assert cn.shape == expected_hidden_shape

    def test_forward_with_initial_hidden_states(self):
        """Test forward pass with provided initial hidden states."""
        model = LSTMNet(input_size=10, hidden_size=32, num_layers=1)

        batch_size, feature_size, frame_size = 2, 10, 20
        x = torch.randn(batch_size, feature_size, frame_size)

        # Create initial hidden and cell states
        # Shape: (num_layers * directions, batch, hidden_size)
        h0 = torch.randn(2, batch_size, 32)
        c0 = torch.randn(2, batch_size, 32)

        output, (hn, cn) = model.forward(x, hn_cn=(h0, c0))

        # Check output shape
        expected_output_shape = (batch_size, frame_size, 64)
        assert output.shape == expected_output_shape

        # Check that we get hidden states back
        assert hn.shape == (2, batch_size, 32)
        assert cn.shape == (2, batch_size, 32)

    def test_forward_unidirectional(self):
        """Test forward pass with unidirectional LSTM."""
        model = LSTMNet(
            input_size=15, hidden_size=48, num_layers=2, bidirectional=False
        )

        batch_size, feature_size, frame_size = 3, 15, 25
        x = torch.randn(batch_size, feature_size, frame_size)

        output, (hn, cn) = model.forward(x)

        # Output should have shape (batch, frame, hidden_size) - no multiplication by 2
        expected_output_shape = (batch_size, frame_size, 48)
        assert output.shape == expected_output_shape

        # Hidden states should have shape (num_layers, batch, hidden_size)
        expected_hidden_shape = (2, batch_size, 48)
        assert hn.shape == expected_hidden_shape
        assert cn.shape == expected_hidden_shape

    def test_output_deterministic(self):
        """Test that the same input produces the same output."""
        model = LSTMNet(input_size=5, hidden_size=16, num_layers=1)

        # Set model to evaluation mode for deterministic behaviours
        model.eval()

        x = torch.randn(1, 5, 10)

        # Run forward pass twice
        with torch.no_grad():
            output1, _ = model.forward(x)
            output2, _ = model.forward(x)

        # Outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_out_features_property(self):
        """Test that out_features property is correctly calculated."""
        # Bidirectional case
        model_bi = LSTMNet(hidden_size=64, bidirectional=True)
        assert model_bi.out_features == 128  # 64 * 2

        # Unidirectional case
        model_uni = LSTMNet(hidden_size=64, bidirectional=False)
        assert model_uni.out_features == 64  # 64 * 1

    def test_dropout_handling(self):
        """Test dropout parameter handling."""
        # With multiple layers, dropout should be applied
        model_multi = LSTMNet(num_layers=3, dropout=0.3)
        assert model_multi.lstm.dropout == 0.3

        # With single layer, dropout should be 0 even if specified
        model_single = LSTMNet(num_layers=1, dropout=0.3)
        assert model_single.lstm.dropout == 0.0

    def test_gradients_flow(self):
        """Test that gradients flow through the network."""
        model = LSTMNet(input_size=8, hidden_size=16, num_layers=1)

        x = torch.randn(2, 8, 15, requires_grad=True)
        output, _ = model.forward(x)

        # Create a simple loss
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
