import pytest
import torch

from dr_sad.pyannet.linearnet import LinearNet


class TestLinearNet:
    """Test cases for the LinearNet class."""

    def test_init_with_hidden_layers(self):
        """Test initialization with hidden layers."""
        input_dim = 64
        hidden_size = 128
        num_layers = 3

        net = LinearNet(input_dim, hidden_size, num_layers)

        assert net.in_features == input_dim
        assert net.hidden_size == hidden_size
        assert net.num_layers == num_layers
        assert net.out_features == hidden_size

    def test_init_no_hidden_layers(self):
        """Test initialization with no hidden layers (identity mapping)."""
        input_dim = 64
        hidden_size = 128
        num_layers = 0

        net = LinearNet(input_dim, hidden_size, num_layers)

        assert net.in_features == input_dim
        assert net.hidden_size == hidden_size
        assert net.num_layers == num_layers
        assert net.out_features == input_dim
        assert isinstance(net.layers, torch.nn.Identity)

    def test_init_negative_layers(self):
        """Test initialization with negative number of layers."""
        input_dim = 64
        hidden_size = 128
        num_layers = -1

        net = LinearNet(input_dim, hidden_size, num_layers)

        assert net.out_features == input_dim
        assert isinstance(net.layers, torch.nn.Identity)

    def test_forward_with_hidden_layers(self):
        """Test forward pass with hidden layers."""
        input_dim = 10
        hidden_size = 20
        num_layers = 2
        batch_size = 5

        net = LinearNet(input_dim, hidden_size, num_layers)
        x = torch.randn(batch_size, input_dim)

        output = net(x)

        assert output.shape == (batch_size, hidden_size)
        assert not torch.isnan(output).any()

    def test_forward_no_hidden_layers(self):
        """Test forward pass with no hidden layers (identity)."""
        input_dim = 10
        hidden_size = 20
        num_layers = 0
        batch_size = 5

        net = LinearNet(input_dim, hidden_size, num_layers)
        x = torch.randn(batch_size, input_dim)

        output = net(x)

        assert output.shape == x.shape
        assert torch.allclose(output, x)

    def test_forward_single_layer(self):
        """Test forward pass with a single hidden layer."""
        input_dim = 8
        hidden_size = 16
        num_layers = 1
        batch_size = 3

        net = LinearNet(input_dim, hidden_size, num_layers)
        x = torch.randn(batch_size, input_dim)

        output = net(x)

        assert output.shape == (batch_size, hidden_size)
        assert not torch.isnan(output).any()

    def test_different_input_sizes(self):
        """Test forward pass with different input tensor sizes."""
        input_dim = 5
        hidden_size = 10
        num_layers = 2

        net = LinearNet(input_dim, hidden_size, num_layers)

        # Test 2D input
        x_2d = torch.randn(4, input_dim)
        output_2d = net(x_2d)
        assert output_2d.shape == (4, hidden_size)

        # Test 3D input
        x_3d = torch.randn(2, 3, input_dim)
        output_3d = net(x_3d)
        assert output_3d.shape == (2, 3, hidden_size)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        input_dim = 4
        hidden_size = 8
        num_layers = 2

        net = LinearNet(input_dim, hidden_size, num_layers)
        x = torch.randn(2, input_dim, requires_grad=True)

        output = net(x)
        loss = output.mean()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that network parameters have gradients
        for param in net.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    @pytest.mark.parametrize(
        ("input_dim", "hidden_size", "num_layers"),
        [
            (16, 32, 1),
            (32, 16, 2),
            (64, 128, 3),
            (10, 10, 1),
        ],
    )
    def test_various_configurations(self, input_dim, hidden_size, num_layers):
        """Test various network configurations."""
        net = LinearNet(input_dim, hidden_size, num_layers)
        x = torch.randn(2, input_dim)

        output = net(x)

        assert output.shape == (2, hidden_size)
        assert not torch.isnan(output).any()
