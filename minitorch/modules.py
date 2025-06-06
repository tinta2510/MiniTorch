from typing import Optional
import minitorch

BACKEND = minitorch.TensorBackend(minitorch.FastOps)

def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value

class ReLU(minitorch.Module):
    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Apply the ReLU activation function element-wise.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tensor with ReLU applied.
        """
        return x.relu()
    
class Sigmoid(minitorch.Module):
    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Apply the Sigmoid activation function element-wise.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tensor with Sigmoid applied.
        """
        return x.sigmoid()
        
class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        out = minitorch.conv1d(input, self.weights.value) + self.bias.value
        return out

class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        out = minitorch.conv2d(input, self.weights.value) + self.bias.value
        return out

class DropOut(minitorch.Module):
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        return minitorch.dropout(x, self.rate, ignore=not self.training)

class MaxPool2d(minitorch.Module):
    def __init__(self, kh: int, kw: int):
        super().__init__()
        self.kh = kh
        self.kw = kw

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        return minitorch.maxpool2d(x, (self.kh, self.kw))
    
class AvgPool2d(minitorch.Module):
    def __init__(self, kh: int, kw: int):
        super().__init__()
        self.kh = kh
        self.kw = kw

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        return minitorch.avgpool2d(x, (self.kh, self.kw))
    
class RNNCell(minitorch.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = RParam(input_size, hidden_size)
        self.W_hh = RParam(hidden_size, hidden_size)
        self.b_ih = RParam(hidden_size)
        self.b_hh = RParam(hidden_size)
        
    def forward(self, x: minitorch.Tensor, h: minitorch.Tensor) -> minitorch.Tensor:
        """
        Forward pass for a single RNN cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            h: Hidden state tensor of shape (batch_size, hidden_size).
        
        Returns:
            New hidden state tensor of shape (batch_size, hidden_size).
        """
        return (
            x @ self.W_ih.value + h @ self.W_hh.value + self.b_ih.value + self.b_hh.value
        ).relu()

class RNN(minitorch.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: minitorch.Tensor, h0: Optional[minitorch.Tensor] = None) -> minitorch.Tensor:
        """
        Forward pass for the RNN over a sequence.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size).
        
        Returns:
            Output tensor of shape (seq_len, batch_size, hidden_size).
        """
        batch_size = x.shape[1]
        if not h0:
            h0 = minitorch.zeros((batch_size, self.hidden_size))
        outputs = []
        
        for t in range(x.shape[0]):
            h = self.cell(x[t], h)
            outputs.append(h)
        
        return minitorch.stack(outputs)