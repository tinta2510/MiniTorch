from typing import Tuple, Optional

from . import operators, Module, Parameter
from .autodiff import Context
from .fast_ops import FastOps
from .fast_conv import conv1d, conv2d 
from .tensor import Tensor
from .tensor_functions import Max, Softmax, LogSoftmax, rand, tensor, zeros, stack
from .tensor_ops import TensorBackend

BACKEND = TensorBackend(FastOps)

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Cut the image into little tiles -> easy to pick the best number from each tile (pooling!)
    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    new_h = height // kh
    new_w = width // kw    
    
    # Reshape height and width into (new_h, kh) and (new_w, kw)
    reshaped = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)

    # Move kh and kw to the last dimension, then flatten them
    transposed = reshaped.permute(0, 1, 2, 4, 3, 5)  # B, C, new_h, new_w, kh, kw
    tiled = transposed.contiguous().view(batch, channel, new_h, new_w, kh * kw)

    return tiled, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # Tile the input
    tiled, new_h, new_w = tile(input, kernel)  # shape: (B, C, new_h, new_w, KH*KW)

    # Take the average across the last dimension
    return tiled.mean(dim=4).contiguous().view(batch, channel, new_h, new_w)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    tiled, new_h, new_w = tile(input, kernel)  # shape: (B, C, H', W', KH×KW)
    return max(tiled, 4).contiguous().view(batch, channel, new_h, new_w)

def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    return Softmax.apply(input, input._ensure_tensor(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return LogSoftmax.apply(input, input._ensure_tensor(dim))


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore or rate == 0.0:
        return input
    if rate >= 1.0:
        return input * 0.0  # drop everything
    mask = rand(input.shape) > rate
    return input * mask / (1.0 - rate)


def RParam(*shape):
    r = 0.1 * (rand(shape, backend=BACKEND) - 0.5)
    return Parameter(r)

class Linear(Module):
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

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the ReLU activation function element-wise.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tensor with ReLU applied.
        """
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Sigmoid activation function element-wise.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tensor with Sigmoid applied.
        """
        return x.sigmoid()
        
class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        out = conv1d(input, self.weights.value) + self.bias.value
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        out = conv2d(input, self.weights.value) + self.bias.value
        return out

class DropOut(Module):
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, self.rate, ignore=not self.training)

class MaxPool2d(Module):
    def __init__(self, kh: int, kw: int):
        super().__init__()
        self.kh = kh
        self.kw = kw

    def forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, (self.kh, self.kw))
    
class AvgPool2d(Module):
    def __init__(self, kh: int, kw: int):
        super().__init__()
        self.kh = kh
        self.kw = kw

    def forward(self, x: Tensor) -> Tensor:
        return avgpool2d(x, (self.kh, self.kw))
    
class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = RParam(input_size, hidden_size)
        self.W_hh = RParam(hidden_size, hidden_size)
        self.b_ih = RParam(hidden_size)
        self.b_hh = RParam(hidden_size)
        
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
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

class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the RNN over a sequence.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size).
        
        Returns:
            Output tensor of shape (seq_len, batch_size, hidden_size).
        """
        seq_len, batch_size, input_size = x.shape
        if not h0:
            h0 = zeros((batch_size, self.hidden_size))
        outputs = []
        
        h = h0
        # Process each time step
        for t in range(seq_len):
            # Extract the input at time t for all batches 
            # due to MiniTorch not supporting partial indexing
            x_t = zeros((batch_size, input_size))
            for b in range(batch_size):
                for i in range(input_size):
                    x_t[b, i] = x._tensor._storage[x._tensor.index((t, b, i))]
            
            # Process through cell
            h = self.cell(x_t, h)
            outputs.append(h)
        
        return stack(outputs)
    
class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate parameters
        self.W_ir = RParam(input_size, hidden_size)
        self.W_hr = RParam(hidden_size, hidden_size)
        self.b_r = RParam(hidden_size)
        
        # Update gate parameters
        self.W_iz = RParam(input_size, hidden_size)
        self.W_hz = RParam(hidden_size, hidden_size)
        self.b_z = RParam(hidden_size)
        
        # Candidate hidden state parameters
        self.W_ih = RParam(input_size, hidden_size)
        self.W_hh = RParam(hidden_size, hidden_size)
        self.b_h = RParam(hidden_size)
        
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Forward pass for a single GRU cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size).
            h: Hidden state tensor of shape (batch_size, hidden_size).
        
        Returns:
            New hidden state tensor of shape (batch_size, hidden_size).
        """
        # GRU Cell Computation:
        # 1. Compute reset gate: r = sigmoid(W_ir @ x + W_hr @ h + b_r)
        # 2. Compute update gate: z = sigmoid(W_iz @ x + W_hz @ h + b_z)
        # 3. Compute new gate: n = tanh(W_in @ x + W_hn @ (r * h) + b_h)
        # 4. Compute new hidden state: h_new = (1 - z) * n + z * h

        # The update gate z_t controls how much of the previous state is retained.
        # The reset gate r_t controls how much of the previous state influences the candidate.

        # Reset gate
        r = (x @ self.W_ir.value + h @ self.W_hr.value + self.b_r.value).sigmoid()
        
        # Update gate
        z = (x @ self.W_iz.value + h @ self.W_hz.value + self.b_z.value).sigmoid()
        
        # Candidate hidden state
        h_tilde = (x @ self.W_ih.value + (r * h) @ self.W_hh.value + self.b_h.value).tanh()
        
        # New hidden state
        h_new = (tensor(1) - z) * h_tilde + z * h
        
        return h_new        
    
class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the GRU over a sequence.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            h0: Initial hidden state of shape (batch_size, hidden_size)
            
        Returns:
            Output tensor of shape (seq_len, batch_size, hidden_size)
        """
        seq_len, batch_size, input_size = x.shape
        if h0 is None:
            h = zeros((batch_size, self.hidden_size))
        else:
            h = h0
            
        outputs = []
                # Process each time step
        for t in range(seq_len):
            # Extract the input at time t for all batches
            x_t = zeros((batch_size, input_size))
            for b in range(batch_size):
                for i in range(input_size):
                    x_t[b, i] = x._tensor._storage[x._tensor.index((t, b, i))]
            
            # Process through cell
            h = self.cell(x_t, h)
            outputs.append(h)
            
        return stack(outputs)

class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate parameters (i)
        self.W_ii = RParam(input_size, hidden_size)  # input to input gate
        self.W_hi = RParam(hidden_size, hidden_size)  # hidden to input gate
        self.b_i = RParam(hidden_size)  # bias for input gate
        
        # Forget gate parameters (f)
        self.W_if = RParam(input_size, hidden_size) 
        self.W_hf = RParam(hidden_size, hidden_size)
        self.b_f = RParam(hidden_size)  
        
        # External input gate parameters (g)
        self.W_ig = RParam(input_size, hidden_size) 
        self.W_hg = RParam(hidden_size, hidden_size)
        self.b_g = RParam(hidden_size)
        
        # Output gate parameters (o)
        self.W_io = RParam(input_size, hidden_size)
        self.W_ho = RParam(hidden_size, hidden_size)
        self.b_o = RParam(hidden_size)
        
    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for a single LSTM cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (h, c) where both are of shape (batch_size, hidden_size)
            
        Returns:
            Tuple containing:
            - New hidden state h of shape (batch_size, hidden_size)
            - Tuple of (new hidden state, new cell state)
        """
        # LSTM cell forward pass
        # 1. Compute input gate: i = sigmoid(W_ii @ x + W_hi @ h + b_i)
        # 2. Compute forget gate: f = sigmoid(W_if @ x + W_hf @ h + b_f)
        # 3. Compute external input gate: g = tanh(W_ig @ x + W_hg @ h + b_g)
        # 4. Compute output gate: o = sigmoid(W_io @ x + W_ho @ h + b_o)
        # 5. Compute new internal state: c_new = f * c + i * g
        # 6. Compute new hidden state: h_new = o * tanh(c_new)
        
        h, c = state
        
        # Calculate the input gate
        i = (x @ self.W_ii.value + h @ self.W_hi.value + self.b_i.value).sigmoid()
        
        # Forget gate (f)
        f = (x @ self.W_if.value + h @ self.W_hf.value + self.b_f.value).sigmoid()
        
        # External input gate (g)
        g = (x @ self.W_ig.value + h @ self.W_hg.value + self.b_g.value).tanh()
        
        # Output gate (o)
        o = (x @ self.W_io.value + h @ self.W_ho.value + self.b_o.value).sigmoid()
        
        # Internal state (c_new)
        c_new = f*c + i*g
        
        # New hidden state (h_new)
        h_new = o * c_new.tanh()
        
        return h_new, (h_new, c_new)


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for the LSTM over a sequence.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            state: Initial state tuple (h0, c0) where both have shape (batch_size, hidden_size)
            
        Returns:
            Tuple containing:
            - Output tensor of shape (seq_len, batch_size, hidden_size)
            - Tuple of final (hidden_state, cell_state)
        """
        seq_len, batch_size, input_size = x.shape
        
        # Initialize hidden and cell states
        if state is None:
            h = zeros((batch_size, self.hidden_size))
            c = zeros((batch_size, self.hidden_size))
        else:
            h, c = state
            
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            # Extract the input at time t for all batches
            x_t = zeros((batch_size, input_size))
            for b in range(batch_size):
                for i in range(input_size):
                    x_t[b, i] = x._tensor._storage[x._tensor.index((t, b, i))]
            
            # Process through cell
            h, (h, c) = self.cell(x_t, (h, c))
            outputs.append(h)
            
        # Stack outputs along time dimension
        return stack(outputs), (h, c)