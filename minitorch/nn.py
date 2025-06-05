from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


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



max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_int = int(dim.item())  # this gets scalar value from tensor
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_int)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        dim_int = int(dim.item())
        is_max = argmax(input, dim_int)
        return is_max * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        dim_int = int(dim.item()) 
        input_max = max_reduce(input, dim_int)
        shifted = input - input_max
        exp = shifted.exp()
        sum_exp = exp.sum(dim_int)
        out = exp / sum_exp
        ctx.save_for_backward(out, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        softmax_out, dim = ctx.saved_values
        dim_int = int(dim.item())
        dot = (grad_output * softmax_out).sum(dim_int)
        return softmax_out * (grad_output - dot), 0

def softmax(input: Tensor, dim: int) -> Tensor:
    return Softmax.apply(input, input._ensure_tensor(dim))


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        dim_int = int(dim.item())
        input_max = max_reduce(input, dim_int)
        shifted = input - input_max
        log_sum_exp = shifted.exp().sum(dim_int).log()
        out = shifted - log_sum_exp
        ctx.save_for_backward(out, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, int]:
        out, dim = ctx.saved_values
        dim_int = int(dim.item())
        softmax_out = out.exp()
        return grad_output - softmax_out * grad_output.sum(dim_int), 0

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    return LogSoftmax.apply(input, input._ensure_tensor(dim))


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
