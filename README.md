# MiniTorch - Re-implement Torch from scratch
MiniTorch is a minimalist, educational re-implementation of core Pytorch's components, based on the [MiniTorch](https://github.com/minitorch/minitorch) educational template. This project covers automatic differentiation, tensors, neural network modules, and optimized computation via parallelism and CUDA. It is designed to mirror the architecture of PyTorch while building it up from fundamental principles.

## Overview
- Rebuilt PyTorch from the [MiniTorch](https://github.com/minitorch/minitorch) template with fundamental libraries, such as `numpy`, `numba`.
- Implemented autograd system with backpropagation.
- Implemented `Tensor` class supporting broadcasting, strides, views, and permutings.
- Integrated Numba-based parallelism.
- Built and trained models using `MiniTorch` for real-world tasks like MNIST and sentiment classification.

## Core Concepts Implemented

### Autograd Engine (`Scalar` & `Tensor`)
- Built forward/backward computation graphs
- Implemented `Scalar` and `Tensor` classes with `.backward()` support
- Topological sorting of computation graph for efficient gradient propagation
- Applied the chain rule for automatic differentiation
- Gradient accumulation for leaf variables

### `Tensor` Class
- Supported broadcasting and shape inference
- Implemented strides, views, and permutation of tensor dimensions
- Core low-level tensor operations:
  - `tensor_map`: apply functions element-wise
  - `tensor_zip`: apply binary functions element-wise
  - `tensor_reduce`: reduce tensor values along a dimension

### Optimized Computation
- Numba-parallelized backends in `fast_ops.py`
- Matrix multiplication with CPU parallelization
<!-- - (Partially) CUDA tensor operations in `cuda_ops.py`
  - Tensor map, zip, and reduce functions
  - Practice GPU kernels for matrix multiplication and reduction -->

### Deep Learning Modules
- Basic neural network layers: Linear, ReLU, Sigmoid
- Advanced layers: Softmax, Dropout, LogSoftmax
- 1D and 2D Convolution using Numba (`fast_conv.py`)
- 2D Pooling operations with tiling for avgpool and maxpool
<!-- - Trained networks for:
  - Point classification (Simple, Split, XOR)
  - MNIST digit recognition (LeNet-style CNN)
  - Sentiment classification (SST2) -->

## Project Structure
```graphql
minitorch/
├── operators.py         # Core math ops and functional utilities
├── scalar.py            # Scalar autodiff engine
├── tensor.py            # Tensor class with full API
├── tensor_ops.py        # Broadcasting-safe ops
├── tensor_functions.py  # Differentiable ops (forward + backward)
├── fast_ops.py          # Numba-parallelized backend
├── cuda_ops.py          # (Partially complete) GPU backend
├── nn.py                # Neural network layers and modules
├── fast_conv.py         # Numba 1D/2D convolutions
└── datasets.py          # ML training datasets
```

## Example Output
### Training an Image Classifier
Command to run the MNIST multiclass classification example:
```bash
cd MiniTorch/
python ./project/run_mnist_multiclass.py
```
Result:
```bash
Epoch 1 loss 2.3092674515719036 valid acc 1/16
Epoch 1 loss 11.454270220623412 valid acc 2/16
Epoch 1 loss 11.497007777524072 valid acc 2/16
...
Epoch 2 loss 1.2143910410220957 valid acc 12/16
Epoch 2 loss 2.710355374980507 valid acc 14/16
Epoch 2 loss 3.5111799494672202 valid acc 13/16
Epoch 2 loss 2.3375041193903994 valid acc 13/16
Epoch 2 loss 2.1580129264641497 valid acc 13/16
```

