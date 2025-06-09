# MiniTorch - Re-implement Torch from scratch
MiniTorch is a minimalist, educational re-implementation of core Pytorch's components, based on the [MiniTorch](https://github.com/minitorch/minitorch) educational template. This project covers automatic differentiation, tensors, neural network modules, and optimized computation via parallelism and CUDA. It is designed to mirror the architecture of PyTorch while building it up from fundamental principles.

## Overview
- Rebuilt PyTorch from the [MiniTorch](https://github.com/minitorch/minitorch) template with fundamental libraries, such as `numpy`, `numba`.
- Implemented autograd system with backpropagation.
- Implemented `Tensor` class supporting broadcasting, strides, views, and permutings.
- Integrated Numba-based parallelism.
- Built and trained models using `MiniTorch` for real-world tasks like MNIST and sentiment classification.
- Implemented RNN layer for sequence modeling and temporal data

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
- Sequence modeling layers: Implemented RNN, LSTM, GRU layers
  
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

## Example of Using MiniTorch
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

### Training a Sentiment Classifier
Use the built  RNN, LSTM, or GRU layers for sentiment analysis on the IMDB dataset. Source Code: "./project/minitorch-imdb-sentiment-analysis.py"
Result of RNN:
```bash
  [Batch 100] Loss: 0.6793
  [Batch 200] Loss: 0.6830
  [Batch 300] Loss: 0.7109
  [Batch 400] Loss: 0.6887
  [Batch 500] Loss: 0.6941
  [Batch 600] Loss: 0.6888
  [Batch 700] Loss: 0.6950
[RNN] Epoch 1: Train Acc: 0.5036, Test Acc: 0.5024
...
  [Batch 500] Loss: 0.6331
  [Batch 600] Loss: 0.7361
  [Batch 700] Loss: 0.7142
[RNN] Epoch 5: Train Acc: 0.5620, Test Acc: 0.5043
```

Result of LSTM:
```bash
  [Batch 100] Loss: 0.6819
  [Batch 200] Loss: 0.6940
  [Batch 300] Loss: 0.6937
  [Batch 400] Loss: 0.7026
  [Batch 500] Loss: 0.6929
  [Batch 600] Loss: 0.6985
  [Batch 700] Loss: 0.6738
[LSTM] Epoch 1: Train Acc: 0.5022, Test Acc: 0.5108
...
  [Batch 500] Loss: 0.5313
  [Batch 600] Loss: 0.5728
  [Batch 700] Loss: 0.4910
[LSTM] Epoch 5: Train Acc: 0.7407, Test Acc: 0.6849
```
Result of GRU:
```bash
  [Batch 100] Loss: 0.6955
  [Batch 200] Loss: 0.6759
  [Batch 300] Loss: 0.6890
  [Batch 400] Loss: 0.6775
  [Batch 500] Loss: 0.7041
  [Batch 600] Loss: 0.7062
  [Batch 700] Loss: 0.6867
[GRU] Epoch 1: Train Acc: 0.5058, Test Acc: 0.5111
...
  [Batch 500] Loss: 0.3454
  [Batch 600] Loss: 0.4701
  [Batch 700] Loss: 0.5790
[GRU] Epoch 5: Train Acc: 0.7718, Test Acc: 0.6997
```