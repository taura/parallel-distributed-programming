# Training a small GPT to sort digits in C++

## Problem

The task is to train a small GPT to sort digit sequences: for example, to turn `0 0 2 1 0 1` into `0 0 0 1 1 2`.
This project uses the sorting task from Karpathy's minGPT demo to study the cost of training and running a GPT model.
The code is a serial C++ port of minGPT, an educational implementation of GPT.
Later assignments will parallelize this code and compare its runtime with the serial version.

GPT predicts the next token from the tokens it has already read.
In a text model, a token might be a word fragment. Here, each digit is one token.
The model assigns a probability to each possible next digit, chooses one, and appends it to the input.
Repeating this process produces the answer.

A conventional sort supplies the training answers. GPT learns weights that let it predict those answers one digit at a time.
This makes it possible to exercise GPT's matrix operations on a task with easily checked results.
The program generates its own examples and needs no text dataset or pretrained weights.

## How the baseline code works

### Training and checking the answers

The data generator uses a fixed random seed to create digit sequences.
It sorts a copy of each sequence, joins the problem and answer, and forms an input and a target shifted by one position:

```text
Problem:  0 0 2 1 0 1
Answer:   0 0 0 1 1 2
Input:    0 0 2 1 0 1 0 0 0 1 1
Targets: -1 -1 -1 -1 -1 0 0 0 1 1 2
```

Each input position predicts the token in the corresponding target position.
A target of `-1` means that the prediction is not scored.
The first scored prediction comes after the model has read the whole problem.
For a problem of length $n$, this gives $2n-1$ input tokens and $n$ scored predictions.

Several examples are processed together as a batch.
The cross-entropy loss penalizes low probabilities for the correct digits. Lower loss means better predictions on the scored positions.
Backpropagation computes how each weight affects this loss.
The trainer clips large gradients, then uses the AdamW optimizer to update the weights.

Training and test-loss calculation use the correct earlier answer digits as input.
To check generation, the model starts with only the problem and produces its own answer.
The program counts how many generated answers match the sorted sequence in every position.
A fixed hash assigns each problem to either the training or test set, so the same problem cannot appear in both.

### Model and implementation

The model first looks up a numerical vector for each digit and adds a vector for its position.
These vectors pass through a series of Transformer blocks.
Within each block, self-attention mixes information from the current and earlier positions.
A mask hides later positions so the model cannot read the digit it is supposed to predict.
A small feed-forward network then transforms each position's vector.
It uses GELU, a nonlinear function, between two fully connected layers; the intermediate vector is four times as wide as the input.

LayerNorm adjusts the scale of the features before each of these two operations.
Residual connections add the result of each operation to its input.
After the last block, another LayerNorm and an output layer produce a score for each possible digit.

| File | Role |
|---|---|
| `model.h/.cpp` | GPT layers, initialization, forward pass, and autoregressive generation |
| `trainer.h/.cpp` | Training loop, gradient clipping, and AdamW |
| `tensor.h/.cpp` | Matrix multiplication, elementwise operations, normalization, and automatic differentiation |
| `code.cpp` | Data generation, argument parsing, evaluation, self-tests, and output |

All numerical operations use serial loops over flat arrays of `double` values.
The code also implements automatic differentiation, which records the calculations and propagates gradients through them.
It requires only the C++17 standard library and uses no OpenMP, threads, GPU kernels, or SIMD intrinsics.

## Where is parallelism

Matrix multiplication is a starting point: each output entry is a dot product that can be computed independently.
These products appear in the fully connected layers, attention, and backpropagation.
The original minGPT calls PyTorch's `nn.Linear` and matrix multiplication operations, which can use CPU threads, SIMD, or GPU kernels.
In this port, the corresponding loops are in `Tensor::matmul()`.

On a multicore CPU, output rows or rectangular tiles of a matrix can be assigned to different threads.
On a GPU, many threads can cooperate on each tile and reuse its input data.
Separate examples in a batch provide more work to distribute.
Normalization and softmax, which turns scores into probabilities, can also run on separate rows, with a reduction inside each row.

Backward computation needs care because many predictions contribute to the same weight's gradient.
The current `add_grad()` performs `+=`; two threads updating the same entry would race.
One approach is to accumulate gradients in per-thread buffers and combine them afterward.
Repeated digits create the same issue when gradients are added to their shared embedding vector.

The amount of work depends on the batch size $B$, input length $T$, and vector width $C$.
Each block has roughly $O(BTC^2)$ work in its linear transformations and $O(BT^2C)$ in attention.
With other sizes fixed, doubling the input length roughly quadruples the attention work.
Doubling the vector width roughly quadruples the linear-transformation work.
For a small model, thread and kernel launch overhead may outweigh the savings.
For larger inputs, storing attention matrices and moving intermediate arrays can become expensive.
The C++ `transpose` copies data, whereas PyTorch usually changes only the array's metadata, so data movement also matters in a comparison.

There are sequential dependencies across layers and training updates.
Within a training layer, all positions can be computed together because the correct input sequence is already available.
Generation must wait for each new digit before predicting the next one.
Parallelism is still available inside that prediction and across separate problems.
Runtime comparisons need the same inputs, model dimensions, update count, measurement scope, and numerical precision.

## Reference implementation

- [karpathy/minGPT](https://github.com/karpathy/minGPT/tree/37baab71b9abea1b76ab957409a1cc2fbfba8a26): `mingpt/model.py`, `mingpt/trainer.py`, and `demo.ipynb`
- Source commit: `37baab71b9abea1b76ab957409a1cc2fbfba8a26`
- Original implementation license: MIT (included in `LICENSE.minGPT`)
