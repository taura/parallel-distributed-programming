# Training a small LLM from scratch

# Problem

Large language models (LLMs) are often decoder-only Transformer networks trained to predict the next word. This project trains a very small character-level version from random weights using Tiny Shakespeare, a roughly 1.1-million-character collection of Shakespeare dialogues in `tinyshakespeare.txt`(taken from karpathy's github [karpathy/char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)).

- Input: raw text, with each distinct byte (a character in this ASCII corpus) mapped to one token. Case, spaces, punctuation and newlines are retained. This file has $V = 65$ characters.
- Task: given characters $x_1, \dots, x_t$, predict the probability of the next character $x_{t+1}$.
- Training minimizes average cross entropy over a batch of $B$ windows of $T$ input characters:

$$ L = -\frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} \log p_\theta\left(x^{(b)}_{t+1} \mid x^{(b)}_1, \dots, x^{(b)}_t\right) $$

- Output: estimated training and held-out loss before and after training, generated text before and after training, and the training-loop time.

This is an end-to-end language-model training problem: preparing text, building a vocabulary, training, evaluating on held-out text, and generating text. The transformer block is one part of it. The goal is to learn basic letter patterns, spaces, common words and some dialogue-like structure, not to generate fluent Shakespeare. The computation has the same broad structure as LLM training, only much smaller.

How to check the results are correct:

- Before training the model guesses almost uniformly, so loss should be near $\ln 65 \approx 4.17$.
- Training and held-out loss should both fall. A quick local check with `./small_llm 200 4 24 24 42` changed held-out loss from about 4.22 to 2.70. Actual results depend on size and steps.
- Generated text should shift from random characters to more frequent letter combinations, fragments of words and eventually more recognizable dialogue-like text; short training runs will still produce nonsense.
- The held-out section is from the same collection, so a low held-out loss does not measure general English understanding.

# How the algorithm/baseline code works

`small_llm.cc` is a single C++17 file using only the standard library. ALl code for layers, backward pass, optimizer etc. will be written by hand as plain serial loops.

## Data preparation

1. Read `tinyshakespeare.txt` as raw bytes and map each distinct byte to an integer id in byte order; do not discard line breaks or punctuation.
2. Use the first 90% of the text for training and the last 10% for validation. Windows from the two parts never cross the boundary.
3. Sample random contiguous windows from the training part. The vocabulary contains the characters in the file, but model weights are learned only from training windows.

## Model

For a window of $T$ characters (default 32) with model width $d$:

1. Embedding: each character becomes a learned vector of size $d$, plus a learned vector for its position.
2. One causal self attention layer (single head):

$$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{d}} + M \right) V $$

   where $Q$, $K$, $V$ are linear projections of the input and the mask $M$ sets future positions to $-\infty$, so position $t$ only uses positions $0, \dots, t$. The result goes through an output projection and is added to the input (residual connection).
3. Feed forward layer: a linear layer to size $2d$, tanh, a linear layer back to size $d$, added to its input (residual connection).
4. Output layer: a linear layer to $V$ scores, then softmax gives next-character probabilities.

There is no layer norm and only one block, which keeps the backward pass short enough to read.

## Training loop

```
for step = 1 .. steps:
    for b = 1 .. batch:
        pick a random start position in the training text
        take T input characters and their T next-character targets
        forward pass: compute and cache every layer's activations
        backward pass: add dL/dw for every parameter w into its gradient
    Adam update of every parameter (learning rate 0.003), then zero the gradients
estimate training and held-out loss on 32 evenly spaced windows in each section
generate 240 characters: prompt with "ROMEO:\n" and repeatedly sample the next character
```

## Building and running

Build on Miyabi from `code/` after loading the course compiler environment:

```bash
clang++ -O3 -Wall -std=c++17 small_llm.cc -o small_llm
./small_llm 1200 8 32 32 42
```

Run on a compute node. The program reads the corpus at run time.

## Problem size

| Argument | Meaning | Default | Allowed |
|---|---|---|---|
| 1 | training steps | 1200 | at least 1 |
| 2 | windows per batch $B$ | 8 | 1 to 1024 |
| 3 | model width $d$ | 32 | 4 to 256 |
| 4 | context length $T$ | 32 | 4 to 128 |

- With the defaults the model has about 14,000 parameters. So, I will need to adjust steps, batch size, width or context to keep the serial run within a few minutes on Miyabi.
- The linear-layer work per training step grows roughly with $B \cdot T \cdot (d^2 + Vd)$, while attention grows with $B \cdot T^2 \cdot d$. This gives several independent parameters with which to scale the computation.
- Compare both training and held-out loss when changing parameters, not only the time. The losses are estimates from fixed windows and need not fall monotonically from one experiment to another.

# Where is parallelism

- Windows in a batch: the forward and backward pass of each window are independent. This is the most direct source of parallelism and maps to CPU threads (for example one OpenMP thread per window).
- Linear layers: every one of the $T$ character vectors is multiplied by the same weight matrix, and every output element is an independent dot product. Stacking all windows of a batch gives one larger matrix multiply with $BT$ rows, which suits SIMD, GPUs, and tensor cores much better than many tiny ones.
- Attention: independent for every window and every query position; each score is a dot product.
- Softmax: independent for every row, but each row needs a reduction (maximum and sum).
- Elementwise operations (tanh, residual adds, Adam update): fully independent per element.
- Evaluation (`estimate_loss`): independent per sampled window, combined with a sum reduction.


What cannot be parallelized directly:

- Training steps depend on each other, since each step uses the parameters updated by the previous one.
- Layers run in order within a window and each layer needs the previous layer's output.
- Text generation is sequential, so each new character depends on the previous characters.

