
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Training a Small Decoder-Only Transformer on MQAR from Scratch.

## The task: remembering facts given in the input

Large language models (LLMs) can answer questions about information that appears earlier in their input ("in-context recall"). Asked "What is Alice's phone number?" after a list of phone numbers, an LLM finds the matching entry and copies the number. We study this ability in a minimal, fully synthetic form called **Multi-Query Associative Recall (MQAR)**.

Each input is a sequence of tokens with two parts:

1. **Context**: `N` key-value pairs. Keys are distinct within a sequence, and each key is assigned a random value.
2. **Queries**: `Q` keys drawn from the context. After each query key, the model must output the value that was paired with it in the context.

Example with `N = 4`, `Q = 3` (`k*` are key tokens, `v*` are value tokens):

```
context:  k12 v3  k5 v40  k33 v3  k7 v18
queries:  k5 [v40]  k7 [v18]  k12 [v3]
```

The bracketed tokens are the answers the model must predict. The key-value assignment is regenerated randomly for every sequence. A model therefore cannot memorize the answers; it has to learn a general *procedure*: find the earlier occurrence of the query key and copy the token after it. In real Transformers this procedure is implemented by attention (so-called "induction heads"), so MQAR is a small but genuine Transformer workload.

## What we build

We train a small decoder-only Transformer (the architecture behind GPT-style LLMs) **from scratch** on MQAR:

- Random initial weights, deterministic from a seed.
- Training data generated on the fly by a seeded random generator; no dataset download.
- Forward pass, backpropagation and the Adam optimizer, all written by hand in C++17 with only the standard library.

## How we know the result is correct

- **Accuracy vs. chance.** With `Vv` possible values, random guessing gets about `1/Vv` of the answers right (about 1.6 % for `Vv = 64`). A model that learned recall gets far above this on held-out sequences that were never used for training.
- **Gradient check.** On a tiny model, the gradients computed by backpropagation are compared against finite differences `(f(w+ε) − f(w−ε)) / 2ε`. Their relative error must be small.
- **Overfit test.** The model must be able to drive the loss of one fixed small batch close to zero.
- **Reproducibility.** The same seed and parameters give the same output.

## Problem size and difficulty

Everything is set by command-line parameters:

| Parameter | Meaning |
|---|---|
| `--pairs N`, `--queries Q` | facts in context and questions; sequence length `T = 2N + 2Q` |
| `--keys K`, `--values Vv` | key/value vocabulary sizes; vocabulary `V = K + Vv` |
| `--layers L`, `--dim d`, `--heads H` | model depth, width and attention heads |
| `--batch B`, `--steps S`, `--lr` | sequences per step, training steps, learning rate |
| `--seed` | random seed for data and weights |
| `--shift 0\|1` | token shift on/off (see "Model" below; default on) |
| `--gradcheck 1`, `--overfit 1` | correctness checks (see above) |

More pairs make the task harder and the sequences longer. A larger `d` and `L` make the model more capable and each step more expensive.

We use the same program at three scales:

| Purpose | Example setting | Notes |
|---|---|---|
| Correctness | `--dim 16 --heads 2 --pairs 4 --queries 4 --keys 8 --values 8 --batch 2 --gradcheck 1 --steps 0` | gradient check; well under a second |
| Overfit test | `--dim 32 --heads 2 --pairs 4 --queries 4 --keys 8 --values 8 --batch 4 --overfit 1 --steps 300` | training loss of one fixed batch goes to about 0 |
| Serial learning run (defaults) | `L=2 d=64 H=4 N=Q=8` (`T` = 32), `K=Vv=64`, `B=32`, 300 steps; 0.13 M parameters | held-out accuracy rises from chance (1.6 %) to about 100 % |
| Performance benchmark | `--layers 4 --dim 256 --heads 8 --pairs 32 --queries 32 --batch 16 --steps 10`; 3.3 M parameters, `T` = 128 | about 40 GFLOP per step; a fixed number of timed steps, the same workload for serial and parallel versions |

The serial baseline must finish within the 15-minute job limit, so the benchmark scale runs only a few steps. The training *speed* of each step is what we compare. A faster (parallel) version can afterwards train the benchmark-size model to high accuracy. The quantity "time to reach X % accuracy" then shows the gain much more clearly than any single number.

Measured on a Miyabi compute node (one NVIDIA Grace CPU core), the default run trained for 300 steps in 21.9 s: 0.073 s per step, about 14,000 tokens/s and an estimated 8.8 GFLOP/s. The whole program, including evaluation, took 23.5 s, and held-out accuracy reached 100 % (chance: 1.6 %). The benchmark scale has not been timed on Miyabi yet. At the same FLOP rate, one benchmark step (about 40 GFLOP) would take about 4.6 s, so 10 steps would take about 46 s.

# How the algorithm/baseline code works

## Data generation

For each sequence:

1. Choose `N` distinct keys from the `K` key tokens.
2. Give each one a random value from the `Vv` value tokens.
3. Write the pairs as the context, then pick `Q` of the stored keys (uniformly, with repetition) and write each query key followed by its value.

Training sequences and evaluation sequences come from different random streams, so evaluation always uses unseen sequences.

## Model

A standard pre-normalization decoder-only Transformer. All tensors are flat `std::vector<float>` arrays in row-major order.

1. **Embedding.** Each token becomes a `d`-dimensional vector: a token embedding plus a learned position embedding, plus a second learned embedding of the *previous* token ("token shift").
2. **`L` Transformer blocks**, each consisting of:
   - LayerNorm, then **causal multi-head self-attention**. For every position the model computes query, key and value vectors (`Q = xW_q`, `K = xW_k`, `V = xW_v`), the scores `QKᵀ / √(d/H)`, and a softmax over earlier positions only (the causal mask). The output is the weighted sum of `V`, projected by `W_o`. The result is added back to the input (residual connection).
   - LayerNorm, then an **MLP** `W_2 · GELU(W_1 · x)` with hidden width `4d`, again added back as a residual.
3. **Output.** A final LayerNorm and a projection to `V` scores (logits). Only the `Q` query positions per sequence are scored, so the projection is computed only there.

The number of parameters is about `P ≈ 12·L·d² + 3·V·d + T·d`.

**Why the token shift.** Without it, the model must first learn to copy each key into the position of the value that follows it (one attention layer), and only then can a second layer match the query against it. Neither layer gets a useful learning signal until the other works. In our runs the plain model stayed at about `1/N` accuracy (it only learned "answer with some value from the context") even after 160,000 training sequences. With the token shift, every value position already knows its key, and the model learns the task in about 200 steps. Similar short "shift" or convolution layers are used in research models designed for this kind of recall. `--shift 0` reproduces the plateau.

## Training

- **Loss.** Softmax cross-entropy, computed only at the query positions where an answer must be predicted. The other positions are not scored.
- **Backward pass.** Every layer has a hand-written backward function that computes gradients for its inputs and weights, in reverse layer order.
- **Optimizer.** Adam updates every parameter. A fresh batch of `B` sequences is generated for each step.
- **Evaluation.** Every few steps, and at the end, the program reports the training loss and the accuracy on held-out sequences.

## Timing and output

The main computation (the training loop) is timed with `std::chrono` separately from setup, gradient checking and evaluation. The program prints:

- all parameters
- loss and accuracy over time, together with the chance level
- the training time and tokens processed per second
- an estimated FLOP rate

The estimate counts only matrix products and attention, with backward taken as twice the forward cost:
`FLOPs/step ≈ 3 · B · ( L·(24·d²·T + 2·d·T·(T+1)) + 2·d·V·Q )`.

# Where is parallelism

Training time is dominated by matrix products and attention, and both have a lot of independent work.

- **Across sequences in a batch.** The `B` sequences are processed independently in the forward and backward passes. Only the weight gradients are summed over the batch, a reduction that each thread can perform into its own buffer and combine at the end.
- **Matrix multiplications.** The projections (`W_q`, `W_k`, `W_v`, `W_o`) and the MLP layers are matrix products of shape `(B·T × d) · (d × d)` or `(d × 4d)`. They account for most of the FLOPs. Every output element is independent. This is where multicore parallelization, cache blocking (tiling), SIMD instructions and the GPU should give the largest gains.
- **Attention.** Each (sequence, head) pair is independent, and so is each row of the `T × T` score matrix. The causal mask makes the work triangular, which causes load imbalance if rows are split naively. The scores grow as `T²`, so for long sequences a fused attention computation that never stores the whole score matrix saves memory traffic.
- **Element-wise operations.** GELU, residual additions and the Adam update touch every element independently. They are trivially parallel but limited by memory bandwidth rather than arithmetic.
- **Reductions.** LayerNorm (mean and variance), softmax (maximum and sum) and the loss need parallel reductions across a row.
- **Data generation and evaluation.** Each sequence can be generated and evaluated independently, using a separate random stream per sequence so that results stay deterministic.

**What is not parallel.** Layers depend on each other in order, and each training step depends on the weights produced by the previous one. Parallelism therefore comes from *within* a step, not across steps.

**Expected performance questions.** At small `d` the weights fit in cache and the GPU is underused, so speedups should grow with `d`, `T` and `B`. Comparing the measured FLOP rate against the hardware's peak tells us whether each part is limited by computation or by memory bandwidth. Parallel versions may sum floating-point numbers in a different order than the serial version, so results are compared with a tolerance rather than bit-for-bit.

