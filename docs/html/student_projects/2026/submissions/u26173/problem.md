
# Making a Tiny GPT-style Character Language Model from Scratch

# Problem

I want a serial baseline for "LLM training from scratch" that can run in a few
minutes and can later be parallelized on multicore CPU and GPU.

The task is next-character prediction on text. Given a sequence of bytes
`x[0], x[1], ..., x[L-1]`, the model predicts `x[1], x[2], ..., x[L]`.
Training minimizes cross-entropy loss.

I use byte-level tokens (vocabulary size 256), so no external tokenizer or
dictionary is needed. The input text is read from a local file by command-line
option (for example `../harrypotter.txt`), or a built-in tiny fallback text.

The baseline should:

- be self-contained C++17 and use only the standard library,
- be deterministic with a fixed random seed,
- expose problem size and runtime knobs via command-line options,
- print training/validation loss, perplexity, and elapsed time.

# How the algorithm/baseline code works

File: `code/app.cc`

Model structure (small GPT-style block):

1. token embedding `E[token] -> R^D`
2. linear projections to query/key/value (`Wq`, `Wk`, `Wv`)
3. causal self-attention over positions `0..t`
4. output projection `Wo`
5. vocabulary projection `Wout` to logits over 256 byte tokens
6. softmax + cross-entropy loss

Training:

- optimizer: plain SGD
- gradients: manual backpropagation through all operations
- batching: `batch_size` random subsequences per step
- split: first 90% train, last 10% validation
- metrics printed every `print_every` steps:
  - `train_loss`, `valid_loss`
  - `train_ppl = exp(train_loss)`, `valid_ppl = exp(valid_loss)`
  - elapsed seconds

Main command-line parameters:

- `--data PATH`
- `--steps N`
- `--seq-len N`
- `--d-model N`
- `--batch-size N`
- `--lr X`
- `--seed N`

This makes runtime tunable from seconds to several minutes by adjusting
`steps`, `seq-len`, `d-model`, and `batch-size`.

# Where is parallelism

This baseline is intentionally serial, but it has many parallel opportunities:

- matrix/vector loops in Q/K/V projections,
- attention score computation for each `(t, j)` pair,
- softmax and loss over vocabulary,
- gradient calculations for all parameters,
- independent sequences inside each training batch.

Potential mapping later:

- multicore CPU: parallelize loops over time positions, vocabulary, and batch,
- GPU: map matrix/attention/softmax kernels to many threads,
- SIMD: vectorize dot products and projection loops,
- cache optimization: improve data layout and loop ordering for locality.

Likely performance limits to analyze later include memory bandwidth,
reduction overhead (softmax/dot products), and arithmetic intensity.
