
# Making a Character-Level MLP Language Model from Scratch

# Problem

I want to train a tiny character-level language model in C++ from scratch, without any external libraries.
The model predicts the next character from the previous fixed-length context of characters.

Why this is a good target for this course:

- It is a real ML workload (forward + backward propagation + optimization), not a toy loop.
- It has many tunable parameters to control runtime: dataset size, context length, embedding size, hidden size, batch size, and number of SGD steps.
- It has clear quality metrics (cross-entropy loss and perplexity), and generated text can be checked informally.
- It has clear parallelization opportunities for later assignments on multicore/GPU/SIMD/cache analysis.

# How the baseline code works

File: `char_mlp_lm.cc`

The baseline is a serial, deterministic C++17 implementation of a one-hidden-layer MLP language model:

1. **Data generation**
   - Generates a synthetic ASCII text corpus in the program (no downloads, no external files).
   - Uses fixed random seed for reproducibility.
   - Encodes characters into a vocabulary of size 96 (`\n` + printable ASCII).
   - Splits data into train/validation (90%/10%).

2. **Model**
   - Input: previous `K` character IDs.
   - Embedding lookup for each context position.
   - Concatenate embeddings into a vector of length `K * E`.
   - Hidden layer: `tanh(W1 * x + b1)`.
   - Output layer: logits over all characters `W2 * h + b2`.
   - Loss: softmax cross-entropy for next-character prediction.

3. **Training (serial SGD)**
   - Randomly samples training positions.
   - Runs forward and backward pass with explicit loops.
   - Accumulates gradients over a mini-batch.
   - Updates parameters by plain SGD.
   - Prints periodic progress: train loss, validation loss, train/val perplexity.

4. **Correctness / sanity checks**
   - Optional finite-difference gradient check (`--grad_check 1`) for selected parameters.
   - Deterministic run with fixed seed.
   - Reports final validation loss/perplexity and parameter checksum.
   - Prints generated text samples to visually confirm learning progress.

5. **Timing**
   - Measures and prints main training time with `std::chrono`.
   - Prints throughput (examples/second).

Main runtime knobs (command line):

- `--train_tokens`
- `--steps`
- `--batch`
- `--context`
- `--embed`
- `--hidden`
- `--lr`
- `--seed`

# Where is parallelism

This baseline has substantial parallel work that can be exploited later:

1. **Across mini-batch samples**
   - Each sample in a batch is independent for forward/backward until gradient reduction.
   - Natural for multicore parallel loops and GPU thread blocks.

2. **Inside matrix-vector/matrix-matrix style operations**
   - Hidden and output layer computations are dot-product heavy.
   - Good target for SIMD/vectorization and GPU kernels.

3. **Gradient accumulation and reduction**
   - Per-sample gradients can be accumulated in parallel then reduced.
   - Requires careful treatment of races and reduction overhead.

4. **Memory locality / cache optimization**
   - Parameter tensors (`W1`, `W2`) dominate traffic.
   - Loop ordering and blocking can improve cache reuse.

5. **Limits to speedup**
   - Softmax normalization and reductions add synchronization/reduction costs.
   - Embedding updates can be scattered and less cache-friendly.
   - Amdahl's law and memory bandwidth can limit scaling.

---

# Results on Miyabi

The serial baseline was compiled with clang++ -O3 -Wall -std=c++17 and run on Miyabi through the lecture-mig queue on October 30, 2026 (job 3738279). This run used the CPU serial implementation; submitting to this queue does not mean the program used the GPU.

Run command (from the code directory):

```bash
./char_mlp_lm --steps 4000 --eval_every 200 --sample_every 1000 --batch 64 --context 16 --embed 32 --hidden 256 --train_tokens 800000 --val_batches 100 --lr 0.05 --seed 42 --grad_check 1
```

Observed results:

- The gradient check passed for the selected parameters, with a maximum relative error of 1.52022e-07. This is a limited sanity check, not a proof that the entire implementation is correct.
- Reported training loss decreased from 4.562459 at step 1 to 0.182730 at step 4000.
- Final validation loss was 0.181096 and final validation perplexity was 1.198530.
- Reported training time was 85.750502 seconds (excluding queue waiting time).
- The run processed 256000 training examples, with a reported throughput of 2985.405270 examples per second.

The decreasing loss indicates improved next-character prediction on the synthetic corpus. However, the generated text at step 4000 was still not fluent English. These results do not demonstrate general English language ability. The difference between validation performance and free-running text generation needs further investigation.

This provides a serial timing baseline for later parallelization. The parallelization opportunities and possible bottlenecks discussed above are hypotheses to investigate; no parallel speedup or detailed profiling has been measured yet.

