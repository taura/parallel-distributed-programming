# Making LLM Inference from Scratch

# Problem

I study text generation (inference) of a small large language model (LLM), implemented from scratch.

The model is a tiny Llama-2 style transformer trained on TinyStories. I use the `stories110M` checkpoint, which has about 110 million parameters. The task is: given a short prompt, generate the next tokens one by one to produce a story.

This is a realistic application because LLM inference is widely used in real systems. It is also a good target for this course because:

- it does a lot of computation,
- it has multiple levels of parallelism,
- but it also has an important sequential dependency (next token depends on previous tokens).

So it is a good baseline for later multicore/GPU/SIMD optimization and performance analysis.

# How the baseline code works

The baseline program is `llama2_infer.cpp` in this directory.

- Single-file C++17, serial implementation.
- No external libraries.
- Command-line options control prompt, generation length, seed, temperature, and number of samples.

Main steps:

1. **Load model checkpoint**
   - Read the model configuration and all pretrained weights from `stories110M.bin`.
   - Store weights as flat float arrays.

2. **Load tokenizer**
   - Read `tokenizer.bin`.
   - Build token tables for encoding prompt text into token IDs and decoding token IDs back to text.

3. **Run transformer forward pass token-by-token**
   - For each generated position, run all transformer layers.
   - Use a KV cache (saved keys/values from previous positions) so attention does not recompute old keys/values every step.
   - Compute logits for the next token.

4. **Sample next token**
   - `--temperature 0` uses greedy decoding (argmax), deterministic.
   - `--temperature > 0` uses probabilistic sampling from softmax(logits / temperature).

5. **Repeat until stop**
   - Continue generating until max steps, EOS, or BOS appears again.

6. **Multiple independent stories with `--samples N`**
   - Generate `N` independent stories in one run.
   - Seeds are `seed, seed+1, ..., seed+N-1`.
   - This is useful to scale serial runtime from seconds to minutes.

# Build and run

Download files:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

Compile:

```bash
clang++ -O3 -Wall -std=c++17 llama2_infer.cpp -o llama2_infer
```

Run example:

```bash
./llama2_infer --checkpoint stories110M.bin --tokenizer tokenizer.bin \
  --prompt "Once upon a time" --steps 1024 --samples 20 --seed 123 --temperature 1.0
```

Observed serial baseline (one run):

- `--samples 20`
- generated 3931 tokens
- total time about 258 s
- throughput about 15 tokens/s

Increasing `--samples` increases total runtime approximately proportionally, so `--samples` is a practical way to set serial runtime to a few minutes.

# How to verify results

I use two simple checks:

1. **Human readability check**
   - Output should look like short, mostly coherent TinyStories-style text.

2. **Deterministic check with `--temperature 0`**
   - Greedy decoding gives deterministic output for a fixed prompt/checkpoint.
   - This can be used as a reference when comparing future optimized parallel versions.

# Where time is spent and where parallelism exists

Most time is spent in the transformer compute for each token, especially:

- matrix multiplications in attention and feed-forward layers,
- repeated reading of large weight matrices from memory for every token.

Parallelization opportunities:

- **Matmul rows**: output rows can be computed independently (multicore).
- **Attention heads**: heads are independent inside one layer.
- **Independent samples**: each story in `--samples N` can run independently.
- **SIMD/vectorization**: inner dot products and elementwise operations.
- **GPU mapping**: matrix multiplies and attention are natural GPU kernels.

What is hard to parallelize:

- **Token-to-token dependency**: next token generation depends on previous tokens, so positions in one story cannot be fully parallelized over time.

This mix (large inner parallel work + outer sequential dependency) is exactly why this problem is interesting for performance analysis.
