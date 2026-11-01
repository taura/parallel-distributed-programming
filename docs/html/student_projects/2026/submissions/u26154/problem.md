# Making Tiny LLM Inference from Scratch

# Problem

I implement a small decoder-only Transformer inference system in C++17 from scratch.
The model is a pretrained tiny language model (`EleutherAI/pythia-14m`) stored as local files in `code/models/EleutherAI_pythia_14m`.

The goal of this baseline is not to build a chatbot as good as a large modern LLM. The goal is to create a realistic inference workload with clear performance bottlenecks that can later be parallelized and optimized.

This baseline uses only standard C++ and local files. It does not use PyTorch or other external libraries at runtime.

# How the algorithm/baseline code works

`app.cc` performs autoregressive decoding.

The main processing steps are:
1. Load the model configuration and pretrained weight tensors from local files.
2. Read the input prompt from a file or command-line argument.
   - Accept token IDs directly or convert text into tokens using the model's tokenizer.
3. Perform Transformer inference:
   - Look up token embeddings.
   - Calculate Q, K, and V projections.
   - Apply rotary positional embeddings (RoPE) and causal self-attention.
   - Calculate feed-forward network (MLP) outputs.
   - Apply layer normalization and residual connections.
4. Calculate the output logits over the vocabulary.
5. Select the next token using greedy decoding (`top-k=1`) or sampling, append it to the sequence, and repeat.


The code prints:

- average time of major stages,
- generated tokens per second,
- verification values (NaN/Inf check, softmax row-sum error, logits checksum, generated token IDs).

# Meaningful and verifiable results
The program produces generated text and token sequences using actual pretrained language-model weights.

To verify correctness, I compared its generated token IDs with the official Hugging Face implementation of Pythia-14M under the same input token IDs and greedy decoding settings.

- "The capital of Japan is"
- "The capital of the United States is"
- "I run in"

After fixing the QKV head layout in attention, token predictions and inspected logits agree with the reference implementation for tested prompts. 

Although the generated text is sometimes repetitive or factually incorrect due to the limited capabilities of the small pretrained model, this does not necessarily indicate an error in the C++ implementation. The main goal is to study inference performance rather than text-generation quality.

# Preliminary Performance Results

Example serial runs showed:

- ~3.85 seconds for 100 generated tokens.
- ~14.93 seconds for 200 generated tokens.

For the 200-token experiment, the largest measured components were:
- MLP: ~8.97 seconds (about 60%).
- QKV projection: ~2.90 seconds (about 19%).
- Attention computation: ~1.16 seconds (about 8%).

These results suggest that matrix operations dominate the current implementation and that inference becomes increasingly expensive as the generated sequence grows.

Also, doubling the generation length from 100 to 200 tokens increased execution time by about 3.9x. 
This suggests that the computational cost grows faster than linearly with generation length, making it interesting to investigate redundant computation and memory reuse.


# Where is parallelism

Parallelism is available in many loops:

- Matrix operations: QKV projections, attention output projections, MLP layers, and vocabulary projection can benefit from SIMD instructions, multicore execution, and GPUs.
- Attention: Attention heads and parts of the attention calculation can be processed independently.
- Memory locality: Matrix layouts, cache reuse, and temporary buffers can be optimized to reduce memory-access overhead.
