# Running a trained small language model from scratch

## Problem

Generate a continuation of an English prompt with the pretrained TinyStories 260K model (a 5-layer Llama 2-style Transformer), one token at a time, in a standalone serial C++17 program. This is the baseline for the cost of model loading, prompt processing (prefill) and token generation (decode) before parallelization.

## How the baseline works

`chibi_inference.cc` loads the weights and tokenizer from local binary files and encodes the prompt with byte-pair merges. Each step embeds the current token, runs five layers of RMSNorm, grouped-query causal attention with rotary position encoding and a KV cache, and a SwiGLU feed-forward network, then projects to the vocabulary and picks the highest-scoring token (greedy, deterministic). No external libraries or network access are used.

    clang++ -std=c++17 -O3 -Wall -Wextra -pedantic chibi_inference.cc -o chibi_inference
    ./chibi_inference --prompt 'Once upon a time' --tokens 128 --repeat 1

- `--repeat N` runs N independent serial requests; only the first one's text is printed, while counts and timings cover all of them.
- The prompt plus the generated tokens must fit the 512-token context.
- Sanity check: `./chibi_inference --prompt '' --tokens 8` prints `Once upon a time, there was a`.
- `--trace` writes token IDs and all logits for every step to stderr, for numerical comparison with a reference implementation. Do not time with it.

Reported times are setup, and cumulative forward time for prefill (prompt steps, including the one that yields the first generated token) and decode. They exclude token selection and output. 

## Where is parallelism

- **Within a step:** the rows of every matrix-vector product, and the attention heads.
- **Across requests:** independent prompts (or the sweep cases below) can run in parallel or be batched, each with its own KV cache.
- **Not across the tokens of one prompt:** each token depends on the previous one.

Candidates to compare are CPU threads, SIMD, GPU execution and memory layout, using the same model, inputs and precision.

## Optional: single-neuron interventions

    ./chibi_inference --sweep --layer 2 --start 0 --neurons 4 --dose 1 --prompt 'Once upon a time' --tokens 32

This runs a control case. Then, for each neuron in `[start, start+neurons)` of `layer` (layers 0–4, 172 neurons each), it runs a `+dose` case and a `-dose` case. 
The offset is added to that SwiGLU activation at every position, before the down projection.

Each `CASE` line reports two measures for the first next-token distribution:
`kl_from_control` = KL(control ‖ intervention), and `control_token_logp_delta`.
It also shows the generated `TEXT`. `--dose 0` must reproduce the control. A shifted distribution alone does not establish what a neuron means.

The idea comes from [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which works on much larger models; this is not a reproduction of it.
