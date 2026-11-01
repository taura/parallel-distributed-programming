# Project guide: accelerating decoder-only Transformer inference

## Scope

Implement and measure the same operations used in a small autoregressive language
model, with synthetic tokens and fixed random weights. There is no training or
external dataset. Generated token IDs are a reproducible computation result,
not meaningful text. Numerical checks establish computational correctness.

The initial implementation is serial, with plain loops and no explicit SIMD,
threads or GPU code. The compiler may auto-vectorize. Repeated full-prefix
evaluation is the baseline; KV caching is a later optimization.

## Miyabi notebook cells

Select **Python (Miyabi G)**. Put this in the existing compile cell:

```bash
%%bash
cd code
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
clang++ -O3 -Wall -Wextra -std=c++17 transformer.cc -o transformer
```

Run `%load_ext miyabi` once in a Python cell, then use the existing run cell for a
small correctness check:

```bash
%%sub
cd code
./transformer --dim 24 --heads 3 --ff 48 --layers 2 --vocab 17 --prompt 9 --generate 4 --repeats 2 --check
```

Look for `checks=PASS` and finite probability entropy. Then measure a larger case:

```bash
%%sub
cd code
./transformer --dim 256 --heads 4 --ff 1024 --layers 2 --vocab 64 --prompt 128 --generate 8 --repeats 3 --seed 42 --check
```

Real performance measurements are yours to run on Miyabi compute nodes using
`%%sub`. The compile cell runs on the login node. A Jupyter terminal runs on
taulec. The course job walltime limit is 15 minutes; target seconds to a few
minutes for serial runs. This project produces numerical output, so no image
conversion is needed. Fill the notebook cells yourself with the commands above.

## Parameters

| Option | Default | Meaning |
|---|---:|---|
| `--dim` | 256 | Model width D |
| `--heads` | 4 | Heads H; D must be divisible by H |
| `--ff` | 1024 | Feed-forward width F, usually 4D |
| `--layers` | 2 | Number of decoder blocks L |
| `--vocab` | 64 | Number of possible token IDs V |
| `--prompt` | 128 | Input context length P |
| `--generate` | 8 | New token count G; zero benchmarks prefill only |
| `--repeats` | 3 | Timed identical requests, after one full warm-up request |
| `--seed` | 42 | Fixed initialization/input seed |
| `--check` | off | Untimed correctness suite |

Use `./transformer --help` for usage. All sizes must be positive except generate.
The parser bounds model width at 2048, FF width at 8192, layers at 32, vocabulary
at 4096 and prompt+generate at 8192. These are parser limits, not recommended
workloads. Large combinations can require substantial memory/time. Start small;
adjust repetitions first if a benchmark finishes too quickly.

Model storage is `2*V*D + L*(4*D*D + 2*D*F)` floats. With maximum context
S=P+G, the scratch arrays use `7*S*D + S*F + S + V` floats. Positional encodings
use another `S*D` floats. Saved output distributions use `max(1,G)*V` floats.
Multiply by 4 for approximate bytes, plus small token vectors and object overhead.
Checks temporarily create a double-precision model copy. Attention arithmetic is
quadratic in context length even though its scores use only a one-row buffer.

## How the computation is organized

`make_model` deterministically initializes embedding, projection, feed-forward
and vocabulary-head weights. Position encodings are sinusoidal. LayerNorm has
fixed gain 1 and bias 0; linear layers are bias-free. This is a small architectural
variant, not a loader for a specific pretrained model format.

`linear`, `normalize`, `softmax` and `attention` are the basic kernels.
`forward` assembles pre-normalized causal decoder blocks with residual connections
and exact GELU, then computes next-token probabilities at the last position.
All buffers are allocated before benchmarking.

`request` first evaluates the prompt. With G>0, this produces the first generated
token. It then recomputes the entire growing prefix G-1 times, appending one greedy
token each time. There is no EOS stopping rule: exactly G tokens are generated.
Argmax ties select the lowest token ID. G=0 still evaluates the prompt and reports
its next-token distribution, but appends nothing.

## Timing and results

- `inference_total_seconds`: summed timed regions across all requests.
- `mean_request_seconds`: total divided by repeats; prefill plus all decode steps.
- `best_request_seconds`: fastest timed request, useful alongside the mean.
- `mean_prefill_seconds`: time to process the prompt, including the vocabulary
  head/softmax and first greedy selection when G>0.
- `prefill_tokens_per_second`: P divided by mean prefill time; this counts input
  tokens and is not a generation rate.
- `mean_decode_seconds`: summed subsequent G-1 passes per request.
- `decode_tokens_per_second`: `(G-1)/mean_decode_seconds`, printed only for G>1.
  This baseline recomputes previous positions, so label results **uncached**.
- `generated_tokens_per_second`: G divided by mean request time, including prefill.
- `mean_output_entropy_nats`: average entropy of each predicted distribution.
  It should be between 0 and `log(V)` up to rounding; it is not model accuracy.
- `checksum`: weighted sum of all probabilities across all generation steps.
- `generated_token_ids`: greedy continuation from the last request.

Timers include embedding lookup, normalization, projections, attention, GELU,
residuals, last-position vocabulary projection, softmax and token selection/append.
They exclude initialization, allocation, input reset, copying distributions to
the validation buffer, validation, checks and printing. These are warmed-up,
in-memory model-computation timings, not network-serving latency. One warm-up
request is excluded. Identical requests must reproduce the warm-up checksum.

## Correctness checks

Every request checks that all probabilities are finite, in [0,1], and sum to one.
`--check` additionally runs:

1. An independent small attention reference: explicitly construct each head's
   causal attention matrix in double precision and multiply it by V. Maximum
   absolute error must be below `1e-6`.
2. The analytical Q=K=0 case: each attention output must be the arithmetic mean
   of visible value vectors, within `1e-6`.
3. Float-versus-double execution on up to eight prompt tokens, using exactly the
   same initialized weights. Hidden-state error must be below `2e-3`, probability
   error below `2e-4`. This uses the same forward algorithm in higher precision;
   it is a precision check, not an independent implementation of the whole model.
4. Causality: changing or removing future tokens must preserve earlier hidden
   states within `1e-6`. A one-token input has no nontrivial suffix to perturb;
   use the supplied nine-token test to exercise this property.

Checks are short numerical tests, not proof of every possible input. For later
optimized implementations, compare full probability vectors at every step on the
same fixed input sequence. Small numerical changes can change a greedy choice
near a tie, causing subsequent free-running sequences to diverge. Feeding the
same reference tokens to both versions isolates numerical errors from that effect.

## Experiments to run on Miyabi

**1. Establish a serial baseline.** Run the larger command above. Record the
compiler/version/options, queue/node type and all parameters. If its total timed
duration is T seconds, estimate repetitions for 20 seconds as `ceil(3*20/T)`.
Allow additional time for the full warm-up, checks and other untimed work.

**2. Study prefill scaling.** Set generate=0 and sweep prompt lengths 64, 128,
256 and 512 with D=256, F=1024, L=2 and H=4. Adjust repeats to obtain useful
durations. Does doubling context approach 2x or 4x runtime? Projection/FF work
is linear in context, while attention arithmetic is quadratic.

**3. Study generation cost.** Use prompt=128 and generate=1, 8 and 32. Keep model
and seed fixed. Compare prefill and subsequent decode time separately. The prefix
grows during generation, so later uncached steps do more work.

**4. Study width.** Compare (D,F)=(64,256), (128,512), (256,1024), holding other
parameters fixed. Changing dimensions changes the model; each configuration
needs its own serial baseline before reporting optimization speedup.

Use three independent jobs for selected cases and report medians. Fill in your
own results; no Miyabi performance values are assumed here:

| D | F | L | H | V | Prompt | Generate | Repeats | Prefill s | Decode s | Generated tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1024 | 2 | 4 | 64 | 64 | 0 | | | | N/A |
| 256 | 1024 | 2 | 4 | 64 | 128 | 0 | | | | N/A |
| 256 | 1024 | 2 | 4 | 64 | 256 | 0 | | | | N/A |
| 256 | 1024 | 2 | 4 | 64 | 128 | 8 | | | | |

## Later optimization milestones

1. **KV caching:** store keys and values for every layer/position during prefill.
   During decode, compute only the new position and attend to cached values.
   Compare against full-prefix probabilities on fixed token sequences. Cache
   storage is approximately `2*L*S*D` floats. An old position's causal state
   cannot depend on future tokens, which makes this reuse valid.
2. **Multicore:** parallelize independent projection/FF output elements and
   attention queries/heads. Provide private score buffers. Measure scaling with
   worker count; autoregressive token steps remain sequential.
3. **Locality/SIMD:** investigate dot-product vectorization and tiled matrix
   operations. Measure prefill and cached decode separately.
4. **GPU:** implement batched matrix operations and attention kernels. Record both
   device-only and transfer/synchronization-inclusive time. State whether weights
   and KV cache stay on the device. Small cached decode can be bandwidth- or
   launch-overhead-limited even if prefill scales well.

Report separate speedups for algorithmic reuse and parallel execution. Avoid
attributing a change from uncached serial to cached GPU entirely to the GPU.
This small benchmark does not represent production-scale vocabulary, serving
batches or trained activation distributions.

## Submission

Proofread and rewrite `problem.md` in your own words and add measured results.
Fill and run the notebook compile/run cells. Keep `transformer.cc`, `problem.md`
and needed documentation. Remove generated executables and the earlier classifier
files yourself before submission. Nothing in the project requires those files.
Include only information you want classmates to see.

## Local verification during project creation

Compiled on taulec with `g++ -O2 -Wall -Wextra -std=c++17` without warnings.
The supplied tiny case (D=24, H=3, F=48, L=2, V=17, P=9, G=4, seed=42)
passed all checks: independent attention error was approximately `6.1e-8`,
hidden-state float/double error `5.4e-7`, and probability error `1.4e-8`.
Prefill-only execution, one-token generation, odd widths, a single head, the
one-feature/one-token-vocabulary case, and invalid arguments were also checked.
These establish small-case correctness, not Miyabi performance. The locally
generated `transformer` executable should be rebuilt on Miyabi and removed before
submission.
