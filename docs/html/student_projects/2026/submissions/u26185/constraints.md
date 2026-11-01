# LLM project: resources, constraints, and sizing estimates

AI-assisted planning draft, based on the assignment, course environment guidance,
and the student's resource probe from 2026-10-25. Read, verify, and rewrite any text
used in the shared problem description in your own words.

## 1. Confirmed resources for the observed lecture-mig job

| Resource | Observation | Planning interpretation |
|---|---|---|
| CPU architecture | aarch64, ARM Neoverse-V2 / NVIDIA Grace | Use portable C++17; ARM SIMD is a later optimization topic. |
| CPUs available to the process | `nproc` reports 18 | Plan later multicore experiments around up to 18 CPUs for this allocation; confirm binding/allocation when measuring. |
| Physical host CPU | 72 cores, one thread per core | The host total is not the job's CPU allowance. |
| GPU instance | MIG `2g.24gb` | One partition of the GH200 GPU. |
| GPU instance memory | 23552 MiB = 23 GiB total; 34 MiB used at probe time | Budget against the instance total, leaving runtime/workspace headroom. |
| GPU compute | 32 streaming multiprocessors (SMs) | Do not use full-GPU peak performance to predict this instance's speed. |
| Physical GPU memory shown | 97871 MiB | Parent-device total, not memory available to this MIG job. |
| CPU cache shown | L1d 4.5 MiB across 72 cores; L2 72 MiB across 72 cores; L3 114 MiB shared | About 64 KiB L1d and 1 MiB L2 per core. The displayed L3 is not an exclusive per-job budget. |
| GPU driver | 595.58.03; nvidia-smi displays CUDA 13.2 | The displayed CUDA version indicates driver compatibility, not the loaded compiler/toolkit version. |

The assignment's compile cell loads `nvidia/25.9 cuda/12.9` and the course LLVM
path. The serial baseline needs only a C++17 compiler and the standard library.

Not established by this probe: the host-memory allocation/limit, guaranteed CPU
binding, sustained CPU/GPU bandwidth, effective arithmetic throughput, account
compute quota, and queue waiting time. The final job output is a summary rather
than detailed resource limits. Do not infer host-memory entitlement from the
physical-node specification or from the MIG GPU-memory size.

The product name containing `120GB` does not override the instance-memory row.
Likewise, the parent GPU's approximately 18 GiB usage is not this job's usage;
the MIG row is the relevant observation.

## 2. Course and assignment constraints

### Execution

- Every educational batch job has a walltime limit of at most 15 minutes (900 s).
  This covers the whole job script, including setup, all program invocations,
  validation and output, not just the timed computation. Queue waiting is separate.
- The serial baseline should normally finish within a few minutes. Target about
  10-120 s for its main computation after calibration, with comfortable headroom.
- Use `%%sub` for measured or substantial runs on Miyabi compute nodes. `%%bash`
  runs on a shared login node and is appropriate for compilation and short checks.
- The AI agent runs on taulec. The student builds/runs the real workloads on Miyabi.
- `lecture-mig`, `lecture-g`, and `lecture-c` are the course queues outside lecture
  time. During Monday 16:50-18:35, use the corresponding `lecture1-*` names.
  Keep the course group `gt81`.
- The introductory notebook permits MIG for small experiments and full-GPU `-g`
  nodes for larger ones. This document sizes the initial project for the observed
  MIG allocation; full-GPU access does not remove the serial-baseline requirement.

### Baseline implementation and inputs

- Self-contained C++17 in `code/`, using only the standard library. No PyTorch,
  Eigen, BLAS/cuBLAS, external inference engine, or other external code/library.
- Serial implementation: no OpenMP, threads, explicit SIMD intrinsics, GPU code,
  or hand-tuned kernels. Use straightforward loops and flat arrays.
- Compiler optimization such as the assignment's `clang++ -O3 -Wall -std=c++17`
  is allowed. "Plain baseline" does not mean compiling at `-O0`; the compiler may
  optimize or automatically vectorize ordinary loops.
- Generate deterministic input with a fixed seed, or include small input files.
  No runtime downloads or dependence on an external model/dataset service.
- Expose model dimensions, batch size, context length, training steps or inference
  repetitions, and seed through command-line parameters with sensible defaults.
- Time the main computation using `std::chrono`. Report correctness or quality
  metrics as well as performance. Initialization, validation, and file I/O should
  be distinguishable from the main-computation timer.
- Produce meaningful, verifiable results: numerical reference agreement for
  inference; gradient checks, loss reduction and task accuracy for training.
- GPU/CPU parallelization and manual optimizations are later course work. The
  baseline is the reference against which those versions are checked.

### Deliverables

- Supply the needed source/input files and a self-contained `code/problem.md`.
- The student fills and runs the notebook compile/run cells so they work as given.
- The submission directory should contain all and only needed files: no executables,
  large generated outputs or unrelated leftovers.
- The problem description is shared with classmates. Proofread and rewrite
  AI-assisted text, and include only information intended for sharing.

## 3. Sizing principles

- GPU memory fit is necessary for a later GPU version, but does not establish that
  a serial training or inference run fits the time limit.
- Model size, sequence length, batch size, and number of steps/repetitions jointly
  determine feasibility. There is no single maximum parameter count for the course.
- Keep at least one same-size, same-workload case for serial/parallel speedup
  comparisons. Larger parallel-only scaling cases may be added separately.
- A run with a bounded number of training steps is not a promise of convergence.
  Choose an easy generated task and measure loss/accuracy to establish learning.
- Tiny models can fit in CPU cache. Their performance need not reproduce the
  memory-bandwidth behavior of production LLMs; vary model size later.

## 4. Work and memory estimation model

Assume a conventional decoder Transformer, hidden width `d`, `L` blocks, vocabulary
`V`, MLP width `4d`, ordinary multi-head attention, untied input/output embeddings,
and FP32 storage. Biases and normalization parameters are omitted in the counts.

Approximate parameter count:

    P = 12 L d^2 + 2 V d

The estimates below count multiply-add as two FLOPs. Embedding lookups are not
dense products, so using all of `P` in the arithmetic formulas is an approximation.
The attention estimate assumes full square attention products for training/prefill;
skipping masked entries can reduce this work. Softmax, normalization, activation
functions, optimizer work and loop overhead are not explicitly counted.

For batch `B`, sequence length `T`, and `S` training steps:

    training FLOPs ~= S (6 P B T + 12 L B T^2 d)

For one batch-one inference request with prompt length `T` and `G` subsequent
cached token-forward steps:

    inference FLOPs ~= 2 P (T + G)
                       + 4 L d (T^2 + sum(t, t=T+1 through T+G))

Actual generation may use the final prompt logits as its first output and therefore
need one fewer decode forward pass; this difference is negligible for sizing.
Uncached prefix recomputation costs more and is not covered by this formula.
A plain, serial KV cache is a valid algorithmic baseline, but should be stated
explicitly when describing the project. Validate it against uncached computation
using tiny inputs, rather than requiring a large uncached validation every run.

Memory components:

- FP32 weights: `4P` bytes.
- Training with FP32 Adam: approximately `16P` bytes for weights, gradients, and
  two moment arrays, before activations and temporary storage. Plain SGD needs
  approximately `8P` bytes for weights and gradients, before those extras.
- Standard multi-head KV cache: `8 L B T_total d` bytes.
- One FP32 attention matrix per layer: `4 B H T^2` bytes, where `H` is head count.
  Backpropagation may retain several intermediates across all layers.

For perspective, 100 million parameters occupy about 381 MiB as FP32 weights,
or 1.49 GiB for Adam's four parameter-sized arrays. This does not include
activations and does not imply that serial training is feasible in one job.

### Runtime assumptions, not measured hardware performance

No serial model benchmark has been run yet. The tables use hypothetical effective
throughputs of **0.2 and 2 GFLOP/s** on one CPU core to illustrate sensitivity.
These numbers are neither a measured range nor a guaranteed lower/upper bound.
The real implementation can fall outside them; operation mix, cache behavior,
loop ordering, compiler decisions and transcendental functions all matter.
Prefill, decode and backward computation can have very different effective rates.

Convert work to an illustrative time with `seconds = FLOPs / FLOPs_per_second`.
Replace these scenarios with measured seconds per step/request before choosing
the submitted defaults.

## 5. Training scale

| Configuration (L, d, V) | Parameters | B, T, steps | Estimated work | Illustrative time at 2 to 0.2 GFLOP/s |
|---|---:|---|---:|---:|
| 2, 64, 64 | 106,496 | 4, 64, 100 | 18.9 GFLOP | 9-94 s |
| 2, 128, 128 | 425,984 | 4, 128, 20 | 30.2 GFLOP | 15-151 s |
| 4, 256, 128 | 3,211,264 | 4, 128, 100 | 1,067 GFLOP | 8.9-88.9 min |

Recommendation: begin around **0.1 million parameters**, width 64, sequence length
32-64, batch 4, with configurable steps. Around **0.4 million parameters** is a
reasonable next experiment with fewer steps. The 3.2-million-parameter example
demonstrates why memory fit is not enough: its workload is too long for a reliable
default under these scenarios. Fewer steps would change that conclusion.

Use a simple generated symbolic prediction task and a small vocabulary. Report
loss, task accuracy, token count and elapsed time. Timing 20 steps is not evidence
that a model has learned; choose task difficulty and training duration accordingly.
First establish that a tiny model can overfit a tiny batch, then test held-out data.
Full Transformer backward computation also has a greater implementation/debugging
cost than inference, even when memory and runtime are small.

## 6. Inference scale

Each row below is one request: batch 1, prompt 128, approximately 128 generated
tokens, standard KV caching, vocabulary as listed. Weight storage excludes all
working buffers. The cache sizes assume total sequence length 256.

| Configuration (L, d, V) | Parameters | FP32 weights | KV cache | Estimated work | Illustrative time at 2 to 0.2 GFLOP/s |
|---|---:|---:|---:|---:|---:|
| 2, 128, 128 | 0.426 M | 1.625 MiB | 0.5 MiB | 0.260 GFLOP | 0.13-1.3 s |
| 4, 256, 128 | 3.21 M | 12.25 MiB | 2 MiB | 1.81 GFLOP | 0.91-9.1 s |
| 4, 512, 128 | 12.7 M | 48.5 MiB | 4 MiB | 6.85 GFLOP | 3.4-34 s |
| 8, 768, 256 | 57.0 M | 217.5 MiB | 12 MiB | 30.2 GFLOP | 15-151 s |

Recommendation: implement and check a **0.4 M** model first, then use roughly
**3-13 M parameters** for the initial performance workload. Adjust the number
of independent requests/repetitions after measuring. A **50-60 M** model is a
plausible later scale-up experiment, subject to measurement, not a guaranteed
runtime promise. Even that model's listed memory is far below the MIG GPU budget;
host-memory limits still need to be respected for its serial version.

Initialize weights deterministically for numerical tests, or supply a small
locally trained model. Random-weight inference can validate kernels and caching
but cannot demonstrate meaningful language generation. These models and small
vocabularies are educational Transformer workloads, not fluent general-purpose
LLMs. Larger vocabularies increase output-projection work and parameter storage.

## 7. Calibration and choice of defaults

1. Build the simple serial implementation with the notebook's compiler settings.
2. Run a tiny correctness case on Miyabi. For training, use a tiny finite-difference
   gradient check and an overfit test; for inference, compare cached/uncached logits.
3. Measure a short representative workload: for example, 5-10 training steps or
   one inference request. Record dimensions, compiler flags, host, and elapsed time.
4. Keeping dimensions fixed, scale steps or repetitions toward 10-120 s. For
   example, 10 steps taking 4 s suggests about 150 steps for a 60 s computation.
   Account separately for validation and setup in the job walltime.
5. Re-measure after changing dimensions: projection work grows roughly as `d^2`,
   attention work as `T^2`, and cache effects can make simple extrapolation fail.
6. Preserve a same-workload baseline when comparing later parallel versions.
   Report training tokens/s and inference prefill/decode timings separately.

## Sources

- `../pd02_app.ipynb`: serial application, self-contained code, correctness, runtime
  configurability, submission requirements, and compile-cell toolchain.
- `../../pd00_intro/pd00_intro.ipynb`: course queues, full-GPU option, and walltime.
- Course `../AGENTS.md`: baseline implementation and environment guidance.
- Student-provided `lecture-mig` resource probe, 2026-10-25.
- [Official Miyabi hardware specifications](https://www.cc.u-tokyo.ac.jp/en/supercomputer/miyabi/system.php):
  physical-node context only; the observed allocation determines the working budget.
