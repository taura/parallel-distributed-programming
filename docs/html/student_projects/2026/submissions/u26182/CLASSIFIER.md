# Earlier example: learned digit classifier

The selected assignment project is now Transformer inference. See `README.md`
and `TRANSFORMER.md` for that project. The instructions below describe the earlier
classifier example only; its description has been replaced in `problem.md`.

## Files and learning goal

- `inference.cc`: complete serial C++17 generator, trainer, inference engine,
  benchmark, and optional numerical reference check.
- `problem.md`: draft description for classmates; proofread and rewrite it yourself.
- `README.md`: build/run instructions and experiments.

Start with the serial baseline for this assignment. Later work can parallelize
the forward pass and analyze what limits speedup. All input is generated locally;
no libraries, model downloads, or Python implementation are needed.

## Build and run on Miyabi

Select the **Python (Miyabi G)** kernel. Fill the notebook's existing compile cell
with the following, retaining its provided module setup:

```bash
%%bash
cd code
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
clang++ -O3 -Wall -Wextra -std=c++17 inference.cc -o inference
```

Run this in a Python notebook cell once per kernel:

```python
%load_ext miyabi
```

First use this small correctness run in the existing run cell:

```bash
%%sub
cd code
./inference --hidden 64 --samples 257 --batch 17 --repeats 2 --train-steps 4000 --check
```

The deliberately uneven sizes exercise the final partial batch. Look for
`reference_check=PASS`, finite loss, and accuracy well above chance. Then replace
the run command with a baseline measurement:

```bash
%%sub
cd code
./inference --hidden 256 --samples 4096 --batch 64 --repeats 100 --train-steps 20000 --seed 42 --check
```

`%%sub` uses the course job configuration and runs on a compute node. The notebook
and its module setup are on Miyabi; a Jupyter terminal is on taulec. Do not use
`%%bash` for the benchmark. If the job configuration is missing, consult the
course instructions/instructor. The job walltime limit is 15 minutes; aim for a
serial benchmark lasting seconds to a few minutes.

This project produces numerical output, so image-conversion cells are optional.

## Command-line parameters

| Option | Default | Meaning |
|---|---:|---|
| `--hidden` | 256 | Hidden layer width H |
| `--samples` | 4096 | Number of held-out test images |
| `--batch` | 64 | Maximum images per forward call; capped at samples |
| `--repeats` | 100 | Number of timed full-dataset passes |
| `--train-steps` | 20000 | One-image SGD updates before benchmarking |
| `--seed` | 42 | Seed for deterministic random streams |
| `--check` | off | Compare up to 16 outputs with a double reference and batch 1 |
| `--help` | | Print usage |

Integer sizes must be positive, except that zero training steps is supported.
Allocation failures or invalid arguments produce an error and nonzero exit code.
Use practical sizes from the examples rather than the parser's maximum limits.

Approximate main-buffer memory, including labels and assuming four-byte floats
and integers, is `4*(75*H + 75*samples + batch*H + 10)` bytes, with batch capped at samples.
Increasing repeats raises runtime without raising memory use. Increasing H raises
both training and inference cost. Start small before increasing parameters.

## How to read the results

- `training_seconds`: setup only, excluded from inference timing.
- `inference_total_seconds`: sum of forward-pass durations over all repeats.
- `mean_pass_seconds`, `best_pass_seconds`: full-test-set pass times.
- `images_per_second`: `samples * repeats / inference_total_seconds`.
- `amortized_us_per_image`: reciprocal throughput, not individual request latency.
- `accuracy`: fraction of held-out images classified correctly on the last pass.
- `cross_entropy`: mean `-log(probability of the correct digit)`; lower is better.
- `checksum`: weighted sum of all probabilities from the last pass. Useful for
  spotting changes; it is not a standalone proof of correctness.
- `reference_max_abs_error`: maximum absolute probability error on checked images;
  must be at most `1e-4`. Batch-one error must be at most `1e-6`.

Probabilities are validated after every repeat, outside the timer. Repetitions
reuse the same images and weights, modeling a warmed-up, in-memory workload.
Inference timing includes dense layers, ReLU, softmax, output writes, and batch
dispatch. It excludes input acquisition and training. Independent whole-program
runs capture variability better than repetitions within one process alone.

## Verify that the network learns

On Miyabi, compare the small correctness run with the same command using
`--train-steps 0`. The trained network should have substantially better accuracy
and lower loss. Random untrained models need not score exactly 10%, especially on
small test sets. The synthetic task is relatively easy; perfect accuracy is not
guaranteed because erasure can make distinct digits ambiguous.

Run the trained command again with batch 1 and then batch 64. For the same seed,
width, training steps and sample count, the baseline should produce the same
accuracy, loss and checksum. Repeat an identical command to check reproducibility.
Floating-point details can differ between compilers/platforms, so use numerical
tolerances when comparing implementations.

## Experiment plan

1. **Choose a useful duration.** Measure the default first. If its reported
   inference total is T seconds, estimate repeats for a 20-second benchmark as
   `ceil(100 * 20 / T)`. Change repeats only, and remeasure. This estimate excludes
   training, warm-up and checking, so leave ample walltime margin.
2. **Batch-size sweep.** Hold H=256, samples=4096, training steps=20000 and seed=42
   fixed; try batch sizes 1, 16, 64 and 256. The plain loops may show little batch
   advantage: optimized matrix multiplication has not been implemented yet.
3. **Width sweep.** Try H=128, 256, 512 and 1024. Each is a different model and
   workload. Record accuracy as well as time; larger does not guarantee better.
4. **Repeat measurements.** Use three independent jobs for each chosen setup and
   report the median throughput, keeping compiler and node type consistent.

Fill this with your own measurements; no Miyabi timings are supplied in advance:

| H | Samples | Batch | Repeats | Train steps | Seed | Mean pass (s) | Images/s | Accuracy | Reference error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4096 | 1 | | 20000 | 42 | | | | |
| 256 | 4096 | 16 | | 20000 | 42 | | | | |
| 256 | 4096 | 64 | | 20000 | 42 | | | | |
| 256 | 4096 | 256 | | 20000 | 42 | | | | |

Also record compiler/version/options and the compute-node type. A rough dense-layer
rate is `148 * H * images_per_second / 1e9` GFLOP/s; it omits activation and softmax
work, and should not be mistaken for a hardware-counter measurement.

## Later parallelization milestones

1. **CPU:** parallelize across images, providing private hidden buffers to workers.
   Measure scaling with worker count against the same serial workload.
2. **Locality/SIMD:** study loop ordering and matrix tiling for batches, followed by
   vectorization. Measure effects of width, batch size and cache capacity.
3. **GPU:** implement the two batched dense layers, ReLU and softmax. Report both
   device computation time and end-to-end inference time including transfers and
   synchronization. State whether weights remain resident across requests.

In every version preserve a serial oracle and check full probabilities, not just
predicted labels. Freeze the trained weights for any serial/optimized comparison;
keep training on the baseline path or pass the same trained network to both
forward implementations. Changed reduction order may require a justified
tolerance, but should not hide a correctness bug.

This workload studies dense classification inference. It does not include
transformer-specific attention, KV caches, or token-by-token generation, and its
speedups cannot directly predict LLM speedups.

## Submission

Read and revise `problem.md` in your own words and add your own measured results.
Ensure the notebook compile and run cells succeed as written. Before submitting,
remove generated executables, large outputs, and other unnecessary artifacts
yourself, retaining the source and needed documentation. All assignment files are
shared with classmates, so include only information you intend to share.

## Local verification performed during project creation

Compiled on taulec with `g++ -O2 -Wall -Wextra -std=c++17`, without warnings.
With H=64, 257 samples, 4000 training steps and seed 42, a tiny check achieved
accuracy 0.996109 and cross-entropy 0.091674. The same untrained network achieved
accuracy 0.241245 and cross-entropy 2.380495. Batch sizes 1, 17 and 64 produced
identical printed checksums (63273.659272); maximum double-reference error was
approximately 1.8e-7. A one-image/one-neuron case and invalid arguments were also
checked. These are local correctness observations, not Miyabi performance results.
The generated `inference` executable is a build artifact; recompile it on Miyabi
and remove it yourself before submission.
