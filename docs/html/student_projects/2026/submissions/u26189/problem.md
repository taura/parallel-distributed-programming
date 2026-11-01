# FIR Decimation and Energy Detection on IQ Samples

## Problem

I chose a signal-processing task that works on IQ data.
IQ data has two values per sample: I (in-phase) and Q (quadrature).

The program does three common steps used in SDR:

1. low-pass FIR filtering,
2. decimation (keep one sample every D samples),
3. block-wise energy detection.

This is a realistic workflow in communication/radar/remote-sensing pipelines and is a good
target for later multicore, SIMD, and GPU optimization.

The program supports two input modes:

- real IQ file (`ci16` or `cf32`, interleaved I/Q), or
- deterministic synthetic QPSK + noise.

The workload is controlled by command-line options (`--samples`, `--taps`, `--decim`,
`--repeats`, ...), so runtime can be tuned from seconds to minutes.

For real-data experiments, I use the HMAP SigMF dataset (Zenodo record 7416081).
Its IQ payload (`*.sigmf-data`) has 8 files:

- 4 files around 4.596 GB each,
- 4 files around 3.116 GB each,
- about 30.85 GB total (about 28.7 GiB), plus small metadata files (`*.sigmf-meta`).

## How the baseline code works

The baseline implementation is `iq_baseline.cc` (C++17, standard library only).

Main steps:

1. **Input preparation**
   - If `--input` is given, load file data into two float arrays (`I[]`, `Q[]`).
   - Otherwise, generate synthetic signal with a fixed random seed.

2. **Filter design**
   - Build low-pass FIR coefficients (windowed-sinc + Hamming window) and normalize them.

3. **Main computation (timed)**
   - Repeat `--repeats` times:
     - FIR + decimation:
       \[
       y[m] = \sum_{k=0}^{T-1} h[k] x[mD + k]
       \]
       where `T` is taps and `D` is decimation factor.
     - Compute energy \(|y_I|^2 + |y_Q|^2\), total energy, checksum, and block-hit count.

4. **Result check**
   - Recompute the first `--check-outputs` results with double precision,
     then report max absolute error and root-mean-square error.

Printed outputs include main compute time, approximate FIR throughput (GFLOP/s), energy,
checksum, and error metrics.

## Baseline and improved versions

I implemented two versions of the same algorithm:

- `iq_baseline.cc`: serial baseline (plain loops, no OpenMP).
- `iq_improved.cc`: OpenMP-parallel version for multicore experiments.



## Where is parallelism

The most expensive part is FIR + decimation. Each output index is mostly independent, so it
maps well to parallel hardware.

- **Multicore CPU**: split output indices across threads.
- **SIMD**: vectorize multiply-add operations in FIR loops.
- **GPU**: map outputs to many threads and reuse input/tap data efficiently.

## Data

Download the data used from the following command lines in code/data:
```bash
# Small Zenodo SigMF set
curl -L "https://zenodo.org/api/records/13143544/files/crab-giantpulse.sigmf-data/content" -o code/data/crab-giantpulse.sigmf-data
curl -L "https://zenodo.org/api/records/13143544/files/crab-giantpulse.sigmf-meta/content" -o code/data/crab-giantpulse.sigmf-meta

# Medium size
curl -L "https://zenodo.org/api/records/7416081/files/HMAP_2022-11-16T15:16:06_1ax.sigmf-data/content" -o code/data/HMAP_2022-11-16T15:16:06_1ax.sigmf-data
curl -L "https://zenodo.org/api/records/7416081/files/HMAP_2022-11-16T15:16:06_1ax.sigmf-meta/content" -o code/data/HMAP_2022-11-16T15:16:06_1ax.sigmf-meta
```