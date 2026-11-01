
# Trajectory analysis from scratch: MSD and diffusion

## Problem

I chose **trajectory analysis** for 3D particle motion. The baseline computes
the **mean-squared displacement (MSD)** as a function of lag time and estimates
the diffusion coefficient from the linear MSD regime.

Given unwrapped trajectories \(\mathbf{r}_p(t)\) (particle index \(p\), frame
index \(t\)), the MSD at lag \(\ell\) is

\[
\mathrm{MSD}(\ell)=\frac{1}{N(T-\ell)}
\sum_{p=0}^{N-1}\sum_{t_0=0}^{T-\ell-1}
\left\lVert\mathbf{r}_p(t_0+\ell)-\mathbf{r}_p(t_0)\right\rVert^2,
\]

where \(N\) is number of particles and \(T\) is number of frames.

For ideal 3D Brownian motion, \(\mathrm{MSD}(t)=6Dt\), so if we fit
\(\mathrm{MSD}(t)\approx a t+b\) on a user-chosen interval,
\(D = a/6\).

### Units and assumptions

- Positions are in arbitrary length units.
- Frame spacing `dt` is in arbitrary time units.
- MSD is in length\(^2\) units.
- Estimated diffusion `D` is length\(^2\)/time.
- Input trajectories are **unwrapped** (no periodic jump correction is done).

### Why this is a good baseline for this course

- Real analysis workflow used in molecular and particle simulations.
- Naturally scalable: particles, frames, and max lag can make serial time from
  seconds to minutes.
- Has easy correctness checks (known synthetic motions and Brownian expectation).
- Clear data-parallel structure for future multicore/GPU work.

## How the baseline code works

File: `code/msd_diffusion.cc` (self-contained C++17, standard library only).

1. Parse command-line parameters:
   - `--particles`, `--frames`, `--max-lag`, `--dt`, `--diffusion`, `--seed`
   - `--fit-start`, `--fit-end`, `--csv`, `--run-checks`
2. Validate input:
   - positive sizes, `max_lag < frames`, valid fit interval, nonnegative diffusion
3. Generate synthetic unwrapped 3D Brownian trajectories:
   - independent Gaussian increments per axis with
     \(\sigma=\sqrt{2D\,dt}\)
4. Compute time-origin/particle-averaged MSD for lags `1..max_lag`:
   - direct triple-loop accumulation over lag, particle, and time origin
5. Fit MSD vs time in `[fit_start, fit_end]` by ordinary least squares:
   - report slope/intercept and `D_est = slope/6`
6. Write reproducible CSV:
   - columns: `lag,time,msd,count,expected_brownian_msd`
   - deterministic for a fixed seed on the same executable/standard-library
     implementation (the exact normal RNG stream can differ across implementations)
7. Print timing:
   - trajectory generation time
   - MSD main computation time
   - CSV+fit time and total time

### Numerical checks included

The program can run built-in checks (`--run-checks 1`, default):

1. **Stationary motion** (`r(t)=const`): MSD should be zero.
2. **Constant velocity** (`r(t)=v t`): MSD should equal \(|v|^2(\ell dt)^2\).
3. **Brownian expectation**: mean relative error vs \(6Dt\) over lags should be
   within a tolerance (finite-sample variability is expected).

These checks are deterministic (fixed seeds and fixed parameters).

### Complexity

- Trajectory generation: \(O(NT)\)
- MSD accumulation: \(O(NTL_{\max})\) (dominant)
- Memory for trajectory array: \(O(NT)\)

This intentionally plain serial implementation is useful as a baseline before
parallel optimization.

## Build and run

In this course notebook workflow:

- compile with `%%bash` (runs on a login node; keep this short),
- initialize scheduler magic once with `%load_ext miyabi`,
- run measurements with `%%sub` (batch job on a compute node).

Example cells (`cd code` first):

```bash
# compile cell
%%bash
cd code
module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
clang++ -O3 -Wall -std=c++17 msd_diffusion.cc -o msd_diffusion
```

```bash
# one-time initialization cell
%load_ext miyabi
```

```bash
# run cell (compute node)
%%sub
cd code
./msd_diffusion --particles 2000 --frames 2000 --max-lag 400 --dt 0.01 --diffusion 1.0 --seed 12345 --fit-start 50 --fit-end 200 --csv msd.csv --run-checks 1
```

For quick tests, reduce particles/frames/lags. For heavier serial runs, increase
them but keep total runtime within assignment limits.

## Verified run on Miyabi

The following numbers are from an actual notebook `%%sub` run on compute node
`mg1098` (`lecture-mig`, job `3750947`) with:

- `--particles 2000 --frames 2000 --max-lag 400 --dt 0.01`
- `--diffusion 1 --seed 12345 --fit-start 50 --fit-end 200`

Observed checks and outputs:

- stationary check: `max_abs_msd = 0`, PASS
- constant-velocity check: `max_relative_error = 1.981250e-13`, PASS (`tol 1e-12`)
- Brownian expectation check: `mean_abs_relative_error = 0.009419749`, PASS (`tol 0.12`)
- timing (s): generation `2.698377`, MSD `1.988603`, CSV+fit `0.001778412`, total `4.688758`
- fit results: slope `5.960500`, intercept `0.02969518`, `D_est = 0.9934167`, relative error `0.006583311`
- CSV output: `msd.csv` has `402` lines (header + lag `0..400`)

The estimated diffusion is statistically consistent with the Brownian target in
this finite-sample run: `D_est` is close to `D_true = 1`, and MSD-vs-time remains
near the expected linear trend.

## Limitations

- Synthetic data only in this baseline (no external trajectory parser yet).
- Assumes unwrapped trajectories; wrapped periodic trajectories require separate
  unwrapping logic.
- Brownian MSD is linear only in expectation; finite trajectories and finite
  particles produce noisy MSD and fit uncertainty.
- Diffusion estimate depends on fit interval choice.

## Where parallelism is

Major parallel opportunities:

1. **Across particles**: MSD contributions are independent per particle.
2. **Across time origins** for each lag: independent squared-displacement terms.
3. **Across lags**: each lag can be computed independently.

Likely future mappings:

- Multicore CPU: OpenMP over particles/lags with reduction.
- GPU: map displacement calculations to many threads; reduction per lag.

Potential speedup limits:

- Memory bandwidth and cache behavior for trajectory reads.
- Reduction overhead and load imbalance for large lag values.
- Host-device transfer overhead (for GPU variants).

---

I should proofread and rewrite this in my own words before submission, because
classmates will read it and I am responsible for the final text.
