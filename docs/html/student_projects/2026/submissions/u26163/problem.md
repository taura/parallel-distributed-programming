# Making CKKS-style NTT Polynomial Multiplication from Scratch

# Problem

I implement a serial baseline for polynomial multiplication in
\(\mathbb{Z}_q[X] / (X^N + 1)\), which is the core operation used by CKKS
homomorphic encryption.

In CKKS, ciphertext and key operations repeatedly multiply polynomials modulo
several NTT-friendly primes (RNS representation). This baseline models that by
computing the same negacyclic polynomial product for multiple moduli.

The goal of this assignment is not full CKKS, but a realistic core kernel that
is performance-critical and has clear parallelization opportunities.

# How the baseline code works

File: `code/ckks_ntt_baseline.cc`

1. Parse command-line parameters:
   - `--logn` so that \(N = 2^{\text{logn}}\)
   - `--mods` number of RNS moduli
   - `--reps` repetition count for timing
   - `--seed` deterministic random seed
   - `--verify-coeffs` number of coefficients checked by a slow reference

2. Choose built-in NTT primes \(q\) such that \((q-1)\) is divisible by
   \(2N\), then build one NTT context per modulus.

3. For each modulus, generate random polynomials `a` and `b`.

4. Compute negacyclic product with NTT:
   - twist by powers of a primitive \(2N\)-th root
   - forward NTT of size \(N\)
   - pointwise multiply
   - inverse NTT
   - untwist

5. Time the main loop (`reps` times over all moduli) using
   `std::chrono::steady_clock`.

6. Verify correctness by comparing the first `verify-coeffs` coefficients with
   a direct \(O(N^2)\) negacyclic formula (for one modulus), and print:
   - elapsed compute time
   - mismatch count
   - checksum for reproducibility

# Where is parallelism

- **Across moduli (RNS):** each modulus is independent and can run in parallel.
- **Across NTT stages:** Within each NTT stage, independent butterfly operations can be executed in parallel. However synchronization is required between consecutive stages.
- **Across coefficients:** pointwise multiplication is embarrassingly parallel.
- **Across repeated multiplications (`--reps`):** repetitions can be batched.

Potential limits to speedup:

- memory bandwidth / cache behavior at large \(N\)
- synchronization between NTT stages
- load imbalance when modulus count is small
