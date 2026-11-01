# Making CKKS Encryption from Scratch

## Problem

CKKS is a homomorphic encryption scheme that supports approximate computations on encrypted real and complex numbers. 
It relies on polynomial arithmetic in the ring

$$
R_q = \mathbb{Z}_q[X]/(X^N+1).
$$

Polynomial multiplication is one of the main computational operations in CKKS. 
A naive implementation requires $O(N^2)$ operations, while the Number Theoretic Transform (NTT) reduces the complexity to $O(N\log N)$.

## How the algorithm/baseline code works

The baseline is a serial C++ implementation consisting of three main components:

1. NTT: Implements the forward and inverse NTT using an iterative Cooley–Tukey algorithm.
2. Polynomial multiplication: Performs negacyclic convolution using NTT, pointwise multiplication, and inverse NTT.
3. CKKS: Implements approximate encoding/decoding and RLWE-based key generation, encryption, and decryption.

For polynomial multiplication, the baseline performs the following steps:

1. Multiply polynomial coefficients by powers of a primitive $2N$-th root of unity (twisting).
2. Apply the forward NTT to both polynomials.
3. Multiply the transformed coefficients elementwise.
4. Apply the inverse NTT.
5. Remove the twisting factors.

Basically, the NTT transforms polynomial multiplication into pointwise multiplication in the NTT domain, making the computation more efficient.
I also added some tests to check the correctness of NTT inversion, polynomial multiplication, and approximate recovery of encrypted plaintexts.

## Where is parallelism

The main target for parallelization is the NTT.

* Butterfly operations: Within each NTT stage, butterfly operations are independent and can be executed in parallel.
* Pointwise multiplication: Each coefficient multiplication is independent.
* Polynomial multiplication: The forward NTTs of the two input polynomials can be executed independently.

The main challenge is that NTT stages have data dependencies, requiring synchronization between stages. 
