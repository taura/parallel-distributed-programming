
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making a Mini Vector Database (IVF Index) from Scratch

# Problem

A *vector database* stores a large number of vectors (e.g., embeddings of text, images or audio produced by a neural network) and answers **nearest-neighbor queries**: given a query vector, return the $k$ stored vectors closest to it. This is the core of semantic search and of retrieval-augmented generation (RAG) for LLMs; systems such as FAISS and Milvus are built around it.

Formally, given $N$ database vectors $x_1, \dots, x_N \in \mathbb{R}^d$ and a query $q \in \mathbb{R}^d$, find the $k$ indices $i$ with the smallest squared Euclidean distance

$$ \|q - x_i\|^2 = \sum_{j=1}^{d} (q_j - x_{ij})^2 . $$

Scanning all $N$ vectors for every query (brute force) costs $O(Nd)$ per query, which is too slow for millions of vectors and many queries. Real systems therefore build an **index** and do *approximate* search: they look at only a small part of the data, and accept missing a few true neighbors.

This program implements the classic **IVF (inverted file) index**:

1. **Build**: cluster the vectors with k-means into `nlist` clusters. Each cluster has a centroid and an *inverted list* holding the vectors closest to that centroid.
2. **Search**: for a query, find the `nprobe` nearest centroids, then compute the distance only to the vectors in those `nprobe` lists, and keep the $k$ best.

With `nlist = 1024` and `nprobe = 16`, a query would scan $16/1024 \approx 1.6\%$ of the data if all lists had the same length; in our runs it scans about 5% (see the results below), because the lists are uneven and queries tend to fall into the large ones. `nprobe` trades speed against accuracy.

**How good is the answer?** We measure **recall@k**: for some queries, we also compute the exact answer by brute force, and count what fraction of the true $k$ nearest neighbors the index returned (1.0 = perfect).

The input is synthetic, generated inside the program from a fixed seed: vectors are drawn from a mixture of Gaussian "blobs" in $d$ dimensions (default $N = 10^6$, $d = 128$, 10,000 queries, similar in size to the standard SIFT1M benchmark).

# How the algorithm/baseline code works

The code is C++17, uses only the standard library, and is serial:

- `ivf.h`: the API (data generation, k-means, index build, search, recall), with comments
- `ivf.cc`: the implementation of the algorithm
- `main.cc`: command-line options, timing and output
- `run.sh`: builds and runs the program on Miyabi

All vectors are stored in flat `float` arrays (vector `i` is `X[i*d .. i*d+d)`).

- **Data generation**: `nblob` random centers in $[0,1]^d$; each vector is a random center plus Gaussian noise with standard deviation `sigma` (default 0.25) in every dimension. Queries are generated the same way. A larger `sigma` makes the clusters overlap, so the search gets harder.
- **k-means (`train_kmeans`)**: Lloyd's algorithm on a training sample of `ntrain` vectors, for `iters` iterations:
  - *assignment step*: for every training vector, compute the distance to all `nlist` centroids and pick the nearest;
  - *update step*: each centroid becomes the mean of the vectors assigned to it.

  The program prints the mean squared distance to the nearest centroid at each iteration; it should decrease.
- **Building the lists (`build_lists`)**: every one of the $N$ vectors is assigned to its nearest centroid. The vectors are then copied so that each list is stored contiguously (a CSR-like layout: list `c` is `offsets[c] .. offsets[c+1]`).
- **Search (`search`)**, for each query:
  1. compute the distance to all `nlist` centroids and keep the `nprobe` smallest;
  2. scan the vectors in those lists, keeping the $k$ smallest distances in a small sorted array (`TopK`).
- **Check (`brute_force`)**: the exact top-$k$ for the first `ngt` queries, used to compute recall@k.

The main computation (k-means + list building + search) is timed with `std::chrono`. The program also reports queries/s and GFLOP/s.

The inner loop of every step is the same distance kernel:

```cpp
float sum = 0;
for (j = 0; j < d; j++) { float diff = a[j] - b[j]; sum += diff * diff; }
```

Command line options: `-n N -d D -q Q -l NLIST -p NPROBE -k K -t NTRAIN -i ITERS -g NGT -b NBLOB -r SIGMA -s SEED`
(`-r` is the noise around each blob: larger values make the clusters overlap and the search harder).

Example: `./ivf -n 1000000 -d 128 -q 10000 -l 1024 -p 16`

## Results of the baseline on Miyabi

One Grace (Neoverse-V2) core, `clang++ -O3`, default parameters
($N = 10^6$, $d = 128$, 10,000 queries, `nlist = 1024`, `nprobe = 16`, `sigma = 0.25`):

| step | time |
|---|---|
| k-means training (50,000 vectors, 10 iterations) | 27.5 s |
| building the lists (assigning all $10^6$ vectors) | 55.6 s |
| search (10,000 queries) | 28.3 s (353 queries/s) |
| **main computation in total** | **111.4 s** |

- recall@10 = 0.994 (checked against brute force for 100 queries)
- the k-means objective decreases at every iteration (21.28 → 12.21)
- list lengths range from 1 to 6366 (average 977); a query scans about 50,900 vectors on average
- the search runs at only about 7 GFLOP/s, a small fraction of what one core can do, so there is a lot of room for optimization

# Where is parallelism

- **Between queries**: all queries are independent, so they can be split among CPU cores or GPU thread blocks.
- **Inside k-means and list building**: the nearest centroid of each vector is computed independently (data-parallel over $N$). The update step is a reduction (sums per cluster), which needs per-thread partial sums or atomics.
- **The distance kernel**: computing many distances at once is essentially a matrix multiplication, since
  $\|q-x\|^2 = \|q\|^2 - 2\,q\cdot x + \|x\|^2$. So it suits SIMD instructions, cache blocking, and GPU (even tensor cores).
- **Inside a query**: the lists (and the vectors within a list) can be scanned in parallel, each thread keeping its own top-$k$, merged at the end.

What may limit the speedup:

- Scanning the lists reads a lot of memory per query with little reuse, so search may be **memory-bandwidth bound**. Centroid distances and k-means are more **compute bound** (good for a roofline analysis).
- List lengths are uneven, which causes load imbalance.
- The top-$k$ selection has branches, which hurt on SIMD and GPU.
- The k-means update needs reductions or atomics.
