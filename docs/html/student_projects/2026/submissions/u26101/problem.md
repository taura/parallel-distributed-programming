# Clustering-Routed Sparse Attention (Serial Baseline)

## Problem

Dense attention compares every Query with all earlier Keys, requiring about N²/2 comparisons for N tokens. Here, k-means routes each Query to a cluster, and attention reads only earlier Keys in that cluster. The idea is inspired by the clustering-based sparse attention of the Routing Transformer.

The program generates a sequence with topics of unequal frequency and noisy, overlapping feature vectors. Each Value contains a topic label and a numerical payload. Topic imbalance is important for performance: if one cluster collects many tokens, its Queries do much more work than those in small clusters. Noise changes cluster assignments, attention work, and retrieval quality. This is a synthetic information-retrieval experiment, not a pretrained language model.

## How the baseline code works

`cluster_attention.cc` runs Lloyd's k-means on a separate calibration sequence, then processes the target sequence in mini-batches. Centers stay fixed while a batch is routed and attended to; the batch updates centers only after attention, so routing cannot use future target tokens. Previous Key assignments remain fixed.

For each Query, the program gathers earlier Keys in its cluster and computes scaled dot-product attention over their Values. It uses tiled online softmax, storing no N×N score matrix. Queries without an earlier Key in their cluster are omitted from retrieval accuracy. A dense run uses the same attention calculation but considers all earlier Keys. The program reports k-means/routing/attention times, numbers of sparse and dense comparisons, retrieval accuracy and its evaluated-Query count, and error against a separate two-pass softmax calculation.

In the notebook's compile cell, after the provided module setup:

```sh
cd code
clang++ -O3 -Wall -Wextra -std=c++17 cluster_attention.cc -o cluster_attention
```

In its `%%sub` run cell:

```sh
cd code
./cluster_attention 4096 32 8 12 64 1 42 1 256 0.9 1.3
```

Arguments: `tokens dim clusters iterations tile passes seed compare_dense batch noise skew` (all optional). `compare_dense=0` skips the expensive dense run. Increase `tokens` to scale the workload; vary `skew` and `noise` to examine how input distribution changes cluster sizes, work balance, and accuracy. Check the runtime on Miyabi before choosing large inputs.

## Where is parallelism?

- k-means assignments and Query computations can run independently across tokens. Centroid updates scatter each vector into label-indexed sums and counts, then reduce those accumulators within the batch before processing the next one. Parallel updates need to handle irregular writes and possible contention on popular clusters.
- Routing performs label-dependent writes, and sparse attention follows irregular Key/Value indices. CPU threads or GPU work groups could process Queries in parallel, but uneven cluster sizes cause load imbalance.
- With roughly balanced clusters, sparse attention examines about N²/(2C) pairs rather than N²/2. Skewed clusters increase the pair count and create long-running Queries. Tiled softmax also requires combining partial results across tiles.
