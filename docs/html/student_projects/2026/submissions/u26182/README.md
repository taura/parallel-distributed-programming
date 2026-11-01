# Transformer inference from scratch

The selected project is a serial, self-contained decoder-only Transformer:
causal multi-head attention, pre-LayerNorm blocks, GELU feed-forward layers,
residual connections, and greedy autoregressive generation.

- **`transformer.cc`** — C++17 implementation and correctness checks.
- **[`TRANSFORMER.md`](TRANSFORMER.md)** — Miyabi build/run commands, parameters,
  timing definitions, experiments and later optimization milestones.
- **[`problem.md`](problem.md)** — draft assignment description. Read, verify and
  rewrite it in your own words, adding your measurements.

Weights are deterministic random values and inputs are synthetic token IDs. This
benchmarks Transformer computation; it does not claim meaningful language output.
The baseline recomputes the prefix at each generation step. KV caching and
CPU/GPU parallelization are follow-on experiments.

The earlier classifier files (`inference.cc`, `inference`, `CLASSIFIER.md`) are
still present. Remove those yourself before submitting the Transformer project,
along with generated executables and other unnecessary artifacts. Compile the
specific source `transformer.cc`, not all `.cc` files together.
