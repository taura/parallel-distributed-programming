
OpenMP

Parallelize outer loop over start offsets i.
Each thread scans a chunk and stores local hits, then merge.
Main limits: memory bandwidth, merge overhead, cache behavior at boundaries.
SIMD

Replace inner byte-compare loop with vector compares (e.g., 16/32 bytes at once).
Big speedup for longer patterns or many candidate positions.
Main limits: alignment, remainder handling, and pattern length variability.
GPU

Assign one thread (or warp) per candidate offset; huge parallelism for large buffers.
Good when input is very large or many patterns must be scanned.
Main limits: PCIe/HBM transfer cost, warp divergence, global memory access patterns.