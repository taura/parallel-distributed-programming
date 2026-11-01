
* _this is an example file you should finish_
* write problem description, how the algorithm/baseline code works, and where you think you can apply parallel computing
* if you are looking at this file from Jupyter, you can see the preview by right-clicking on the cell and "Show Markdown Preview"

# Making Wavelet Matrix from scratch

# Problem

A Wavelet Matrix is a succinct data structure for storing a sequence of integers. It supports efficient queries such as access, rank, and select.

Given an array of $n$ non-negative integers, the goal is to construct a Wavelet Matrix efficiently.

The construction processes the integers bit by bit, from the most significant bit to the least significant bit. 

For each bit position, it builds a bit vector and performs a stable partition of the array.

The sequential construction takes $O(n \log \sigma)$ time, where $\sigma$ is the size of the value domain.


# How the algorithm/baseline code works

The baseline is implemented in C++20 without external libraries.

For each bit level, the algorithm:

1. Extracts the corresponding bit from every integer and stores it in a bit vector.
2. Builds a rank data structure for the bit vector.
3. Counts the number of zeros.
4. Performs a stable partition, placing elements with bit 0 before elements with bit 1 while preserving their relative order.

Repeats the process for the next bit level.

# Where is parallelism

Several parts of the construction can be parallelized:

- Bit extraction: Each element's bit can be extracted independently.
- Zero counting: The number of zero bits can be calculated using reduction.
- Prefix sum / rank construction: Parallel prefix sum can compute the rank information and destination indices for each element.
- Stable partition: After destination indices being computed, elements can be written to their new positions in parallel without conflicts.
