// rng.h : small deterministic random number generator
// Each (pixel, sample) gets its own seed, so serial and parallel runs
// produce bit-identical images. Do not use rand() or std::random_device.
#pragma once
#include <cstdint>

// splitmix64: turns any 64-bit value into a well-mixed one
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// xorshift64* generator
struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed ? seed : 0x2545f4914f6cdd1dull) {}

  uint32_t next_u32() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 0x2545f4914f6cdd1dull) >> 32);
  }

  // uniform double in [0, 1)
  double uniform() { return next_u32() * (1.0 / 4294967296.0); }
};

// mix pixel index and sample index into a well-distributed seed,
// so neighbouring pixels are not correlated
inline uint64_t make_seed(uint64_t pixel, uint64_t sample) {
  return splitmix64(splitmix64(pixel) ^ (sample * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull));
}
