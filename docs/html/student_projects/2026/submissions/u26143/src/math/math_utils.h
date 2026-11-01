#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace math {

// Numerical constants.
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-6;

// Angle conversion.
inline constexpr double degrees_to_radians(double deg) {
    return deg * kPi / 180.0;
}

// Clamp helper for safe tone mapping / color conversion.
inline constexpr double clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : (x > hi ? hi : x);
}

// Simple deterministic hash-based RNG utility (thread-safe if state is local).
// Use one state per pixel/sample path, never a shared global mutable state.
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

// Uniform random in [0, 1).
inline double rand01(uint64_t& state) {
    // Take top 53 bits for IEEE double mantissa precision.
    const uint64_t r = splitmix64(state);
    return static_cast<double>((r >> 11U)) * (1.0 / 9007199254740992.0);
}

}  // namespace math