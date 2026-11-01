// render.h : Monte Carlo path tracing
#pragma once
#include "scene.h"
#include <cstdint>
#include <vector>

struct RenderParams {
  int width = 256;
  int height = 256;
  int spp = 16;        // samples per pixel
  int max_depth = 8;   // maximum number of bounces
  int rr_depth = 3;    // start Russian roulette after this many bounces
};

// Radiance arriving along ray r (one path sample).
// `rays` is incremented by the number of rays cast (for Mrays/s statistics).
Vec3 trace_path(const Scene& scene, Ray r, const RenderParams& p, Rng& rng, uint64_t& rays);

// Render the full image; returns linear RGB, row-major (top row first),
// size width*height. `rays` returns the total number of rays cast.
// This loop over pixels/samples is the main target for parallelization.
std::vector<Vec3> render(const Scene& scene, const RenderParams& p, uint64_t& rays);
