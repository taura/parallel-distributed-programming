#pragma once

#include "math/vec3.h"

namespace config {

constexpr int kImageWidth = 800;
constexpr int kImageHeight = 450;

constexpr int kSamplesPerPixel = 4;  // 2x2 sampling
constexpr int kMaxDepth = 5;

constexpr double kShadowEpsilon = 1e-4;
constexpr double kRayEpsilon = 1e-4;

constexpr double kGamma = 2.2;

// Fallback/background colors (deterministic)
const Vec3 kBackgroundColor(0.70, 0.82, 1.00);
const Vec3 kDepthExhaustedColor(0.0, 0.0, 0.0);

constexpr const char* kOutputPath = "build/output.ppm";

}  // namespace config