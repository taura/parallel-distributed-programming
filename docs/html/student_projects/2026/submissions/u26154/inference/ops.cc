#include "ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void LayerNorm(const std::vector<float>& x, int rows, int cols, const std::vector<float>& w,
               const std::vector<float>& b, float eps, std::vector<float>& y) {
  y.assign(static_cast<size_t>(rows) * cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[static_cast<size_t>(r) * cols];
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
      mean += xr[c];
    }
    mean /= static_cast<float>(cols);

    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      const float d = xr[c] - mean;
      var += d * d;
    }
    var /= static_cast<float>(cols);

    const float inv = 1.0f / std::sqrt(var + eps);
    float* yr = &y[static_cast<size_t>(r) * cols];
    for (int c = 0; c < cols; ++c) {
      yr[c] = (xr[c] - mean) * inv * w[c] + b[c];
    }
  }
}

void Linear(const std::vector<float>& x, int rows, int in_dim, int out_dim,
            const std::vector<float>& w, const std::vector<float>& b, std::vector<float>& y) {
  y.assign(static_cast<size_t>(rows) * out_dim, 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[static_cast<size_t>(r) * in_dim];
    float* yr = &y[static_cast<size_t>(r) * out_dim];
    for (int o = 0; o < out_dim; ++o) {
      const float* wr = &w[static_cast<size_t>(o) * in_dim];
      float s = b.empty() ? 0.0f : b[o];
      for (int i = 0; i < in_dim; ++i) {
        s += wr[i] * xr[i];
      }
      yr[o] = s;
    }
  }
}

float Gelu(float x) {
  const float inv_sqrt2 = 0.7071067811865475f;
  return 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
}

void ApplyGelu(std::vector<float>& x) {
  for (float& v : x) {
    v = Gelu(v);
  }
}

void AddInPlace(std::vector<float>& dst, const std::vector<float>& src) {
  if (dst.size() != src.size()) {
    throw std::runtime_error("AddInPlace size mismatch");
  }
  for (size_t i = 0; i < dst.size(); ++i) {
    dst[i] += src[i];
  }
}

bool HasNanOrInf(const std::vector<float>& x) {
  for (float v : x) {
    if (!std::isfinite(v)) {
      return true;
    }
  }
  return false;
}

int Argmax(const std::vector<float>& x) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(x.size()); ++i) {
    if (x[i] > x[best]) {
      best = i;
    }
  }
  return best;
}

double ChecksumPrefix(const std::vector<float>& x, int n) {
  double s = 0.0;
  const int m = std::min(n, static_cast<int>(x.size()));
  for (int i = 0; i < m; ++i) {
    s += static_cast<double>(x[i]) * (i + 1);
  }
  return s;
}

double ElapsedMs(std::chrono::steady_clock::time_point t0,
                 std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
