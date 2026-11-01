#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<double> data;

  Tensor();
  explicit Tensor(std::vector<int64_t> shape_);

  int64_t dim() const;
  double &at2(int64_t i, int64_t j);
  const double &at2(int64_t i, int64_t j) const;
};

Tensor matmul(const Tensor &a, const Tensor &b, bool trans_a = false,
              bool trans_b = false);

void fill_random(Tensor &t, std::mt19937 &rng, double lo, double hi);
void scale_inplace(Tensor &t, double alpha);
void softmax_rows_inplace(Tensor &t);
double checksum(const Tensor &t);

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v,
                 int64_t num_heads, double *head0_row0_sum = nullptr);
