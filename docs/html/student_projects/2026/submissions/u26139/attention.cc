#include "attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<int64_t> shape_) : shape(std::move(shape_)) {
  if (shape.empty()) {
    throw std::runtime_error("shape must not be empty");
  }
  strides.resize(shape.size());
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[static_cast<size_t>(i)] <= 0) {
      throw std::runtime_error("shape values must be positive");
    }
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  data.assign(static_cast<size_t>(stride), 0.0);
}

int64_t Tensor::dim() const { return static_cast<int64_t>(shape.size()); }

double &Tensor::at2(int64_t i, int64_t j) {
  return data[static_cast<size_t>(i * strides[0] + j * strides[1])];
}

const double &Tensor::at2(int64_t i, int64_t j) const {
  return data[static_cast<size_t>(i * strides[0] + j * strides[1])];
}

namespace {

void matmul_impl(double *c, const double *a, const double *b, int64_t m, int64_t n,
                 int64_t k, int64_t c_rs, int64_t c_cs, int64_t a_rs,
                 int64_t a_cs, int64_t b_rs, int64_t b_cs) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int64_t p = 0; p < k; ++p) {
        sum += a[i * a_rs + p * a_cs] * b[p * b_rs + j * b_cs];
      }
      c[i * c_rs + j * c_cs] = sum;
    }
  }
}

void fill_random_impl(double *dst, int64_t n, std::mt19937 &rng, double lo,
                      double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = dist(rng);
  }
}

void scale_inplace_impl(double *x, int64_t n, double alpha) {
  for (int64_t i = 0; i < n; ++i) {
    x[i] *= alpha;
  }
}

void softmax_rows_inplace_impl(double *x, int64_t rows, int64_t cols,
                               int64_t row_stride, int64_t col_stride) {
  for (int64_t i = 0; i < rows; ++i) {
    double row_max = x[i * row_stride];
    for (int64_t j = 1; j < cols; ++j) {
      row_max = std::max(row_max, x[i * row_stride + j * col_stride]);
    }

    double sum = 0.0;
    for (int64_t j = 0; j < cols; ++j) {
      const int64_t idx = i * row_stride + j * col_stride;
      const double e = std::exp(x[idx] - row_max);
      x[idx] = e;
      sum += e;
    }

    const double inv = 1.0 / sum;
    for (int64_t j = 0; j < cols; ++j) {
      x[i * row_stride + j * col_stride] *= inv;
    }
  }
}

double checksum_impl(const double *x, int64_t n) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    s += x[i];
  }
  return s;
}

void attention_impl(double *out, const double *q, const double *k,
                    const double *v, int64_t seq_len, int64_t dim,
                    int64_t num_heads, int64_t out_rs, int64_t out_cs,
                    int64_t q_rs, int64_t q_cs, int64_t k_rs, int64_t k_cs,
                    int64_t v_rs, int64_t v_cs, double *head0_row0_sum) {
  const int64_t head_dim = dim / num_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  Tensor scores({seq_len, seq_len});
  for (int64_t h = 0; h < num_heads; ++h) {
    const int64_t off = h * head_dim;
    const double *qh = q + off * q_cs;
    const double *kh = k + off * k_cs;
    const double *vh = v + off * v_cs;
    double *outh = out + off * out_cs;

    matmul_impl(scores.data.data(), qh, kh, seq_len, seq_len, head_dim,
                scores.strides[0], scores.strides[1], q_rs, q_cs, k_cs, k_rs);
    scale_inplace_impl(scores.data.data(),
                       static_cast<int64_t>(scores.data.size()), scale);
    softmax_rows_inplace_impl(scores.data.data(), seq_len, seq_len,
                              scores.strides[0], scores.strides[1]);

    if (head0_row0_sum != nullptr && h == 0) {
      double s = 0.0;
      for (int64_t j = 0; j < seq_len; ++j) {
        s += scores.data[static_cast<size_t>(j)];
      }
      *head0_row0_sum = s;
    }

    matmul_impl(outh, scores.data.data(), vh, seq_len, head_dim, seq_len, out_rs,
                out_cs, scores.strides[0], scores.strides[1], v_rs, v_cs);
  }
}

}  // namespace

Tensor matmul(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b) {
  if (a.dim() != 2 || b.dim() != 2) {
    throw std::runtime_error("matmul currently supports only 2D tensors");
  }

  const int64_t a_rows = trans_a ? a.shape[1] : a.shape[0];
  const int64_t a_cols = trans_a ? a.shape[0] : a.shape[1];
  const int64_t b_rows = trans_b ? b.shape[1] : b.shape[0];
  const int64_t b_cols = trans_b ? b.shape[0] : b.shape[1];

  if (a_cols != b_rows) {
    throw std::runtime_error("matmul shape mismatch");
  }

  Tensor c({a_rows, b_cols});
  const int64_t a_rs = trans_a ? a.strides[1] : a.strides[0];
  const int64_t a_cs = trans_a ? a.strides[0] : a.strides[1];
  const int64_t b_rs = trans_b ? b.strides[1] : b.strides[0];
  const int64_t b_cs = trans_b ? b.strides[0] : b.strides[1];

  matmul_impl(c.data.data(), a.data.data(), b.data.data(), a_rows, b_cols, a_cols,
              c.strides[0], c.strides[1], a_rs, a_cs, b_rs, b_cs);
  return c;
}

void fill_random(Tensor &t, std::mt19937 &rng, double lo, double hi) {
  fill_random_impl(t.data.data(), static_cast<int64_t>(t.data.size()), rng, lo,
                   hi);
}

void scale_inplace(Tensor &t, double alpha) {
  scale_inplace_impl(t.data.data(), static_cast<int64_t>(t.data.size()), alpha);
}

void softmax_rows_inplace(Tensor &t) {
  if (t.dim() != 2) {
    throw std::runtime_error("softmax_rows_inplace expects 2D tensor");
  }
  softmax_rows_inplace_impl(t.data.data(), t.shape[0], t.shape[1], t.strides[0],
                            t.strides[1]);
}

double checksum(const Tensor &t) {
  return checksum_impl(t.data.data(), static_cast<int64_t>(t.data.size()));
}

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v,
                 int64_t num_heads, double *head0_row0_sum) {
  if (q.dim() != 2 || k.dim() != 2 || v.dim() != 2) {
    throw std::runtime_error("attention expects 2D q, k, v");
  }
  if (q.shape != k.shape || q.shape != v.shape) {
    throw std::runtime_error("q, k, v must have the same shape");
  }
  if (num_heads <= 0) {
    throw std::runtime_error("num_heads must be positive");
  }

  const int64_t seq_len = q.shape[0];
  const int64_t dim = q.shape[1];
  if (dim % num_heads != 0) {
    throw std::runtime_error("dim must be divisible by num_heads");
  }

  Tensor out({seq_len, dim});
  attention_impl(out.data.data(), q.data.data(), k.data.data(), v.data.data(),
                 seq_len, dim, num_heads, out.strides[0], out.strides[1],
                 q.strides[0], q.strides[1], k.strides[0], k.strides[1],
                 v.strides[0], v.strides[1], head0_row0_sum);
  return out;
}
