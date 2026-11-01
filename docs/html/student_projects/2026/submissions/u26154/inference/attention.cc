#include "attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "ops.h"

namespace {

void ApplyRotary(std::vector<float>& q, std::vector<float>& k, int seq_len, int num_heads,
                 int head_dim, int rotary_dims, double base) {
  if (rotary_dims <= 0) {
    return;
  }
  if (rotary_dims % 2 != 0) {
    throw std::runtime_error("rotary_dims must be even");
  }
  const int half = rotary_dims / 2;
  std::vector<double> inv_freq(half);
  for (int i = 0; i < half; ++i) {
    inv_freq[i] = std::pow(base, -2.0 * static_cast<double>(i) / static_cast<double>(rotary_dims));
  }
  for (int t = 0; t < seq_len; ++t) {
    for (int h = 0; h < num_heads; ++h) {
      float* qv = &q[(static_cast<size_t>(t) * num_heads + h) * head_dim];
      float* kv = &k[(static_cast<size_t>(t) * num_heads + h) * head_dim];
      for (int i = 0; i < half; ++i) {
        const double theta = static_cast<double>(t) * inv_freq[i];
        const float ct = static_cast<float>(std::cos(theta));
        const float st = static_cast<float>(std::sin(theta));
        const int i0 = i;
        const int i1 = i + half;

        const float q0 = qv[i0];
        const float q1 = qv[i1];
        qv[i0] = q0 * ct - q1 * st;
        qv[i1] = q0 * st + q1 * ct;

        const float k0 = kv[i0];
        const float k1 = kv[i1];
        kv[i0] = k0 * ct - k1 * st;
        kv[i1] = k0 * st + k1 * ct;
      }
    }
  }
}

}  // namespace

void SelfAttention(const std::vector<float>& ln_x, const LayerWeights& w, const ModelConfig& cfg,
                   VerifyStats& verify, Timers& timers, std::vector<float>& out) {
  const int t_len = static_cast<int>(ln_x.size()) / cfg.hidden_size;
  const int hidden = cfg.hidden_size;
  const int num_heads = cfg.num_attention_heads;
  const int head_dim = hidden / num_heads;
  int rotary_dims = static_cast<int>(head_dim * cfg.rotary_pct);
  rotary_dims = std::max(0, std::min(rotary_dims, head_dim));
  if (rotary_dims % 2 != 0) {
    --rotary_dims;
  }

  auto t0 = std::chrono::steady_clock::now();
  std::vector<float> qkv;
  Linear(ln_x, t_len, hidden, 3 * hidden, w.qkv_w, w.qkv_b, qkv);
  auto t1 = std::chrono::steady_clock::now();
  timers.qkv_ms += ElapsedMs(t0, t1);

  std::vector<float> q(static_cast<size_t>(t_len) * num_heads * head_dim);
  std::vector<float> k(static_cast<size_t>(t_len) * num_heads * head_dim);
  std::vector<float> v(static_cast<size_t>(t_len) * num_heads * head_dim);

  for (int t = 0; t < t_len; ++t) {
    const float* src = &qkv[static_cast<size_t>(t) * 3 * hidden];
    for (int h = 0; h < num_heads; ++h) {
      const int base = h * 3 * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        const size_t idx = (static_cast<size_t>(t) * num_heads + h) * head_dim + d;
        q[idx] = src[base + d];
        k[idx] = src[base + head_dim + d];
        v[idx] = src[base + 2 * head_dim + d];
      }
    }
  }

  ApplyRotary(q, k, t_len, num_heads, head_dim, rotary_dims, cfg.rotary_emb_base);

  t0 = std::chrono::steady_clock::now();
  std::vector<float> attn_ctx(static_cast<size_t>(t_len) * num_heads * head_dim, 0.0f);
  std::vector<float> scores(t_len);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int h = 0; h < num_heads; ++h) {
    for (int t = 0; t < t_len; ++t) {
      float max_s = -std::numeric_limits<float>::infinity();
      for (int s = 0; s <= t; ++s) {
        float dot = 0.0f;
        const float* qv = &q[(static_cast<size_t>(t) * num_heads + h) * head_dim];
        const float* kv = &k[(static_cast<size_t>(s) * num_heads + h) * head_dim];
        for (int d = 0; d < head_dim; ++d) {
          dot += qv[d] * kv[d];
        }
        scores[s] = dot * scale;
        if (scores[s] > max_s) {
          max_s = scores[s];
        }
      }

      float sum = 0.0f;
      for (int s = 0; s <= t; ++s) {
        scores[s] = std::exp(scores[s] - max_s);
        sum += scores[s];
      }
      const float inv_sum = 1.0f / sum;
      double row_sum = 0.0;
      for (int s = 0; s <= t; ++s) {
        scores[s] *= inv_sum;
        row_sum += scores[s];
      }
      verify.max_softmax_row_sum_error =
          std::max(verify.max_softmax_row_sum_error, std::abs(row_sum - 1.0));

      float* outv = &attn_ctx[(static_cast<size_t>(t) * num_heads + h) * head_dim];
      for (int d = 0; d < head_dim; ++d) {
        float val = 0.0f;
        for (int s = 0; s <= t; ++s) {
          const float* vv = &v[(static_cast<size_t>(s) * num_heads + h) * head_dim];
          val += scores[s] * vv[d];
        }
        outv[d] = val;
      }
    }
  }
  t1 = std::chrono::steady_clock::now();
  timers.attn_ms += ElapsedMs(t0, t1);

  std::vector<float> concat(static_cast<size_t>(t_len) * hidden);
  for (int t = 0; t < t_len; ++t) {
    float* dst = &concat[static_cast<size_t>(t) * hidden];
    for (int h = 0; h < num_heads; ++h) {
      const float* src = &attn_ctx[(static_cast<size_t>(t) * num_heads + h) * head_dim];
      for (int d = 0; d < head_dim; ++d) {
        dst[h * head_dim + d] = src[d];
      }
    }
  }

  t0 = std::chrono::steady_clock::now();
  Linear(concat, t_len, hidden, hidden, w.attn_out_w, w.attn_out_b, out);
  t1 = std::chrono::steady_clock::now();
  timers.attn_proj_ms += ElapsedMs(t0, t1);
}
