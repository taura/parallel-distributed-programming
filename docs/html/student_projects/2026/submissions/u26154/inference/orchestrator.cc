#include "orchestrator.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "attention.h"
#include "mlp_block.h"
#include "ops.h"

namespace {

int SampleFromLogits(const std::vector<float>& logits, const std::vector<int>& history,
                     float temperature, int top_k, float repetition_penalty,
                     std::mt19937& rng) {
  std::vector<float> adjusted = logits;

  if (repetition_penalty > 1.0f) {
    std::unordered_set<int> seen(history.begin(), history.end());
    for (int id : seen) {
      if (id < 0 || id >= static_cast<int>(adjusted.size())) {
        continue;
      }
      if (adjusted[id] > 0.0f) {
        adjusted[id] /= repetition_penalty;
      } else {
        adjusted[id] *= repetition_penalty;
      }
    }
  }

  if (temperature <= 0.0f) {
    return Argmax(adjusted);
  }

  std::vector<int> idx(adjusted.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    idx[i] = i;
  }
  const int k = (top_k > 0) ? std::min(top_k, static_cast<int>(idx.size()))
                            : static_cast<int>(idx.size());
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return adjusted[a] > adjusted[b]; });

  float max_logit = -1e30f;
  for (int i = 0; i < k; ++i) {
    max_logit = std::max(max_logit, adjusted[idx[i]] / temperature);
  }

  std::vector<double> probs(k, 0.0);
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    const double p = std::exp(static_cast<double>(adjusted[idx[i]] / temperature - max_logit));
    probs[i] = p;
    sum += p;
  }
  if (sum <= 0.0) {
    return idx[0];
  }
  for (double& p : probs) {
    p /= sum;
  }

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  const int pick = dist(rng);
  return idx[pick];
}

}  // namespace

RunResult RunOne(const ModelWeights& model, const std::vector<int>& prompt, int gen_len,
                 int layers_to_run, int seed, float temperature, int top_k,
                 float repetition_penalty) {
  RunResult rr;
  const ModelConfig& cfg = model.cfg;
  const int hidden = cfg.hidden_size;
  std::mt19937 rng(static_cast<uint32_t>(seed));

  std::vector<int> tokens = prompt;
  rr.generated_tokens = prompt;

  auto total_begin = std::chrono::steady_clock::now();

  for (int step = 0; step < gen_len; ++step) {
    const int t_len = static_cast<int>(tokens.size());
    if (t_len > cfg.max_position_embeddings) {
      throw std::runtime_error("Sequence length exceeds max_position_embeddings");
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> hidden_states(static_cast<size_t>(t_len) * hidden);
    for (int t = 0; t < t_len; ++t) {
      const int id = tokens[t];
      const float* src = &model.embed_in[static_cast<size_t>(id) * hidden];
      float* dst = &hidden_states[static_cast<size_t>(t) * hidden];
      for (int i = 0; i < hidden; ++i) {
        dst[i] = src[i];
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    rr.timers.embed_ms += ElapsedMs(t0, t1);

    const int active_layers =
        (layers_to_run < 0) ? cfg.num_hidden_layers : std::min(layers_to_run, cfg.num_hidden_layers);

    for (int l = 0; l < active_layers; ++l) {
      const LayerWeights& w = model.layers[l];
      std::vector<float> ln1;
      t0 = std::chrono::steady_clock::now();
      LayerNorm(hidden_states, t_len, hidden, w.input_ln_w, w.input_ln_b,
                static_cast<float>(cfg.layer_norm_eps), ln1);
      t1 = std::chrono::steady_clock::now();
      rr.timers.ln1_ms += ElapsedMs(t0, t1);

      std::vector<float> attn_out;
      SelfAttention(ln1, w, cfg, rr.verify, rr.timers, attn_out);

      std::vector<float> mlp_out;
      if (cfg.use_parallel_residual) {
        std::vector<float> ln2;
        t0 = std::chrono::steady_clock::now();
        LayerNorm(hidden_states, t_len, hidden, w.post_ln_w, w.post_ln_b,
                  static_cast<float>(cfg.layer_norm_eps), ln2);
        t1 = std::chrono::steady_clock::now();
        rr.timers.ln2_ms += ElapsedMs(t0, t1);
        MlpForward(ln2, w, cfg, rr.timers, mlp_out);

        AddInPlace(hidden_states, attn_out);
        AddInPlace(hidden_states, mlp_out);
      } else {
        AddInPlace(hidden_states, attn_out);
        std::vector<float> ln2;
        t0 = std::chrono::steady_clock::now();
        LayerNorm(hidden_states, t_len, hidden, w.post_ln_w, w.post_ln_b,
                  static_cast<float>(cfg.layer_norm_eps), ln2);
        t1 = std::chrono::steady_clock::now();
        rr.timers.ln2_ms += ElapsedMs(t0, t1);
        MlpForward(ln2, w, cfg, rr.timers, mlp_out);
        AddInPlace(hidden_states, mlp_out);
      }
    }

    if (HasNanOrInf(hidden_states)) {
      rr.verify.has_nan = true;
    }

    std::vector<float> final_hidden;
    t0 = std::chrono::steady_clock::now();
    LayerNorm(hidden_states, t_len, hidden, model.final_ln_w, model.final_ln_b,
              static_cast<float>(cfg.layer_norm_eps), final_hidden);
    t1 = std::chrono::steady_clock::now();
    rr.timers.final_ln_ms += ElapsedMs(t0, t1);

    std::vector<float> last(hidden);
    for (int i = 0; i < hidden; ++i) {
      last[i] = final_hidden[static_cast<size_t>(t_len - 1) * hidden + i];
    }

    t0 = std::chrono::steady_clock::now();
    std::vector<float> logits(cfg.vocab_size, 0.0f);
    for (int v = 0; v < cfg.vocab_size; ++v) {
      const float* row = &model.embed_out[static_cast<size_t>(v) * hidden];
      float s = 0.0f;
      for (int i = 0; i < hidden; ++i) {
        s += row[i] * last[i];
      }
      logits[v] = s;
    }
    t1 = std::chrono::steady_clock::now();
    rr.timers.lm_head_ms += ElapsedMs(t0, t1);

    if (HasNanOrInf(logits)) {
      rr.verify.has_nan = true;
    }

    const int next_id =
        SampleFromLogits(logits, tokens, temperature, top_k, repetition_penalty, rng);
    tokens.push_back(next_id);
    rr.generated_tokens.push_back(next_id);
    rr.last_logits = std::move(logits);
  }

  auto total_end = std::chrono::steady_clock::now();
  rr.timers.total_ms = ElapsedMs(total_begin, total_end);
  return rr;
}

void AddTimers(Timers& a, const Timers& b) {
  a.total_ms += b.total_ms;
  a.embed_ms += b.embed_ms;
  a.ln1_ms += b.ln1_ms;
  a.qkv_ms += b.qkv_ms;
  a.attn_ms += b.attn_ms;
  a.attn_proj_ms += b.attn_proj_ms;
  a.ln2_ms += b.ln2_ms;
  a.mlp_ms += b.mlp_ms;
  a.final_ln_ms += b.final_ln_ms;
  a.lm_head_ms += b.lm_head_ms;
}
