#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <stdexcept>
#include <vector>

struct Config {
  int train_samples = 12000;
  int test_samples = 1200;
  int steps = 1000;
  int context = 256; // Keep up to 256 previous characters.
  int d_model = 48;
  int d_ff = 96;
  float lr = 0.005f; // Use a small learning rate to avoid large changes.
  float clip_norm = 5.0f;
  int seed = 42;
  int print_every = 100;
  int eval_exact_n = 200;
};

struct Sample {
  std::string intent;
  std::string command;
  std::string line;
};

struct Vocab {
  int pad_id = 0;
  std::vector<char> id_to_char;
  std::unordered_map<char, int> char_to_id;
};

static bool parse_int(const std::string &s, int *out) {
  try {
    *out = std::stoi(s);
    return true;
  } catch (...) {
    return false;
  }
}

static bool parse_float(const std::string &s, float *out) {
  try {
    *out = std::stof(s);
    return true;
  } catch (...) {
    return false;
  }
}

static bool parse_args(int argc, char **argv, Config *cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](int *idx) -> bool {
      if (*idx + 1 >= argc) {
        return false;
      }
      ++(*idx);
      return true;
    };

    if (a == "--train-samples") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->train_samples)) return false;
    } else if (a == "--test-samples") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->test_samples)) return false;
    } else if (a == "--steps") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->steps)) return false;
    } else if (a == "--ctx") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->context)) return false;
    } else if (a == "--d-model") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->d_model)) return false;
    } else if (a == "--d-ff") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->d_ff)) return false;
    } else if (a == "--lr") {
      if (!need_value(&i) || !parse_float(argv[i], &cfg->lr)) return false;
    } else if (a == "--clip-norm") {
      if (!need_value(&i) || !parse_float(argv[i], &cfg->clip_norm)) return false;
    } else if (a == "--seed") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->seed)) return false;
    } else if (a == "--print-every") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->print_every)) return false;
    } else if (a == "--eval-exact-n") {
      if (!need_value(&i) || !parse_int(argv[i], &cfg->eval_exact_n)) return false;
    } else {
      return false;
    }
  }

  if (cfg->train_samples <= 0 || cfg->test_samples <= 0 || cfg->steps <= 0 ||
      cfg->context <= 0 || cfg->d_model <= 0 || cfg->d_ff <= 0 || cfg->lr <= 0.0f ||
      cfg->clip_norm <= 0.0f || cfg->print_every <= 0 || cfg->eval_exact_n <= 0) {
    return false;
  }
  return true;
}

static void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --train-samples N\n"
            << "  --test-samples N\n"
            << "  --steps N\n"
            << "  --ctx N\n"
            << "  --d-model N\n"
            << "  --d-ff N\n"
            << "  --lr X\n"
            << "  --clip-norm X\n"
            << "  --seed N\n"
            << "  --print-every N\n"
            << "  --eval-exact-n N\n";
}

static Sample make_sample(std::mt19937 &rng) {
  std::uniform_int_distribution<int> b01(0, 1);

  const int channels[3] = {1, 6, 11};
  const char *dnses[4] = {"1.1.1.1", "8.8.8.8", "9.9.9.9", "8.8.4.4"};
  const char *firewall_modes[2] = {"normal", "strict"};
  const int powers[3] = {40, 70, 100};
  const int clients[4] = {8, 16, 32, 64};

  int guest = b01(rng);
  int channel = channels[rng() % 3];
  const char *dns = dnses[rng() % 4];
  const char *fw = firewall_modes[rng() % 2];
  int tx_power = powers[rng() % 3];
  int max_client = clients[rng() % 4];

  std::string intent =
      std::string("set guest wifi ") + (guest ? "on" : "off") +
      ", set channel " + std::to_string(channel) +
      ", set dns " + dns +
      ", firewall " + fw +
      ", tx power " + std::to_string(tx_power) +
      ", max clients " + std::to_string(max_client);

  std::string command =
      std::string("set wifi.guest=") + (guest ? "1" : "0") + "; " +
      "set wifi.channel=" + std::to_string(channel) + "; " +
      "set wan.dns=" + dns + "; " +
      "set firewall.mode=" + fw + "; " +
      "set wifi.tx_power=" + std::to_string(tx_power) + "; " +
      "set wifi.max_clients=" + std::to_string(max_client) + "; " +
      "commit;";

  Sample s;
  s.intent = intent;
  s.command = command;
  s.line = "[INTENT] " + intent + " [CMD] " + command + "\n";
  return s;
}

// Make all 576 possible settings, then separate training and test settings.
// A setting may repeat within a group, but cannot appear in both groups.
static std::vector<Sample> make_dataset(int n, int seed, bool test_split) {
  std::mt19937 generator(12345);
  std::vector<Sample> unique;
  std::unordered_map<std::string, bool> seen;
  while (unique.size() < 576) {
    Sample sample = make_sample(generator);
    if (seen.emplace(sample.intent, true).second) unique.push_back(sample);
  }
  std::mt19937 splitter(seed);
  std::shuffle(unique.begin(), unique.end(), splitter);
  const int boundary = 460;
  std::mt19937 sampler(seed + (test_split ? 1 : 0));
  std::uniform_int_distribution<int> pick(test_split ? boundary : 0,
                                         test_split ? 575 : boundary - 1);
  std::vector<Sample> samples;
  samples.reserve(n);
  for (int i = 0; i < n; ++i) samples.push_back(unique[pick(sampler)]);
  return samples;
}

static Vocab build_vocab(const std::vector<Sample> &a, const std::vector<Sample> &b) {
  Vocab v;
  v.id_to_char.push_back('\0');

  auto add_char = [&](char c) {
    if (v.char_to_id.find(c) == v.char_to_id.end()) {
      int id = static_cast<int>(v.id_to_char.size());
      v.char_to_id[c] = id;
      v.id_to_char.push_back(c);
    }
  };

  add_char('\n');
  add_char('[');
  add_char(']');

  for (const auto &s : a) {
    for (char c : s.line) add_char(c);
  }
  for (const auto &s : b) {
    for (char c : s.line) add_char(c);
  }
  return v;
}

static std::vector<int> encode(const std::string &s, const Vocab &v) {
  std::vector<int> ids;
  ids.reserve(s.size());
  for (char c : s) {
    auto it = v.char_to_id.find(c);
    if (it != v.char_to_id.end()) ids.push_back(it->second);
  }
  return ids;
}

static std::vector<std::vector<int>> encode_dataset(const std::vector<Sample> &ds, const Vocab &v) {
  std::vector<std::vector<int>> out;
  out.reserve(ds.size());
  for (const auto &s : ds) out.push_back(encode(s.line, v));
  return out;
}

class TinyLM {
 public:
  TinyLM(int vocab, int ctx, int d_model, int d_ff, int seed)
      : vocab_(vocab), ctx_(ctx), d_model_(d_model), d_ff_(d_ff) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uf(-0.06f, 0.06f);

    E_.resize(static_cast<size_t>(vocab_) * d_model_);
    P_.resize(static_cast<size_t>(ctx_) * d_model_);

    Wq_.resize(static_cast<size_t>(d_model_) * d_model_);
    Wk_.resize(static_cast<size_t>(d_model_) * d_model_);
    Wv_.resize(static_cast<size_t>(d_model_) * d_model_);
    Wo_.resize(static_cast<size_t>(d_model_) * d_model_);

    W1_.resize(static_cast<size_t>(d_ff_) * d_model_);
    b1_.assign(d_ff_, 0.0f);
    W2_.resize(static_cast<size_t>(d_model_) * d_ff_);
    b2_.assign(d_model_, 0.0f);

    Wcls_.resize(static_cast<size_t>(vocab_) * d_model_);
    bcls_.assign(vocab_, 0.0f);

    for (auto &x : E_) x = uf(rng);
    for (auto &x : P_) x = uf(rng);
    for (auto &x : Wq_) x = uf(rng);
    for (auto &x : Wk_) x = uf(rng);
    for (auto &x : Wv_) x = uf(rng);
    for (auto &x : Wo_) x = uf(rng);
    for (auto &x : W1_) x = uf(rng);
    for (auto &x : W2_) x = uf(rng);
    for (auto &x : Wcls_) x = uf(rng);

    x_.resize(static_cast<size_t>(ctx_) * d_model_);
    q_.resize(static_cast<size_t>(ctx_) * d_model_);
    k_.resize(static_cast<size_t>(ctx_) * d_model_);
    v_.resize(static_cast<size_t>(ctx_) * d_model_);

    scores_.resize(ctx_);
    attn_.resize(ctx_);
    attn_out_.resize(d_model_);
    attn_proj_.resize(d_model_);

    h1_.resize(d_model_);
    zff_.resize(d_ff_);
    ff1_.resize(d_ff_);
    ff2_.resize(d_model_);
    h2_.resize(d_model_);

    logits_.resize(vocab_);
    probs_.resize(vocab_);

    dE_.resize(E_.size());
    dP_.resize(P_.size());
    dWq_.resize(Wq_.size());
    dWk_.resize(Wk_.size());
    dWv_.resize(Wv_.size());
    dWo_.resize(Wo_.size());
    dW1_.resize(W1_.size());
    db1_.resize(b1_.size());
    dW2_.resize(W2_.size());
    db2_.resize(b2_.size());
    dWcls_.resize(Wcls_.size());
    dbcls_.resize(bcls_.size());

    dx_.resize(static_cast<size_t>(ctx_) * d_model_);
    dq_.resize(static_cast<size_t>(ctx_) * d_model_);
    dk_.resize(static_cast<size_t>(ctx_) * d_model_);
    dv_.resize(static_cast<size_t>(ctx_) * d_model_);

    dscores_.resize(ctx_);
    dattn_.resize(ctx_);
    dattn_out_.resize(d_model_);
    dattn_proj_.resize(d_model_);
    dh1_.resize(d_model_);
    dh2_.resize(d_model_);
    dff2_.resize(d_model_);
    dff1_.resize(d_ff_);
    dzff_.resize(d_ff_);
    dlogits_.resize(vocab_);
  }

  // Read the previous characters and predict the next one.
  // Empty places on the left are filled with padding (ID 0).
  float forward(const std::vector<int> &context_ids, int target_id) {
    // Save the input IDs for the backward pass.
    last_context_ids_ = context_ids;

    // Give each character a learned number vector, then add its position.
    for (int t = 0; t < ctx_; ++t) {
      int tok = context_ids[t];
      for (int i = 0; i < d_model_; ++i) {
        x_[XIdx(t, i)] = E_[EmbIdx(tok, i)] + P_[PosIdx(t, i)];
      }
    }

    // Keys are used to score each position. Values carry its information.
    for (int t = 0; t < ctx_; ++t) {

      matvec_row_major(Wk_, d_model_, d_model_, &x_[XIdx(t, 0)], &k_[XIdx(t, 0)]);
      matvec_row_major(Wv_, d_model_, d_model_, &x_[XIdx(t, 0)], &v_[XIdx(t, 0)]);
    }

    const int last = ctx_ - 1;
    // Only the last position needs a query: we predict just one next character.
    matvec_row_major(Wq_, d_model_, d_model_, &x_[XIdx(last, 0)], &q_[XIdx(last, 0)]);
    // Use only earlier characters. The answer is not part of the input.
    // Ignore padding when deciding which characters to look at.
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model_));
    // Compare the query with each key to get an attention score.
    float max_score = -1e30f;
    for (int t = 0; t <= last; ++t) {
      float s = 0.0f;
      for (int i = 0; i < d_model_; ++i) {
        s += q_[XIdx(last, i)] * k_[XIdx(t, i)];
      }
      s *= scale;
      if (context_ids[t] == 0) s = -1e30f;
      scores_[t] = s;
      max_score = std::max(max_score, s);
    }
    // Softmax turns the scores into positive weights that add up to 1.
    float z = 0.0f;
    for (int t = 0; t <= last; ++t) {
      attn_[t] = std::exp(scores_[t] - max_score);
      z += attn_[t];
    }
    for (int t = 0; t <= last; ++t) {
      attn_[t] /= z;
    }

    // Mix the value vectors. A larger attention weight gives more influence.
    std::fill(attn_out_.begin(), attn_out_.end(), 0.0f);
    for (int t = 0; t <= last; ++t) {
      float a = attn_[t];
      for (int i = 0; i < d_model_; ++i) {
        attn_out_[i] += a * v_[XIdx(t, i)];
      }
    }

    // Add the attention result to the original last-character vector.
    matvec_row_major(Wo_, d_model_, d_model_, attn_out_.data(), attn_proj_.data());
    for (int i = 0; i < d_model_; ++i) {
      h1_[i] = x_[XIdx(last, i)] + attn_proj_[i];
    }

    // A small neural network processes the result. ReLU changes negatives to 0.
    for (int j = 0; j < d_ff_; ++j) {
      float s = b1_[j];
      const size_t row = static_cast<size_t>(j) * d_model_;
      for (int i = 0; i < d_model_; ++i) {
        s += W1_[row + i] * h1_[i];
      }
      zff_[j] = s;
      ff1_[j] = s > 0.0f ? s : 0.0f;
    }

    for (int i = 0; i < d_model_; ++i) {
      float s = b2_[i];
      const size_t row = static_cast<size_t>(i) * d_ff_;
      for (int j = 0; j < d_ff_; ++j) {
        s += W2_[row + j] * ff1_[j];
      }
      ff2_[i] = s;
      h2_[i] = h1_[i] + ff2_[i];
    }

    // Give every possible next character a score.
    for (int o = 0; o < vocab_; ++o) {
      float s = bcls_[o];
      const size_t row = static_cast<size_t>(o) * d_model_;
      for (int i = 0; i < d_model_; ++i) {
        s += Wcls_[row + i] * h2_[i];
      }
      logits_[o] = s;
    }

    // Convert character scores to probabilities.
    // Subtract the largest score to keep exp() from becoming too large.
    float m = logits_[0];
    for (int o = 1; o < vocab_; ++o) m = std::max(m, logits_[o]);
    float denom = 0.0f;
    for (int o = 0; o < vocab_; ++o) {
      probs_[o] = std::exp(logits_[o] - m);
      denom += probs_[o];
    }
    for (int o = 0; o < vocab_; ++o) probs_[o] /= denom;

    // Loss is smaller when the correct character has a higher probability.
    float p = std::max(1e-12f, probs_[target_id]);
    if (!std::isfinite(denom)) throw std::runtime_error("Non-finite logits");
    return -std::log(p);
  }

  // Work backward to find how each weight affects the prediction error.
  // A gradient tells us which way to change a weight to reduce that error.
  // Find all gradients first, then change the weights at the end.
  void backward_and_update(int target_id, float lr, float clip_norm) {
    zero_grads();

    for (int o = 0; o < vocab_; ++o) {
      dlogits_[o] = probs_[o];
    }
    dlogits_[target_id] -= 1.0f;

    for (int o = 0; o < vocab_; ++o) {
      dbcls_[o] += dlogits_[o];
      const size_t row = static_cast<size_t>(o) * d_model_;
      for (int i = 0; i < d_model_; ++i) {
        dWcls_[row + i] += dlogits_[o] * h2_[i];
      }
    }

    std::fill(dh2_.begin(), dh2_.end(), 0.0f);
    for (int i = 0; i < d_model_; ++i) {
      float s = 0.0f;
      for (int o = 0; o < vocab_; ++o) {
        s += Wcls_[static_cast<size_t>(o) * d_model_ + i] * dlogits_[o];
      }
      dh2_[i] = s;
    }

    for (int i = 0; i < d_model_; ++i) {
      dh1_[i] = dh2_[i];
      dff2_[i] = dh2_[i];
      db2_[i] += dff2_[i];
      const size_t row = static_cast<size_t>(i) * d_ff_;
      for (int j = 0; j < d_ff_; ++j) {
        dW2_[row + j] += dff2_[i] * ff1_[j];
      }
    }

    std::fill(dff1_.begin(), dff1_.end(), 0.0f);
    for (int j = 0; j < d_ff_; ++j) {
      float s = 0.0f;
      for (int i = 0; i < d_model_; ++i) {
        s += W2_[static_cast<size_t>(i) * d_ff_ + j] * dff2_[i];
      }
      dff1_[j] = s;
    }

    for (int j = 0; j < d_ff_; ++j) {
      dzff_[j] = zff_[j] > 0.0f ? dff1_[j] : 0.0f;
      db1_[j] += dzff_[j];
      const size_t row = static_cast<size_t>(j) * d_model_;
      for (int i = 0; i < d_model_; ++i) {
        dW1_[row + i] += dzff_[j] * h1_[i];
        dh1_[i] += W1_[row + i] * dzff_[j];
      }
    }

    std::fill(dattn_proj_.begin(), dattn_proj_.end(), 0.0f);
    for (int i = 0; i < d_model_; ++i) dattn_proj_[i] = dh1_[i];

    const int last = ctx_ - 1;
    for (int i = 0; i < d_model_; ++i) {
      dx_[XIdx(last, i)] += dh1_[i];
    }

    for (int r = 0; r < d_model_; ++r) {
      const size_t row = static_cast<size_t>(r) * d_model_;
      for (int c = 0; c < d_model_; ++c) {
        dWo_[row + c] += dattn_proj_[r] * attn_out_[c];
      }
    }
    for (int c = 0; c < d_model_; ++c) {
      float s = 0.0f;
      for (int r = 0; r < d_model_; ++r) {
        s += Wo_[static_cast<size_t>(r) * d_model_ + c] * dattn_proj_[r];
      }
      dattn_out_[c] = s;
    }

    for (int t = 0; t <= last; ++t) {
      float s = 0.0f;
      for (int i = 0; i < d_model_; ++i) {
        s += dattn_out_[i] * v_[XIdx(t, i)];
      }
      dattn_[t] = s;
      for (int i = 0; i < d_model_; ++i) {
        dv_[XIdx(t, i)] += attn_[t] * dattn_out_[i];
      }
    }

    // Work backward through softmax to find the gradient of each score.
    float weighted = 0.0f;
    for (int t = 0; t <= last; ++t) weighted += dattn_[t] * attn_[t];
    for (int t = 0; t <= last; ++t) {
      dscores_[t] = attn_[t] * (dattn_[t] - weighted);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model_));
    for (int t = 0; t <= last; ++t) {
      float ds = dscores_[t] * scale;
      for (int i = 0; i < d_model_; ++i) {
        dq_[XIdx(last, i)] += ds * k_[XIdx(t, i)];
        dk_[XIdx(t, i)] += ds * q_[XIdx(last, i)];
      }
    }

    // All characters use the same key and value weights.
    // If these loops run in parallel later, their gradient sums need care.
    for (int t = 0; t < ctx_; ++t) {
      backproj_add(dx_.data(), dWv_, Wv_, &dv_[XIdx(t, 0)], &x_[XIdx(t, 0)], t);
      backproj_add(dx_.data(), dWk_, Wk_, &dk_[XIdx(t, 0)], &x_[XIdx(t, 0)], t);
    }
    backproj_add(dx_.data(), dWq_, Wq_, &dq_[XIdx(last, 0)], &x_[XIdx(last, 0)], last);

    for (int t = 0; t < ctx_; ++t) {
      int tok = last_context_ids_[t];
      for (int i = 0; i < d_model_; ++i) {
        dE_[EmbIdx(tok, i)] += dx_[XIdx(t, i)];
        dP_[PosIdx(t, i)] += dx_[XIdx(t, i)];
      }
    }

    // If the gradients are too large, shrink them all by the same amount.
    // This keeps one training update from making a very large change.
    std::vector<std::vector<float> *> gradients = {
      &dE_, &dP_, &dWq_, &dWk_, &dWv_, &dWo_,
      &dW1_, &db1_, &dW2_, &db2_, &dWcls_, &dbcls_};
    double norm2 = 0.0;
    for (const auto *g : gradients) for (float x : *g) norm2 += double(x) * x;
    if (!std::isfinite(norm2)) throw std::runtime_error("Non-finite gradient");
    const double norm = std::sqrt(norm2);
    if (norm > static_cast<double>(clip_norm)) {
      ++clipped_updates_;
      const float scale = static_cast<float>(clip_norm / norm);
      for (auto *g : gradients) for (float &x : *g) x *= scale;
    }
    ++updates_;
    // Move each weight a small step in the direction that reduces the error.
    sgd_update(E_, dE_, lr);
    sgd_update(P_, dP_, lr);
    sgd_update(Wq_, dWq_, lr);
    sgd_update(Wk_, dWk_, lr);
    sgd_update(Wv_, dWv_, lr);
    sgd_update(Wo_, dWo_, lr);
    sgd_update(W1_, dW1_, lr);
    sgd_update(b1_, db1_, lr);
    sgd_update(W2_, dW2_, lr);
    sgd_update(b2_, db2_, lr);
    sgd_update(Wcls_, dWcls_, lr);
    sgd_update(bcls_, dbcls_, lr);
  }

  // Choose the character with the highest predicted probability.
  int predict_argmax(const std::vector<int> &context_ids) {
    forward(context_ids, 0);
    int best = 0;
    for (int o = 1; o < vocab_; ++o) {
      if (probs_[o] > probs_[best]) best = o;
    }
    return best;
  }

  const std::vector<float> &probs() const { return probs_; }
  int vocab_size() const { return vocab_; }
  long long updates() const { return updates_; }
  long long clipped_updates() const { return clipped_updates_; }

 private:
  long long updates_ = 0;
  long long clipped_updates_ = 0;
  int vocab_;
  int ctx_;
  int d_model_;
  int d_ff_;

  std::vector<float> E_;
  std::vector<float> P_;

  std::vector<float> Wq_;
  std::vector<float> Wk_;
  std::vector<float> Wv_;
  std::vector<float> Wo_;

  std::vector<float> W1_;
  std::vector<float> b1_;
  std::vector<float> W2_;
  std::vector<float> b2_;

  std::vector<float> Wcls_;
  std::vector<float> bcls_;

  std::vector<int> last_context_ids_;
  std::vector<float> x_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> scores_;
  std::vector<float> attn_;
  std::vector<float> attn_out_;
  std::vector<float> attn_proj_;
  std::vector<float> h1_;
  std::vector<float> zff_;
  std::vector<float> ff1_;
  std::vector<float> ff2_;
  std::vector<float> h2_;
  std::vector<float> logits_;
  std::vector<float> probs_;

  std::vector<float> dE_;
  std::vector<float> dP_;
  std::vector<float> dWq_;
  std::vector<float> dWk_;
  std::vector<float> dWv_;
  std::vector<float> dWo_;
  std::vector<float> dW1_;
  std::vector<float> db1_;
  std::vector<float> dW2_;
  std::vector<float> db2_;
  std::vector<float> dWcls_;
  std::vector<float> dbcls_;

  std::vector<float> dx_;
  std::vector<float> dq_;
  std::vector<float> dk_;
  std::vector<float> dv_;
  std::vector<float> dscores_;
  std::vector<float> dattn_;
  std::vector<float> dattn_out_;
  std::vector<float> dattn_proj_;
  std::vector<float> dh1_;
  std::vector<float> dh2_;
  std::vector<float> dff2_;
  std::vector<float> dff1_;
  std::vector<float> dzff_;
  std::vector<float> dlogits_;

  size_t XIdx(int t, int i) const { return static_cast<size_t>(t) * d_model_ + i; }
  size_t EmbIdx(int token, int i) const { return static_cast<size_t>(token) * d_model_ + i; }
  size_t PosIdx(int t, int i) const { return static_cast<size_t>(t) * d_model_ + i; }

  static void sgd_update(std::vector<float> &w, const std::vector<float> &g, float lr) {
    for (size_t i = 0; i < w.size(); ++i) w[i] -= lr * g[i];
  }

  void zero_grads() {
    std::fill(dE_.begin(), dE_.end(), 0.0f);
    std::fill(dP_.begin(), dP_.end(), 0.0f);
    std::fill(dWq_.begin(), dWq_.end(), 0.0f);
    std::fill(dWk_.begin(), dWk_.end(), 0.0f);
    std::fill(dWv_.begin(), dWv_.end(), 0.0f);
    std::fill(dWo_.begin(), dWo_.end(), 0.0f);
    std::fill(dW1_.begin(), dW1_.end(), 0.0f);
    std::fill(db1_.begin(), db1_.end(), 0.0f);
    std::fill(dW2_.begin(), dW2_.end(), 0.0f);
    std::fill(db2_.begin(), db2_.end(), 0.0f);
    std::fill(dWcls_.begin(), dWcls_.end(), 0.0f);
    std::fill(dbcls_.begin(), dbcls_.end(), 0.0f);

    std::fill(dx_.begin(), dx_.end(), 0.0f);
    std::fill(dq_.begin(), dq_.end(), 0.0f);
    std::fill(dk_.begin(), dk_.end(), 0.0f);
    std::fill(dv_.begin(), dv_.end(), 0.0f);
    std::fill(dscores_.begin(), dscores_.end(), 0.0f);
    std::fill(dattn_.begin(), dattn_.end(), 0.0f);
    std::fill(dattn_out_.begin(), dattn_out_.end(), 0.0f);
    std::fill(dattn_proj_.begin(), dattn_proj_.end(), 0.0f);
    std::fill(dh1_.begin(), dh1_.end(), 0.0f);
    std::fill(dh2_.begin(), dh2_.end(), 0.0f);
    std::fill(dff2_.begin(), dff2_.end(), 0.0f);
    std::fill(dff1_.begin(), dff1_.end(), 0.0f);
    std::fill(dzff_.begin(), dzff_.end(), 0.0f);
    std::fill(dlogits_.begin(), dlogits_.end(), 0.0f);
  }

  static void matvec_row_major(const std::vector<float> &W, int rows, int cols,
                               const float *x, float *y) {
    for (int r = 0; r < rows; ++r) {
      float s = 0.0f;
      size_t row = static_cast<size_t>(r) * cols;
      for (int c = 0; c < cols; ++c) {
        s += W[row + c] * x[c];
      }
      y[r] = s;
    }
  }

  void backproj_add(float *dx_all, std::vector<float> &dW, const std::vector<float> &W,
                    const float *dproj, const float *x, int token_pos) {
    for (int r = 0; r < d_model_; ++r) {
      size_t row = static_cast<size_t>(r) * d_model_;
      for (int c = 0; c < d_model_; ++c) {
        dW[row + c] += dproj[r] * x[c];
      }
    }
    for (int c = 0; c < d_model_; ++c) {
      float s = 0.0f;
      for (int r = 0; r < d_model_; ++r) {
        s += W[static_cast<size_t>(r) * d_model_ + c] * dproj[r];
      }
      dx_all[XIdx(token_pos, c)] += s;
    }
  }
};

static void build_context(const std::vector<int> &seq, int t, int ctx, int pad_id,
                          std::vector<int> *out) {
  out->assign(ctx, pad_id);
  for (int p = 0; p < ctx; ++p) {
    int src = t - ctx + p;
    if (src >= 0) (*out)[p] = seq[src];
  }
}

static void build_context_from_prefix(const std::vector<int> &prefix, int ctx, int pad_id,
                                      std::vector<int> *out) {
  out->assign(ctx, pad_id);
  int n = static_cast<int>(prefix.size());
  int copy = std::min(ctx, n);
  for (int i = 0; i < copy; ++i) {
    (*out)[ctx - copy + i] = prefix[n - copy + i];
  }
}

static float teacher_forced_eval(TinyLM *model, const std::vector<std::vector<int>> &enc,
                                 const std::vector<Sample> &samples, int pad_id, int ctx, double *token_acc) {
  long long n_token = 0;
  long long n_correct = 0;
  double total_loss = 0.0;
  std::vector<int> context;

  for (size_t sample_index = 0; sample_index < enc.size(); ++sample_index) {
    const auto &seq = enc[sample_index];
    // Check predictions for the command, not for the given request.
    const std::size_t pos = samples[sample_index].line.find("[CMD] ");
    if (pos == std::string::npos) continue;
    const int start = static_cast<int>(pos + 6);
    if (seq.size() < 2) continue;
    if (start >= static_cast<int>(seq.size())) continue;
    for (int t = start; t < static_cast<int>(seq.size()); ++t) {
      build_context(seq, t, ctx, pad_id, &context);
      float loss = model->forward(context, seq[t]);
      total_loss += loss;
      int pred = 0;
      const auto &p = model->probs();
      for (int o = 1; o < model->vocab_size(); ++o) {
        if (p[o] > p[pred]) pred = o;
      }
      if (pred == seq[t]) ++n_correct;
      ++n_token;
    }
  }

  if (n_token == 0) {
    *token_acc = 0.0;
    return 0.0f;
  }
  *token_acc = static_cast<double>(n_correct) / static_cast<double>(n_token);
  return static_cast<float>(total_loss / static_cast<double>(n_token));
}

static std::string greedy_generate_command(TinyLM *model, const Sample &s, const Vocab &v,
                                           int max_new_tokens, int ctx) {
  std::string prompt = "[INTENT] " + s.intent + " [CMD] ";
  std::vector<int> seq = encode(prompt, v);
  std::vector<int> context;
  std::string generated;

  for (int step = 0; step < max_new_tokens; ++step) {
    build_context_from_prefix(seq, ctx, v.pad_id, &context);
    int next_id = model->predict_argmax(context);
    char c = v.id_to_char[next_id];
    if (c == '\n') break;
    // Use our own output as part of the input for the next prediction.
    generated.push_back(c);
    seq.push_back(next_id);
  }
  return generated;
}

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, &cfg)) {
    print_usage(argv[0]);
    return 1;
  }

  std::cout << "Model=TinyLM(single-self-attention)"
            << " train_samples=" << cfg.train_samples
            << " test_samples=" << cfg.test_samples
            << " steps=" << cfg.steps
            << " ctx=" << cfg.context
            << " d_model=" << cfg.d_model
            << " d_ff=" << cfg.d_ff
            << " lr=" << cfg.lr
            << " clip_norm=" << cfg.clip_norm
            << " seed=" << cfg.seed << "\n";

  std::vector<Sample> train = make_dataset(cfg.train_samples, cfg.seed, false);
  std::vector<Sample> test = make_dataset(cfg.test_samples, cfg.seed, true);
  Vocab vocab = build_vocab(train, {});

  std::cout << "Vocab size: " << vocab.id_to_char.size() << "\n";

  std::vector<std::vector<int>> train_enc = encode_dataset(train, vocab);
  std::vector<std::vector<int>> test_enc = encode_dataset(test, vocab);

  TinyLM model(static_cast<int>(vocab.id_to_char.size()), cfg.context, cfg.d_model,
               cfg.d_ff, cfg.seed + 7);

  std::mt19937 rng(cfg.seed + 123);
  std::uniform_int_distribution<int> pick_train(0, static_cast<int>(train_enc.size()) - 1);

  std::vector<int> context_ids;
  double running_loss = 0.0;

  // Start timing training here, after the data and model are ready.
  // Testing and command generation are outside this timer.
  auto t0 = std::chrono::steady_clock::now();
  for (int step = 1; step <= cfg.steps; ++step) {
    int sidx = pick_train(rng);
    const std::vector<int> &seq = train_enc[sidx];
    if (seq.size() < 2) continue;

    double sample_loss = 0.0;
    // During training, use the correct earlier characters as input.
    // Update the weights after each character. The next prediction uses
    // these new weights, so we cannot simply train characters in parallel.
    const std::size_t pos = train[sidx].line.find("[CMD] ");
    if (pos == std::string::npos) continue;
    const int start = static_cast<int>(pos + 6);
    if (start >= static_cast<int>(seq.size())) continue;
    for (int t = start; t < static_cast<int>(seq.size()); ++t) {
      build_context(seq, t, cfg.context, vocab.pad_id, &context_ids);
      float loss = model.forward(context_ids, seq[t]);
      model.backward_and_update(seq[t], cfg.lr, cfg.clip_norm);
      sample_loss += loss;
    }
    sample_loss /= static_cast<double>(seq.size() - start);
    running_loss += sample_loss;

    if (step % cfg.print_every == 0) {
      std::cout << "step " << step
                << " avg_train_loss=" << (running_loss / cfg.print_every)
                << "\n";
      running_loss = 0.0;
    }
  }
  auto t1 = std::chrono::steady_clock::now();

  double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "training_time_ms=" << std::fixed << std::setprecision(3) << train_ms << "\n";

  std::cout << "character_updates=" << model.updates()
            << " clipped_updates=" << model.clipped_updates() << "\n";
  std::cout << "training_characters_per_second=" << model.updates() * 1000.0 / train_ms << "\n";
  auto eval_start = std::chrono::steady_clock::now();
  double test_token_acc = 0.0;
  float test_loss = teacher_forced_eval(&model, test_enc, test, vocab.pad_id, cfg.context,
                                        &test_token_acc);
  std::cout << "test_command_teacher_forced_loss=" << test_loss << "\n";
  std::cout << "test_command_next_char_accuracy=" << test_token_acc << "\n";

  auto eval_end = std::chrono::steady_clock::now();
  std::cout << "teacher_forced_eval_time_ms="
            << std::chrono::duration<double, std::milli>(eval_end-eval_start).count() << "\n";
  auto generation_start = std::chrono::steady_clock::now();
  int n_eval = std::min(cfg.eval_exact_n, static_cast<int>(test.size()));
  int exact_match = 0;
  for (int i = 0; i < n_eval; ++i) {
    std::string pred = greedy_generate_command(&model, test[i], vocab, 220, cfg.context);
    if (pred == test[i].command) ++exact_match;
    // Print examples so we can compare the requested and generated settings.
    if (i < 3) std::cout << "example_intent=" << test[i].intent
                        << "\nexample_expected=" << test[i].command
                        << "\nexample_generated=" << pred << "\n";
  }
  double exact_rate = (n_eval > 0)
                          ? static_cast<double>(exact_match) / static_cast<double>(n_eval)
                          : 0.0;
  std::cout << "exact_command_match_rate=" << exact_rate
            << " (" << exact_match << "/" << n_eval << ")\n";

  std::cout << "generation_time_ms=" << std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now()-generation_start).count() << "\n";
  return 0;
}
