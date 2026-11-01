#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Config {
  int32_t dim;
  int32_t hidden_dim;
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t vocab_size;
  int32_t seq_len;
};

struct TransformerWeights {
  const float* token_embedding_table = nullptr;
  const float* rms_att_weight = nullptr;
  const float* wq = nullptr;
  const float* wk = nullptr;
  const float* wv = nullptr;
  const float* wo = nullptr;
  const float* rms_ffn_weight = nullptr;
  const float* w1 = nullptr;
  const float* w2 = nullptr;
  const float* w3 = nullptr;
  const float* rms_final_weight = nullptr;
  const float* freq_cis_real = nullptr;
  const float* freq_cis_imag = nullptr;
  const float* wcls = nullptr;
};

struct Transformer {
  Config config{};
  TransformerWeights weights{};
  std::vector<float> data;
};

struct Tokenizer {
  std::vector<std::string> vocab;
  std::vector<float> vocab_scores;
  std::unordered_map<std::string, int> token_to_id;
  int max_token_length = 0;
};

struct RunState {
  std::vector<float> x;
  std::vector<float> xb;
  std::vector<float> xb2;
  std::vector<float> hb;
  std::vector<float> hb2;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> att;
  std::vector<float> logits;
  std::vector<float> key_cache;
  std::vector<float> value_cache;
};

struct RNG {
  uint64_t state;

  explicit RNG(uint64_t seed) : state(seed ? seed : 1ULL) {}

  uint32_t next_u32() {
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return static_cast<uint32_t>((x * 0x2545F4914F6CDD1DULL) >> 32);
  }

  float next_f32() {
    return (next_u32() >> 8) * (1.0f / 16777216.0f);
  }
};

static void check(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

static void matmul(float* out, const float* w, const float* x, int n, int d) {
  for (int i = 0; i < n; ++i) {
    float val = 0.0f;
    const float* row = w + static_cast<size_t>(i) * d;
    for (int j = 0; j < d; ++j) {
      val += row[j] * x[j];
    }
    out[i] = val;
  }
}

static void rmsnorm(float* out, const float* x, const float* weight, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; ++i) {
    ss += x[i] * x[i];
  }
  ss /= static_cast<float>(size);
  ss += 1e-5f;
  ss = 1.0f / std::sqrt(ss);
  for (int i = 0; i < size; ++i) {
    out[i] = weight[i] * (ss * x[i]);
  }
}

static void softmax(float* x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; ++i) {
    max_val = std::max(max_val, x[i]);
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }
  const float inv_sum = 1.0f / sum;
  for (int i = 0; i < size; ++i) {
    x[i] *= inv_sum;
  }
}

static int sample_argmax(const std::vector<float>& logits) {
  int max_i = 0;
  for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
    if (logits[i] > logits[max_i]) {
      max_i = i;
    }
  }
  return max_i;
}

static int sample_mult(const std::vector<float>& probs, RNG& rng) {
  const float r = rng.next_f32();
  float cdf = 0.0f;
  for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
    cdf += probs[i];
    if (r < cdf) {
      return i;
    }
  }
  return static_cast<int>(probs.size()) - 1;
}

static const std::vector<float>& forward(Transformer& model, RunState& s, int token, int pos) {
  const Config& p = model.config;
  const TransformerWeights& w = model.weights;
  const int dim = p.dim;
  const int hidden_dim = p.hidden_dim;
  const int n_layers = p.n_layers;
  const int n_heads = p.n_heads;
  const int n_kv_heads = p.n_kv_heads;
  const int vocab_size = p.vocab_size;
  const int seq_len = p.seq_len;

  const int head_size = dim / n_heads;
  const int kv_dim = (dim * n_kv_heads) / n_heads;
  const int kv_mul = n_heads / n_kv_heads;
  const float inv_sqrt_head_size = 1.0f / std::sqrt(static_cast<float>(head_size));

  const float* content_row = w.token_embedding_table + static_cast<size_t>(token) * dim;
  for (int i = 0; i < dim; ++i) {
    s.x[i] = content_row[i];
  }

  for (int l = 0; l < n_layers; ++l) {
    rmsnorm(s.xb.data(), s.x.data(), w.rms_att_weight + static_cast<size_t>(l) * dim, dim);

    matmul(s.q.data(), w.wq + static_cast<size_t>(l) * dim * dim, s.xb.data(), dim, dim);
    matmul(s.k.data(), w.wk + static_cast<size_t>(l) * dim * kv_dim, s.xb.data(), kv_dim, dim);
    matmul(s.v.data(), w.wv + static_cast<size_t>(l) * dim * kv_dim, s.xb.data(), kv_dim, dim);

    float* key_cache_row = s.key_cache.data() + static_cast<size_t>(l) * seq_len * kv_dim + pos * kv_dim;
    float* value_cache_row = s.value_cache.data() + static_cast<size_t>(l) * seq_len * kv_dim + pos * kv_dim;
    for (int i = 0; i < kv_dim; ++i) {
      key_cache_row[i] = s.k[i];
      value_cache_row[i] = s.v[i];
    }

    for (int i = 0; i < dim; i += 2) {
      const int head_dim = i % head_size;
      const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / head_size);
      const float val = pos * freq;
      const float fcr = std::cos(val);
      const float fci = std::sin(val);

      const float q0 = s.q[i];
      const float q1 = s.q[i + 1];
      s.q[i] = q0 * fcr - q1 * fci;
      s.q[i + 1] = q0 * fci + q1 * fcr;

      if (i < kv_dim) {
        const float k0 = s.k[i];
        const float k1 = s.k[i + 1];
        s.k[i] = k0 * fcr - k1 * fci;
        s.k[i + 1] = k0 * fci + k1 * fcr;
        key_cache_row[i] = s.k[i];
        key_cache_row[i + 1] = s.k[i + 1];
      }
    }

    for (int h = 0; h < n_heads; ++h) {
      float* att = s.att.data() + static_cast<size_t>(h) * seq_len;
      const float* q = s.q.data() + static_cast<size_t>(h) * head_size;

      for (int t = 0; t <= pos; ++t) {
        const float* k = s.key_cache.data() + static_cast<size_t>(l) * seq_len * kv_dim +
                         static_cast<size_t>(t) * kv_dim + static_cast<size_t>(h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) {
          score += q[i] * k[i];
        }
        att[t] = score * inv_sqrt_head_size;
      }

      softmax(att, pos + 1);

      float* xb = s.xb.data() + static_cast<size_t>(h) * head_size;
      for (int i = 0; i < head_size; ++i) {
        xb[i] = 0.0f;
      }

      for (int t = 0; t <= pos; ++t) {
        const float* v = s.value_cache.data() + static_cast<size_t>(l) * seq_len * kv_dim +
                         static_cast<size_t>(t) * kv_dim + static_cast<size_t>(h / kv_mul) * head_size;
        const float a = att[t];
        for (int i = 0; i < head_size; ++i) {
          xb[i] += a * v[i];
        }
      }
    }

    matmul(s.xb2.data(), w.wo + static_cast<size_t>(l) * dim * dim, s.xb.data(), dim, dim);
    for (int i = 0; i < dim; ++i) {
      s.x[i] += s.xb2[i];
    }

    rmsnorm(s.xb.data(), s.x.data(), w.rms_ffn_weight + static_cast<size_t>(l) * dim, dim);

    matmul(s.hb.data(), w.w1 + static_cast<size_t>(l) * dim * hidden_dim, s.xb.data(), hidden_dim, dim);
    matmul(s.hb2.data(), w.w3 + static_cast<size_t>(l) * dim * hidden_dim, s.xb.data(), hidden_dim, dim);

    for (int i = 0; i < hidden_dim; ++i) {
      const float x = s.hb[i];
      const float silu = x / (1.0f + std::exp(-x));
      s.hb[i] = silu * s.hb2[i];
    }

    matmul(s.xb.data(), w.w2 + static_cast<size_t>(l) * dim * hidden_dim, s.hb.data(), dim, hidden_dim);
    for (int i = 0; i < dim; ++i) {
      s.x[i] += s.xb[i];
    }
  }

  rmsnorm(s.x.data(), s.x.data(), w.rms_final_weight, dim);
  matmul(s.logits.data(), w.wcls, s.x.data(), vocab_size, dim);
  return s.logits;
}

static Transformer load_checkpoint(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "failed to open checkpoint: " + path);

  Transformer model;
  in.read(reinterpret_cast<char*>(&model.config), sizeof(Config));
  check(in.good(), "failed to read checkpoint header from: " + path);

  bool shared_weights = model.config.vocab_size > 0;
  model.config.vocab_size = std::abs(model.config.vocab_size);
  const Config& p = model.config;

  check(p.dim > 0 && p.hidden_dim > 0 && p.n_layers > 0 && p.n_heads > 0 && p.n_kv_heads > 0 &&
            p.vocab_size > 0 && p.seq_len > 0,
        "invalid config in checkpoint");
  check(p.dim % p.n_heads == 0, "dim must be divisible by n_heads");
  check(p.n_heads % p.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");

  const int head_size = p.dim / p.n_heads;
  const int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;

  size_t n_floats = 0;
  n_floats += static_cast<size_t>(p.vocab_size) * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * kv_dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * kv_dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * p.hidden_dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.hidden_dim * p.dim;
  n_floats += static_cast<size_t>(p.n_layers) * p.dim * p.hidden_dim;
  n_floats += static_cast<size_t>(p.dim);
  n_floats += static_cast<size_t>(p.seq_len) * (head_size / 2);
  n_floats += static_cast<size_t>(p.seq_len) * (head_size / 2);
  if (!shared_weights) {
    n_floats += static_cast<size_t>(p.vocab_size) * p.dim;
  }

  model.data.resize(n_floats);
  in.read(reinterpret_cast<char*>(model.data.data()), static_cast<std::streamsize>(n_floats * sizeof(float)));
  check(in.good(), "checkpoint file ended early: " + path);

  const float* ptr = model.data.data();
  model.weights.token_embedding_table = ptr;
  ptr += static_cast<size_t>(p.vocab_size) * p.dim;
  model.weights.rms_att_weight = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim;
  model.weights.wq = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * p.dim;
  model.weights.wk = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * kv_dim;
  model.weights.wv = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * kv_dim;
  model.weights.wo = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * p.dim;
  model.weights.rms_ffn_weight = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim;
  model.weights.w1 = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * p.hidden_dim;
  model.weights.w2 = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.hidden_dim * p.dim;
  model.weights.w3 = ptr;
  ptr += static_cast<size_t>(p.n_layers) * p.dim * p.hidden_dim;
  model.weights.rms_final_weight = ptr;
  ptr += p.dim;
  model.weights.freq_cis_real = ptr;
  ptr += static_cast<size_t>(p.seq_len) * (head_size / 2);
  model.weights.freq_cis_imag = ptr;
  ptr += static_cast<size_t>(p.seq_len) * (head_size / 2);
  if (shared_weights) {
    model.weights.wcls = model.weights.token_embedding_table;
  } else {
    model.weights.wcls = ptr;
    ptr += static_cast<size_t>(p.vocab_size) * p.dim;
  }

  check(ptr == model.data.data() + model.data.size(), "internal checkpoint parsing error");
  return model;
}

static Tokenizer load_tokenizer(const std::string& path, int expected_vocab_size) {
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "failed to open tokenizer: " + path);

  Tokenizer tok;
  tok.vocab.resize(expected_vocab_size);
  tok.vocab_scores.resize(expected_vocab_size);

  int32_t max_token_length = 0;
  in.read(reinterpret_cast<char*>(&max_token_length), sizeof(int32_t));
  check(in.good(), "failed to read tokenizer header");
  tok.max_token_length = max_token_length;

  for (int i = 0; i < expected_vocab_size; ++i) {
    float score = 0.0f;
    int32_t len = 0;
    in.read(reinterpret_cast<char*>(&score), sizeof(float));
    in.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
    check(in.good(), "failed to read tokenizer entry header");
    check(len >= 0 && len < (1 << 20), "invalid token length in tokenizer");
    std::string token(len, '\0');
    if (len > 0) {
      in.read(&token[0], len);
      check(in.good(), "failed to read tokenizer token bytes");
    }
    tok.vocab_scores[i] = score;
    tok.vocab[i] = token;
    tok.token_to_id[token] = i;
  }

  return tok;
}

static int lookup_token(const Tokenizer& tok, const std::string& piece) {
  auto it = tok.token_to_id.find(piece);
  if (it == tok.token_to_id.end()) {
    return -1;
  }
  return it->second;
}

static std::vector<int> encode(const Tokenizer& tok, const std::string& text, bool bos, bool eos) {
  std::vector<int> tokens;
  if (bos) {
    tokens.push_back(1);
  }

  if (!text.empty()) {
    const int dummy_prefix = lookup_token(tok, " ");
    if (dummy_prefix >= 0) {
      tokens.push_back(dummy_prefix);
    }
  }

  for (unsigned char c : text) {
    std::string s(1, static_cast<char>(c));
    int id = lookup_token(tok, s);
    if (id == -1) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(c));
      id = lookup_token(tok, buf);
    }
    check(id >= 0, "failed to encode byte; tokenizer does not contain byte fallback token");
    tokens.push_back(id);
  }

  while (true) {
    float best_score = -std::numeric_limits<float>::infinity();
    int best_id = -1;
    int best_pos = -1;
    for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i) {
      const std::string merged = tok.vocab[tokens[i]] + tok.vocab[tokens[i + 1]];
      const int id = lookup_token(tok, merged);
      if (id >= 0 && tok.vocab_scores[id] > best_score) {
        best_score = tok.vocab_scores[id];
        best_id = id;
        best_pos = i;
      }
    }

    if (best_pos == -1) {
      break;
    }

    tokens[best_pos] = best_id;
    tokens.erase(tokens.begin() + best_pos + 1);
  }

  if (eos) {
    tokens.push_back(2);
  }
  return tokens;
}

static std::string decode_piece(const Tokenizer& tok, int prev_token, int token) {
  std::string piece = tok.vocab[token];
  if (prev_token == 1 && !piece.empty() && piece[0] == ' ') {
    piece.erase(piece.begin());
  }

  if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
    auto hex = [](char c) -> int {
      if ('0' <= c && c <= '9') return c - '0';
      if ('A' <= c && c <= 'F') return c - 'A' + 10;
      if ('a' <= c && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    int hi = hex(piece[3]);
    int lo = hex(piece[4]);
    if (hi >= 0 && lo >= 0) {
      char byte = static_cast<char>((hi << 4) | lo);
      return std::string(1, byte);
    }
  }

  return piece;
}

static RunState create_run_state(const Config& p) {
  RunState s;
  const int dim = p.dim;
  const int hidden_dim = p.hidden_dim;
  const int n_heads = p.n_heads;
  const int seq_len = p.seq_len;
  const int vocab_size = p.vocab_size;
  const int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;

  s.x.resize(dim);
  s.xb.resize(dim);
  s.xb2.resize(dim);
  s.hb.resize(hidden_dim);
  s.hb2.resize(hidden_dim);
  s.q.resize(dim);
  s.k.resize(kv_dim);
  s.v.resize(kv_dim);
  s.att.resize(static_cast<size_t>(n_heads) * seq_len);
  s.logits.resize(vocab_size);
  s.key_cache.resize(static_cast<size_t>(p.n_layers) * seq_len * kv_dim);
  s.value_cache.resize(static_cast<size_t>(p.n_layers) * seq_len * kv_dim);

  return s;
}

struct Args {
  std::string checkpoint = "stories260K.bin";
  std::string tokenizer = "tokenizer.bin";
  std::string prompt = "Once upon a time";
  int steps = 128;
  int samples = 1;
  uint64_t seed = 42;
  float temperature = 0.8f;
};

static void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "  --checkpoint PATH    llama2.c checkpoint file (default: stories260K.bin)\n"
      << "  --tokenizer PATH     tokenizer binary (default: tokenizer.bin)\n"
      << "  --prompt TEXT        prompt text (default: \"Once upon a time\")\n"
      << "  --steps N            total tokens to run (default: 128)\n"
      << "  --samples N          number of independent stories (default: 1)\n"
      << "  --seed N             RNG seed (default: 42)\n"
      << "  --temperature X      sampling temperature, 0=greedy (default: 0.8)\n"
      << "  --help               show this message\n";
}

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](const std::string& name) {
      check(i + 1 < argc, "missing value for argument: " + name);
    };

    if (a == "--checkpoint") {
      need_value(a);
      args.checkpoint = argv[++i];
    } else if (a == "--tokenizer") {
      need_value(a);
      args.tokenizer = argv[++i];
    } else if (a == "--prompt") {
      need_value(a);
      args.prompt = argv[++i];
    } else if (a == "--steps") {
      need_value(a);
      args.steps = std::stoi(argv[++i]);
    } else if (a == "--samples") {
      need_value(a);
      args.samples = std::stoi(argv[++i]);
    } else if (a == "--seed") {
      need_value(a);
      args.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
    } else if (a == "--temperature") {
      need_value(a);
      args.temperature = std::stof(argv[++i]);
    } else if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }

  check(args.steps >= 1, "steps must be >= 1");
  check(args.samples >= 1, "samples must be >= 1");
  check(args.temperature >= 0.0f, "temperature must be >= 0");
  return args;
}

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    Transformer model = load_checkpoint(args.checkpoint);
    Tokenizer tok = load_tokenizer(args.tokenizer, model.config.vocab_size);

    if (args.steps > model.config.seq_len) {
      args.steps = model.config.seq_len;
    }

    std::vector<int> prompt_tokens = encode(tok, args.prompt, true, false);
    check(!prompt_tokens.empty(), "prompt produced no tokens");

    int64_t total_generated = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int sidx = 0; sidx < args.samples; ++sidx) {
      RunState state = create_run_state(model.config);
      RNG rng(args.seed + static_cast<uint64_t>(sidx));

      int token = prompt_tokens[0];
      int story_generated = 0;

      for (int pos = 0; pos < args.steps - 1; ++pos) {
        const std::vector<float>& logits = forward(model, state, token, pos);

        int next = 0;
        if (pos < static_cast<int>(prompt_tokens.size()) - 1) {
          next = prompt_tokens[pos + 1];
        } else {
          if (args.temperature == 0.0f) {
            next = sample_argmax(logits);
          } else {
            std::vector<float> probs = logits;
            for (float& x : probs) {
              x /= args.temperature;
            }
            softmax(probs.data(), static_cast<int>(probs.size()));
            next = sample_mult(probs, rng);
          }
        }

        ++story_generated;
        ++total_generated;

        if (next == 1 || next == 2) {
          break;
        }

        std::string piece = decode_piece(tok, token, next);
        std::cout << piece;
        token = next;
      }

      if (sidx + 1 < args.samples) {
        std::cout << "\n\n--- sample " << (sidx + 2) << " ---\n";
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double tps = (elapsed > 0.0) ? (static_cast<double>(total_generated) / elapsed) : 0.0;

    std::cout << "\n\n";
    std::cout << "total generated tokens: " << total_generated << "\n";
    std::cout << "total time (s): " << elapsed << "\n";
    std::cout << "tokens/sec: " << tps << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }
}
