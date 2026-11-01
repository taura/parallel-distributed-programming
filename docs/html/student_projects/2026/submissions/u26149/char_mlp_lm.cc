#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
  int context = 16;
  int embed = 32;
  int hidden = 256;
  int batch = 64;
  int steps = 2000;
  int eval_every = 200;
  int val_batches = 100;
  int sample_every = 1000;
  int sample_len = 200;
  int grad_check = 0;
  int seed = 42;
  int train_tokens = 800000;
  double lr = 0.05;
  double temperature = 1.0;
};

constexpr int kVocabSize = 96;  // '\n' + printable ASCII [32, 126]
constexpr int kUnknownCharId = 1;  // space

struct Model {
  int context = 0;
  int embed = 0;
  int hidden = 0;

  std::vector<double> emb;  // [V, E]
  std::vector<double> w1;   // [K*E, H]
  std::vector<double> b1;   // [H]
  std::vector<double> w2;   // [H, V]
  std::vector<double> b2;   // [V]
};

struct Grads {
  std::vector<double> emb;
  std::vector<double> w1;
  std::vector<double> b1;
  std::vector<double> w2;
  std::vector<double> b2;

  void zero() {
    std::fill(emb.begin(), emb.end(), 0.0);
    std::fill(w1.begin(), w1.end(), 0.0);
    std::fill(b1.begin(), b1.end(), 0.0);
    std::fill(w2.begin(), w2.end(), 0.0);
    std::fill(b2.begin(), b2.end(), 0.0);
  }
};

[[noreturn]] void die(const std::string& msg) {
  throw std::runtime_error(msg);
}

void print_usage(const char* prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --context N        context length (default 16)\n"
      << "  --embed N          embedding size (default 32)\n"
      << "  --hidden N         hidden size (default 256)\n"
      << "  --batch N          batch size (default 64)\n"
      << "  --steps N          SGD steps (default 2000)\n"
      << "  --lr X             learning rate (default 0.05)\n"
      << "  --train_tokens N   synthetic corpus length (default 800000)\n"
      << "  --eval_every N     evaluate every N steps (default 200)\n"
      << "  --val_batches N    validation batches per eval (default 100)\n"
      << "  --sample_every N   print sample every N steps (default 1000)\n"
      << "  --sample_len N     generated sample length (default 200)\n"
      << "  --temperature X    sampling temperature (default 1.0)\n"
      << "  --grad_check 0|1   finite-diff gradient check at startup\n"
      << "  --seed N           random seed (default 42)\n"
      << "  --help             show this help\n";
}

int parse_int_arg(const std::string& key, const std::string& val) {
  try {
    size_t p = 0;
    int x = std::stoi(val, &p);
    if (p != val.size()) {
      die("invalid integer for " + key + ": " + val);
    }
    return x;
  } catch (...) {
    die("invalid integer for " + key + ": " + val);
  }
}

double parse_double_arg(const std::string& key, const std::string& val) {
  try {
    size_t p = 0;
    double x = std::stod(val, &p);
    if (p != val.size()) {
      die("invalid number for " + key + ": " + val);
    }
    return x;
  } catch (...) {
    die("invalid number for " + key + ": " + val);
  }
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      die("missing value for " + arg);
    }
    const std::string val = argv[++i];
    if (arg == "--context") cfg.context = parse_int_arg(arg, val);
    else if (arg == "--embed") cfg.embed = parse_int_arg(arg, val);
    else if (arg == "--hidden") cfg.hidden = parse_int_arg(arg, val);
    else if (arg == "--batch") cfg.batch = parse_int_arg(arg, val);
    else if (arg == "--steps") cfg.steps = parse_int_arg(arg, val);
    else if (arg == "--eval_every") cfg.eval_every = parse_int_arg(arg, val);
    else if (arg == "--val_batches") cfg.val_batches = parse_int_arg(arg, val);
    else if (arg == "--sample_every") cfg.sample_every = parse_int_arg(arg, val);
    else if (arg == "--sample_len") cfg.sample_len = parse_int_arg(arg, val);
    else if (arg == "--grad_check") cfg.grad_check = parse_int_arg(arg, val);
    else if (arg == "--seed") cfg.seed = parse_int_arg(arg, val);
    else if (arg == "--train_tokens") cfg.train_tokens = parse_int_arg(arg, val);
    else if (arg == "--lr") cfg.lr = parse_double_arg(arg, val);
    else if (arg == "--temperature") cfg.temperature = parse_double_arg(arg, val);
    else die("unknown argument: " + arg);
  }

  if (cfg.context <= 0) die("--context must be > 0");
  if (cfg.embed <= 0) die("--embed must be > 0");
  if (cfg.hidden <= 0) die("--hidden must be > 0");
  if (cfg.batch <= 0) die("--batch must be > 0");
  if (cfg.steps <= 0) die("--steps must be > 0");
  if (cfg.eval_every <= 0) die("--eval_every must be > 0");
  if (cfg.val_batches <= 0) die("--val_batches must be > 0");
  if (cfg.sample_every <= 0) die("--sample_every must be > 0");
  if (cfg.sample_len <= 0) die("--sample_len must be > 0");
  if (cfg.train_tokens < 1000) die("--train_tokens must be >= 1000");
  if (!(cfg.lr > 0.0)) die("--lr must be > 0");
  if (!(cfg.temperature > 0.0)) die("--temperature must be > 0");
  if (cfg.grad_check != 0 && cfg.grad_check != 1) die("--grad_check must be 0 or 1");

  return cfg;
}

std::vector<char> make_id_to_char() {
  std::vector<char> id_to_char;
  id_to_char.reserve(kVocabSize);
  id_to_char.push_back('\n');
  for (int c = 32; c <= 126; ++c) id_to_char.push_back(static_cast<char>(c));
  return id_to_char;
}

int char_to_id(unsigned char c) {
  if (c == '\n') return 0;
  if (c >= 32 && c <= 126) return static_cast<int>(c) - 31;
  return kUnknownCharId;
}

std::string make_synthetic_corpus(int target_len, std::mt19937& rng) {
  const std::vector<std::string> templates = {
      "parallel programming lets us solve larger problems faster.",
      "this baseline trains a tiny character language model in plain c plus plus.",
      "performance analysis begins with a clear serial implementation.",
      "we measure runtime, loss, perplexity, and generated sample quality.",
      "gpu and multicore versions can parallelize matrix multiply and batches.",
      "cache behavior matters when dimensions become large.",
      "numerical stability uses logsumexp during softmax loss.",
      "deterministic runs use a fixed random seed for reproducibility.",
      "small models are useful for experimentation and correctness checks.",
      "an n gram context captures short range dependencies in text."};

  std::uniform_int_distribution<int> pick(0, static_cast<int>(templates.size()) - 1);
  std::uniform_int_distribution<int> year(2000, 2035);
  std::uniform_int_distribution<int> small_num(0, 999);

  std::string out;
  out.reserve(static_cast<size_t>(target_len) + 1024);
  while (static_cast<int>(out.size()) < target_len) {
    const std::string& a = templates[pick(rng)];
    const std::string& b = templates[pick(rng)];
    out += a;
    out += " ";
    out += b;
    out += " epoch=" + std::to_string(small_num(rng));
    out += " year=" + std::to_string(year(rng));
    out += "\n";
  }
  out.resize(static_cast<size_t>(target_len));
  return out;
}

std::vector<int> encode_text(const std::string& text) {
  std::vector<int> t;
  t.reserve(text.size());
  for (unsigned char c : text) t.push_back(char_to_id(c));
  return t;
}

Model init_model(const Config& cfg, std::mt19937& rng) {
  Model m;
  m.context = cfg.context;
  m.embed = cfg.embed;
  m.hidden = cfg.hidden;

  const int d = cfg.context * cfg.embed;

  m.emb.resize(kVocabSize * cfg.embed);
  m.w1.resize(static_cast<size_t>(d) * cfg.hidden);
  m.b1.resize(cfg.hidden, 0.0);
  m.w2.resize(static_cast<size_t>(cfg.hidden) * kVocabSize);
  m.b2.resize(kVocabSize, 0.0);

  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const double emb_scale = std::sqrt(1.0 / static_cast<double>(cfg.embed));
  const double w1_scale = std::sqrt(1.0 / static_cast<double>(d));
  const double w2_scale = std::sqrt(1.0 / static_cast<double>(cfg.hidden));

  for (double& x : m.emb) x = u(rng) * emb_scale;
  for (double& x : m.w1) x = u(rng) * w1_scale;
  for (double& x : m.w2) x = u(rng) * w2_scale;
  return m;
}

Grads make_grads(const Model& m) {
  Grads g;
  g.emb.resize(m.emb.size(), 0.0);
  g.w1.resize(m.w1.size(), 0.0);
  g.b1.resize(m.b1.size(), 0.0);
  g.w2.resize(m.w2.size(), 0.0);
  g.b2.resize(m.b2.size(), 0.0);
  return g;
}

double forward_backward_single(const Model& m, const std::vector<int>& ctx_ids, int target_id,
                               Grads* grads) {
  const int k = m.context;
  const int e = m.embed;
  const int h = m.hidden;
  const int d = k * e;

  std::vector<double> x(d, 0.0);
  for (int i = 0; i < k; ++i) {
    const int token = ctx_ids[i];
    const size_t src = static_cast<size_t>(token) * e;
    const size_t dst = static_cast<size_t>(i) * e;
    for (int j = 0; j < e; ++j) x[dst + j] = m.emb[src + j];
  }

  std::vector<double> z1(h, 0.0), h1(h, 0.0);
  for (int j = 0; j < h; ++j) {
    double s = m.b1[j];
    for (int i = 0; i < d; ++i) s += x[i] * m.w1[static_cast<size_t>(i) * h + j];
    z1[j] = s;
    h1[j] = std::tanh(s);
  }

  std::vector<double> logits(kVocabSize, 0.0);
  for (int v = 0; v < kVocabSize; ++v) {
    double s = m.b2[v];
    for (int j = 0; j < h; ++j) s += h1[j] * m.w2[static_cast<size_t>(j) * kVocabSize + v];
    logits[v] = s;
  }

  double max_logit = logits[0];
  for (int v = 1; v < kVocabSize; ++v) max_logit = std::max(max_logit, logits[v]);
  double sum_exp = 0.0;
  for (double z : logits) sum_exp += std::exp(z - max_logit);
  const double logsumexp = max_logit + std::log(sum_exp);
  const double loss = logsumexp - logits[target_id];

  if (grads == nullptr) return loss;

  std::vector<double> dlogits(kVocabSize, 0.0);
  for (int v = 0; v < kVocabSize; ++v) {
    dlogits[v] = std::exp(logits[v] - logsumexp);
  }
  dlogits[target_id] -= 1.0;

  for (int j = 0; j < h; ++j) {
    const double hj = h1[j];
    for (int v = 0; v < kVocabSize; ++v) {
      grads->w2[static_cast<size_t>(j) * kVocabSize + v] += hj * dlogits[v];
    }
  }
  for (int v = 0; v < kVocabSize; ++v) grads->b2[v] += dlogits[v];

  std::vector<double> dh1(h, 0.0), dz1(h, 0.0);
  for (int j = 0; j < h; ++j) {
    double s = 0.0;
    for (int v = 0; v < kVocabSize; ++v) s += m.w2[static_cast<size_t>(j) * kVocabSize + v] * dlogits[v];
    dh1[j] = s;
    dz1[j] = (1.0 - h1[j] * h1[j]) * dh1[j];
  }

  for (int i = 0; i < d; ++i) {
    const double xi = x[i];
    for (int j = 0; j < h; ++j) {
      grads->w1[static_cast<size_t>(i) * h + j] += xi * dz1[j];
    }
  }
  for (int j = 0; j < h; ++j) grads->b1[j] += dz1[j];

  std::vector<double> dx(d, 0.0);
  for (int i = 0; i < d; ++i) {
    double s = 0.0;
    for (int j = 0; j < h; ++j) s += m.w1[static_cast<size_t>(i) * h + j] * dz1[j];
    dx[i] = s;
  }

  for (int i = 0; i < k; ++i) {
    const int token = ctx_ids[i];
    const size_t dst = static_cast<size_t>(token) * e;
    const size_t src = static_cast<size_t>(i) * e;
    for (int j = 0; j < e; ++j) grads->emb[dst + j] += dx[src + j];
  }

  return loss;
}

void sgd_update(Model& m, const Grads& g, double lr_scale) {
  for (size_t i = 0; i < m.emb.size(); ++i) m.emb[i] -= lr_scale * g.emb[i];
  for (size_t i = 0; i < m.w1.size(); ++i) m.w1[i] -= lr_scale * g.w1[i];
  for (size_t i = 0; i < m.b1.size(); ++i) m.b1[i] -= lr_scale * g.b1[i];
  for (size_t i = 0; i < m.w2.size(); ++i) m.w2[i] -= lr_scale * g.w2[i];
  for (size_t i = 0; i < m.b2.size(); ++i) m.b2[i] -= lr_scale * g.b2[i];
}

int sample_position(int low_inclusive, int high_inclusive, std::mt19937& rng) {
  std::uniform_int_distribution<int> d(low_inclusive, high_inclusive);
  return d(rng);
}

double eval_loss(const Model& m, const std::vector<int>& tokens, int split_idx, int batches,
                 int batch_size, int context, std::mt19937& rng) {
  const int val_low = split_idx + context;
  const int val_high = static_cast<int>(tokens.size()) - 1;
  if (val_low > val_high) return std::numeric_limits<double>::quiet_NaN();

  double total = 0.0;
  std::vector<int> ctx(context, 0);
  for (int bi = 0; bi < batches; ++bi) {
    for (int b = 0; b < batch_size; ++b) {
      const int pos = sample_position(val_low, val_high, rng);
      for (int i = 0; i < context; ++i) ctx[i] = tokens[pos - context + i];
      total += forward_backward_single(m, ctx, tokens[pos], nullptr);
    }
  }
  const double denom = static_cast<double>(batches) * batch_size;
  return total / denom;
}

std::vector<double> logits_for_context(const Model& m, const std::vector<int>& ctx_ids) {
  const int d = m.context * m.embed;
  std::vector<double> x(d, 0.0);
  for (int i = 0; i < m.context; ++i) {
    const int token = ctx_ids[i];
    const size_t src = static_cast<size_t>(token) * m.embed;
    const size_t dst = static_cast<size_t>(i) * m.embed;
    for (int j = 0; j < m.embed; ++j) x[dst + j] = m.emb[src + j];
  }

  std::vector<double> h1(m.hidden, 0.0);
  for (int j = 0; j < m.hidden; ++j) {
    double s = m.b1[j];
    for (int i = 0; i < d; ++i) s += x[i] * m.w1[static_cast<size_t>(i) * m.hidden + j];
    h1[j] = std::tanh(s);
  }

  std::vector<double> logits(kVocabSize, 0.0);
  for (int v = 0; v < kVocabSize; ++v) {
    double s = m.b2[v];
    for (int j = 0; j < m.hidden; ++j) s += h1[j] * m.w2[static_cast<size_t>(j) * kVocabSize + v];
    logits[v] = s;
  }
  return logits;
}

int sample_from_logits(std::vector<double> logits, double temperature, std::mt19937& rng) {
  for (double& z : logits) z /= temperature;
  const double max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probs(logits.size(), 0.0);
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  for (double& p : probs) p /= sum;
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return dist(rng);
}

std::string generate_text(const Model& m, const std::vector<char>& id_to_char, const std::string& prompt,
                          int length, double temperature, std::mt19937& rng) {
  std::vector<int> ctx(m.context, kUnknownCharId);
  for (char c : prompt) {
    const int id = char_to_id(static_cast<unsigned char>(c));
    std::rotate(ctx.begin(), ctx.begin() + 1, ctx.end());
    ctx.back() = id;
  }

  std::string out = prompt;
  out.reserve(prompt.size() + static_cast<size_t>(length));
  for (int i = 0; i < length; ++i) {
    const std::vector<double> logits = logits_for_context(m, ctx);
    const int next = sample_from_logits(logits, temperature, rng);
    std::rotate(ctx.begin(), ctx.begin() + 1, ctx.end());
    ctx.back() = next;
    out.push_back(id_to_char[next]);
  }
  return out;
}

double run_gradient_check(Model& m, const std::vector<int>& tokens, int split_idx, std::mt19937& rng) {
  const int pos = sample_position(m.context, split_idx - 1, rng);
  std::vector<int> ctx(m.context, 0);
  for (int i = 0; i < m.context; ++i) ctx[i] = tokens[pos - m.context + i];
  const int target = tokens[pos];

  Grads g = make_grads(m);
  g.zero();
  forward_backward_single(m, ctx, target, &g);

  struct Ref {
    std::string name;
    double* p;
    double grad;
  };

  std::vector<Ref> refs;
  refs.push_back({"emb[0]", &m.emb[0], g.emb[0]});
  refs.push_back({"w1[0]", &m.w1[0], g.w1[0]});
  refs.push_back({"b1[0]", &m.b1[0], g.b1[0]});
  refs.push_back({"w2[0]", &m.w2[0], g.w2[0]});
  refs.push_back({"b2[0]", &m.b2[0], g.b2[0]});

  const double eps = 1e-5;
  double max_rel = 0.0;
  std::cout << "gradient check (finite difference):\n";
  for (Ref& r : refs) {
    const double orig = *r.p;
    *r.p = orig + eps;
    const double l1 = forward_backward_single(m, ctx, target, nullptr);
    *r.p = orig - eps;
    const double l2 = forward_backward_single(m, ctx, target, nullptr);
    *r.p = orig;
    const double num = (l1 - l2) / (2.0 * eps);
    const double denom = std::max(1e-8, std::abs(num) + std::abs(r.grad));
    const double rel = std::abs(num - r.grad) / denom;
    max_rel = std::max(max_rel, rel);
    std::cout << "  " << std::setw(8) << r.name << " analytic=" << std::setw(12) << r.grad
              << " numeric=" << std::setw(12) << num << " rel_err=" << rel << "\n";
  }
  return max_rel;
}

double parameter_checksum(const Model& m) {
  double s = 0.0;
  for (double x : m.emb) s += std::abs(x);
  for (double x : m.w1) s += std::abs(x);
  for (double x : m.b1) s += std::abs(x);
  for (double x : m.w2) s += std::abs(x);
  for (double x : m.b2) s += std::abs(x);
  return s;
}

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    std::mt19937 rng(cfg.seed);

    const std::vector<char> id_to_char = make_id_to_char();
    const std::string text = make_synthetic_corpus(cfg.train_tokens, rng);
    const std::vector<int> tokens = encode_text(text);
    if (static_cast<int>(tokens.size()) <= cfg.context + 100) {
      die("dataset too small for this context length");
    }

    const int split_idx = static_cast<int>(tokens.size() * 9 / 10);
    if (split_idx <= cfg.context + 1) {
      die("training split too small; increase --train_tokens or reduce --context");
    }

    Model model = init_model(cfg, rng);
    Grads grads = make_grads(model);
    std::vector<int> ctx(cfg.context, 0);

    std::cout << "char_mlp_lm baseline\n";
    std::cout << "vocab=" << kVocabSize << " context=" << cfg.context << " embed=" << cfg.embed
              << " hidden=" << cfg.hidden << " batch=" << cfg.batch << " steps=" << cfg.steps
              << " train_tokens=" << cfg.train_tokens << " seed=" << cfg.seed << "\n";

    if (cfg.grad_check == 1) {
      const double max_rel = run_gradient_check(model, tokens, split_idx, rng);
      std::cout << "gradient_check_max_relative_error=" << max_rel;
      if (max_rel < 1e-4) std::cout << " (PASS)\n";
      else std::cout << " (WARN: expected roughly < 1e-4)\n";
    }

    double running_train_loss = 0.0;
    int running_samples = 0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step <= cfg.steps; ++step) {
      grads.zero();
      double batch_loss = 0.0;

      for (int b = 0; b < cfg.batch; ++b) {
        const int pos = sample_position(cfg.context, split_idx - 1, rng);
        for (int i = 0; i < cfg.context; ++i) ctx[i] = tokens[pos - cfg.context + i];
        batch_loss += forward_backward_single(model, ctx, tokens[pos], &grads);
      }

      const double inv_batch = 1.0 / static_cast<double>(cfg.batch);
      sgd_update(model, grads, cfg.lr * inv_batch);

      running_train_loss += batch_loss;
      running_samples += cfg.batch;

      if (step % cfg.eval_every == 0 || step == 1 || step == cfg.steps) {
        std::mt19937 eval_rng(cfg.seed + 1000 + step);
        const double val = eval_loss(model, tokens, split_idx, cfg.val_batches, cfg.batch, cfg.context, eval_rng);
        const double train_avg = running_train_loss / static_cast<double>(running_samples);
        const double train_ppl = std::exp(train_avg);
        const double val_ppl = std::exp(val);
        std::cout << "step=" << std::setw(6) << step << " train_loss=" << std::fixed << std::setprecision(6)
                  << train_avg << " train_ppl=" << train_ppl << " val_loss=" << val
                  << " val_ppl=" << val_ppl << "\n";
        running_train_loss = 0.0;
        running_samples = 0;
      }

      if (step % cfg.sample_every == 0 || step == cfg.steps) {
        std::mt19937 sample_rng(cfg.seed + 2000 + step);
        const std::string sample = generate_text(model, id_to_char, "prompt: ", cfg.sample_len,
                                                 cfg.temperature, sample_rng);
        std::cout << "----- sample at step " << step << " -----\n";
        std::cout << sample << "\n";
        std::cout << "--------------------------------\n";
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double train_seconds = std::chrono::duration<double>(t1 - t0).count();

    std::mt19937 final_eval_rng(cfg.seed + 9999);
    const double final_val =
        eval_loss(model, tokens, split_idx, cfg.val_batches * 2, cfg.batch, cfg.context, final_eval_rng);

    const int64_t train_tokens_processed = static_cast<int64_t>(cfg.steps) * cfg.batch;
    const double tok_per_sec = static_cast<double>(train_tokens_processed) / std::max(1e-9, train_seconds);

    std::cout << "training_time_seconds=" << std::fixed << std::setprecision(6) << train_seconds << "\n";
    std::cout << "train_examples_processed=" << train_tokens_processed << "\n";
    std::cout << "throughput_examples_per_sec=" << tok_per_sec << "\n";
    std::cout << "final_val_loss=" << final_val << " final_val_ppl=" << std::exp(final_val) << "\n";
    std::cout << "parameter_abs_checksum=" << parameter_checksum(model) << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
