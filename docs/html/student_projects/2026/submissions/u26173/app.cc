#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
  std::string data_path = "";
  int steps = 400;
  int seq_len = 32;
  int d_model = 64;
  int batch_size = 4;
  int eval_batches = 16;
  int print_every = 50;
  int seed = 123;
  double lr = 0.02;
  double grad_clip = 1.0;
};

static void usage(const char* prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --data PATH         Training text path (default: built-in tiny text)\n"
      << "  --steps N           Training steps (default: 400)\n"
      << "  --seq-len N         Context length (default: 32)\n"
      << "  --d-model N         Hidden size (default: 64)\n"
      << "  --batch-size N      Sequences per step (default: 4)\n"
      << "  --eval-batches N    Validation sequences per eval (default: 16)\n"
      << "  --print-every N     Print interval (default: 50)\n"
      << "  --lr X              Learning rate (default: 0.02)\n"
      << "  --grad-clip X       Per-parameter grad clip (default: 1.0)\n"
      << "  --seed N            RNG seed (default: 123)\n"
      << "  --help              Show this message\n";
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](const std::string& name) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return std::string(argv[++i]);
    };

    if (a == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--data") {
      cfg.data_path = need_value(a);
    } else if (a == "--steps") {
      cfg.steps = std::stoi(need_value(a));
    } else if (a == "--seq-len") {
      cfg.seq_len = std::stoi(need_value(a));
    } else if (a == "--d-model") {
      cfg.d_model = std::stoi(need_value(a));
    } else if (a == "--batch-size") {
      cfg.batch_size = std::stoi(need_value(a));
    } else if (a == "--eval-batches") {
      cfg.eval_batches = std::stoi(need_value(a));
    } else if (a == "--print-every") {
      cfg.print_every = std::stoi(need_value(a));
    } else if (a == "--lr") {
      cfg.lr = std::stod(need_value(a));
    } else if (a == "--grad-clip") {
      cfg.grad_clip = std::stod(need_value(a));
    } else if (a == "--seed") {
      cfg.seed = std::stoi(need_value(a));
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }
  return cfg;
}

static std::vector<int> read_bytes_or_default(const std::string& path) {
  std::vector<int> out;
  if (!path.empty()) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Failed to open data file: " + path);
    }
    char c;
    while (ifs.get(c)) {
      out.push_back(static_cast<unsigned char>(c));
    }
    return out;
  }

  const std::string fallback =
      "This is a tiny fallback corpus for debugging a character-level model. "
      "It is deterministic and repeats simple patterns. "
      "Parallel and distributed programming is fun. "
      "Tiny GPT from scratch in C++17. ";
  for (int rep = 0; rep < 200; ++rep) {
    for (unsigned char c : fallback) {
      out.push_back(static_cast<int>(c));
    }
  }
  return out;
}

static inline double clamp_value(double x, double lim) {
  if (x > lim) return lim;
  if (x < -lim) return -lim;
  return x;
}

struct TinyGPT {
  int V = 256;
  int D;
  int L;

  std::vector<double> E;
  std::vector<double> Wq;
  std::vector<double> Wk;
  std::vector<double> Wv;
  std::vector<double> Wo;
  std::vector<double> Wout;

  std::vector<double> gE;
  std::vector<double> gWq;
  std::vector<double> gWk;
  std::vector<double> gWv;
  std::vector<double> gWo;
  std::vector<double> gWout;

  explicit TinyGPT(int d_model, int seq_len)
      : D(d_model),
        L(seq_len),
        E(static_cast<size_t>(V) * D),
        Wq(static_cast<size_t>(D) * D),
        Wk(static_cast<size_t>(D) * D),
        Wv(static_cast<size_t>(D) * D),
        Wo(static_cast<size_t>(D) * D),
        Wout(static_cast<size_t>(D) * V),
        gE(static_cast<size_t>(V) * D),
        gWq(static_cast<size_t>(D) * D),
        gWk(static_cast<size_t>(D) * D),
        gWv(static_cast<size_t>(D) * D),
        gWo(static_cast<size_t>(D) * D),
        gWout(static_cast<size_t>(D) * V) {}

  void init(std::mt19937& rng) {
    auto init_vec = [&](std::vector<double>& w, double scale) {
      std::uniform_real_distribution<double> dist(-scale, scale);
      for (double& x : w) {
        x = dist(rng);
      }
    };

    const double s_e = std::sqrt(1.0 / std::max(1, D));
    const double s_w = std::sqrt(1.0 / std::max(1, D));
    const double s_out = std::sqrt(1.0 / std::max(1, V));
    init_vec(E, s_e);
    init_vec(Wq, s_w);
    init_vec(Wk, s_w);
    init_vec(Wv, s_w);
    init_vec(Wo, s_w);
    init_vec(Wout, s_out);
  }

  void zero_grads() {
    std::fill(gE.begin(), gE.end(), 0.0);
    std::fill(gWq.begin(), gWq.end(), 0.0);
    std::fill(gWk.begin(), gWk.end(), 0.0);
    std::fill(gWv.begin(), gWv.end(), 0.0);
    std::fill(gWo.begin(), gWo.end(), 0.0);
    std::fill(gWout.begin(), gWout.end(), 0.0);
  }

  void sgd_step(double lr, double grad_clip) {
    auto update = [&](std::vector<double>& w, std::vector<double>& g) {
      for (size_t i = 0; i < w.size(); ++i) {
        double gg = clamp_value(g[i], grad_clip);
        w[i] -= lr * gg;
      }
    };
    update(E, gE);
    update(Wq, gWq);
    update(Wk, gWk);
    update(Wv, gWv);
    update(Wo, gWo);
    update(Wout, gWout);
  }

  double forward_backward_one(const std::vector<int>& x,
                              const std::vector<int>& y,
                              bool do_backward,
                              double loss_scale) {
    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(D));

    std::vector<double> xv(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> q(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> k(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> v(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> att(static_cast<size_t>(L) * L, 0.0);
    std::vector<double> h(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> o(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> probs(static_cast<size_t>(L) * V, 0.0);

    auto vd = [&](int t, int d) -> size_t { return static_cast<size_t>(t) * D + d; };
    auto ll = [&](int t, int j) -> size_t { return static_cast<size_t>(t) * L + j; };
    auto lv = [&](int t, int tok) -> size_t { return static_cast<size_t>(t) * V + tok; };
    auto dd = [&](int i, int j) -> size_t { return static_cast<size_t>(i) * D + j; };
    auto dv = [&](int i, int tok) -> size_t { return static_cast<size_t>(i) * V + tok; };
    auto ev = [&](int tok, int i) -> size_t { return static_cast<size_t>(tok) * D + i; };

    for (int t = 0; t < L; ++t) {
      int tok = x[t];
      for (int i = 0; i < D; ++i) {
        xv[vd(t, i)] = E[ev(tok, i)];
      }
    }

    for (int t = 0; t < L; ++t) {
      for (int ocol = 0; ocol < D; ++ocol) {
        double qsum = 0.0;
        double ksum = 0.0;
        double vsum = 0.0;
        for (int i = 0; i < D; ++i) {
          double xi = xv[vd(t, i)];
          qsum += xi * Wq[dd(i, ocol)];
          ksum += xi * Wk[dd(i, ocol)];
          vsum += xi * Wv[dd(i, ocol)];
        }
        q[vd(t, ocol)] = qsum;
        k[vd(t, ocol)] = ksum;
        v[vd(t, ocol)] = vsum;
      }
    }

    for (int t = 0; t < L; ++t) {
      double max_score = -std::numeric_limits<double>::infinity();
      for (int j = 0; j <= t; ++j) {
        double dot = 0.0;
        for (int i = 0; i < D; ++i) {
          dot += q[vd(t, i)] * k[vd(j, i)];
        }
        double s = dot * inv_sqrt_d;
        att[ll(t, j)] = s;
        if (s > max_score) max_score = s;
      }

      double sum_exp = 0.0;
      for (int j = 0; j <= t; ++j) {
        double e = std::exp(att[ll(t, j)] - max_score);
        att[ll(t, j)] = e;
        sum_exp += e;
      }

      for (int j = 0; j <= t; ++j) {
        att[ll(t, j)] /= sum_exp;
      }

      for (int m = 0; m < D; ++m) {
        double hs = 0.0;
        for (int j = 0; j <= t; ++j) {
          hs += att[ll(t, j)] * v[vd(j, m)];
        }
        h[vd(t, m)] = hs;
      }

      for (int m = 0; m < D; ++m) {
        double os = 0.0;
        for (int i = 0; i < D; ++i) {
          os += h[vd(t, i)] * Wo[dd(i, m)];
        }
        o[vd(t, m)] = os;
      }

      std::vector<double> logits(V, 0.0);
      for (int tok = 0; tok < V; ++tok) {
        double ls = 0.0;
        for (int i = 0; i < D; ++i) {
          ls += o[vd(t, i)] * Wout[dv(i, tok)];
        }
        logits[tok] = ls;
      }

      double max_logit = -std::numeric_limits<double>::infinity();
      for (int tok = 0; tok < V; ++tok) {
        if (logits[tok] > max_logit) max_logit = logits[tok];
      }
      double sum = 0.0;
      for (int tok = 0; tok < V; ++tok) {
        double e = std::exp(logits[tok] - max_logit);
        probs[lv(t, tok)] = e;
        sum += e;
      }
      for (int tok = 0; tok < V; ++tok) {
        probs[lv(t, tok)] /= sum;
      }
    }

    double loss = 0.0;
    for (int t = 0; t < L; ++t) {
      double p = std::max(probs[lv(t, y[t])], 1e-12);
      loss += -std::log(p);
    }
    loss /= static_cast<double>(L);

    if (!do_backward) {
      return loss;
    }

    std::vector<double> dx(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> dq(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> dk(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> dvv(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> dh(static_cast<size_t>(L) * D, 0.0);
    std::vector<double> do_(static_cast<size_t>(L) * D, 0.0);

    const double dlogit_scale = loss_scale / static_cast<double>(L);

    for (int t = 0; t < L; ++t) {
      for (int tok = 0; tok < V; ++tok) {
        double g = probs[lv(t, tok)];
        if (tok == y[t]) g -= 1.0;
        g *= dlogit_scale;

        for (int i = 0; i < D; ++i) {
          gWout[dv(i, tok)] += o[vd(t, i)] * g;
          do_[vd(t, i)] += Wout[dv(i, tok)] * g;
        }
      }
    }

    for (int t = 0; t < L; ++t) {
      for (int m = 0; m < D; ++m) {
        double gom = do_[vd(t, m)];
        for (int i = 0; i < D; ++i) {
          gWo[dd(i, m)] += h[vd(t, i)] * gom;
          dh[vd(t, i)] += Wo[dd(i, m)] * gom;
        }
      }
    }

    for (int t = 0; t < L; ++t) {
      std::vector<double> da(L, 0.0);
      for (int j = 0; j <= t; ++j) {
        double s = 0.0;
        for (int m = 0; m < D; ++m) {
          s += dh[vd(t, m)] * v[vd(j, m)];
          dvv[vd(j, m)] += att[ll(t, j)] * dh[vd(t, m)];
        }
        da[j] = s;
      }

      double dot_da = 0.0;
      for (int j = 0; j <= t; ++j) {
        dot_da += att[ll(t, j)] * da[j];
      }

      for (int j = 0; j <= t; ++j) {
        double ds = att[ll(t, j)] * (da[j] - dot_da);
        for (int m = 0; m < D; ++m) {
          dq[vd(t, m)] += inv_sqrt_d * ds * k[vd(j, m)];
          dk[vd(j, m)] += inv_sqrt_d * ds * q[vd(t, m)];
        }
      }
    }

    for (int t = 0; t < L; ++t) {
      for (int ocol = 0; ocol < D; ++ocol) {
        double gq = dq[vd(t, ocol)];
        double gk = dk[vd(t, ocol)];
        double gv = dvv[vd(t, ocol)];
        for (int i = 0; i < D; ++i) {
          double xi = xv[vd(t, i)];
          gWq[dd(i, ocol)] += xi * gq;
          gWk[dd(i, ocol)] += xi * gk;
          gWv[dd(i, ocol)] += xi * gv;
          dx[vd(t, i)] += Wq[dd(i, ocol)] * gq + Wk[dd(i, ocol)] * gk + Wv[dd(i, ocol)] * gv;
        }
      }
    }

    for (int t = 0; t < L; ++t) {
      int tok = x[t];
      for (int i = 0; i < D; ++i) {
        gE[ev(tok, i)] += dx[vd(t, i)];
      }
    }

    return loss;
  }
};

static std::vector<int> slice_tokens(const std::vector<int>& data, int start, int len) {
  std::vector<int> out(len);
  for (int i = 0; i < len; ++i) {
    out[i] = data[static_cast<size_t>(start + i)];
  }
  return out;
}

static double eval_loss(TinyGPT& model,
                        const std::vector<int>& valid,
                        int eval_batches,
                        std::mt19937& rng) {
  if (valid.size() < static_cast<size_t>(model.L + 1)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int max_start = static_cast<int>(valid.size()) - model.L - 1;
  std::uniform_int_distribution<int> dist(0, max_start);

  double sum = 0.0;
  for (int b = 0; b < eval_batches; ++b) {
    int st = dist(rng);
    std::vector<int> x = slice_tokens(valid, st, model.L);
    std::vector<int> y = slice_tokens(valid, st + 1, model.L);
    sum += model.forward_backward_one(x, y, false, 0.0);
  }
  return sum / static_cast<double>(eval_batches);
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);

    if (cfg.steps <= 0 || cfg.seq_len <= 0 || cfg.d_model <= 0 || cfg.batch_size <= 0 ||
        cfg.eval_batches <= 0 || cfg.print_every <= 0 || cfg.lr <= 0.0 || cfg.grad_clip <= 0.0) {
      throw std::runtime_error("All numeric arguments must be positive.");
    }

    std::vector<int> tokens = read_bytes_or_default(cfg.data_path);
    if (tokens.size() < static_cast<size_t>(cfg.seq_len + 2)) {
      throw std::runtime_error("Dataset is too small for the selected --seq-len.");
    }

    size_t split = tokens.size() * 9 / 10;
    if (split < static_cast<size_t>(cfg.seq_len + 2)) {
      split = static_cast<size_t>(cfg.seq_len + 2);
    }
    if (tokens.size() - split < static_cast<size_t>(cfg.seq_len + 2)) {
      split = tokens.size() - static_cast<size_t>(cfg.seq_len + 2);
    }
    if (split < static_cast<size_t>(cfg.seq_len + 2) || tokens.size() - split < static_cast<size_t>(cfg.seq_len + 2)) {
      throw std::runtime_error("Need enough train and valid data. Provide a larger text file.");
    }

    std::vector<int> train(tokens.begin(), tokens.begin() + static_cast<long>(split));
    std::vector<int> valid(tokens.begin() + static_cast<long>(split), tokens.end());

    TinyGPT model(cfg.d_model, cfg.seq_len);
    std::mt19937 rng(cfg.seed);
    model.init(rng);

    const int max_start = static_cast<int>(train.size()) - cfg.seq_len - 1;
    std::uniform_int_distribution<int> train_dist(0, max_start);

    auto t0 = std::chrono::high_resolution_clock::now();
    double running_loss = 0.0;
    int running_count = 0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "dataset_bytes=" << tokens.size() << " train_bytes=" << train.size()
              << " valid_bytes=" << valid.size() << " vocab=256\n";
    std::cout << "steps=" << cfg.steps << " seq_len=" << cfg.seq_len << " d_model=" << cfg.d_model
              << " batch_size=" << cfg.batch_size << " lr=" << cfg.lr << "\n";

    for (int step = 1; step <= cfg.steps; ++step) {
      model.zero_grads();
      double step_loss = 0.0;
      const double loss_scale = 1.0 / static_cast<double>(cfg.batch_size);

      for (int b = 0; b < cfg.batch_size; ++b) {
        int st = train_dist(rng);
        std::vector<int> x = slice_tokens(train, st, cfg.seq_len);
        std::vector<int> y = slice_tokens(train, st + 1, cfg.seq_len);
        step_loss += model.forward_backward_one(x, y, true, loss_scale);
      }

      step_loss /= static_cast<double>(cfg.batch_size);
      running_loss += step_loss;
      running_count += 1;
      model.sgd_step(cfg.lr, cfg.grad_clip);

      if (step % cfg.print_every == 0 || step == 1 || step == cfg.steps) {
        double train_loss_avg = running_loss / static_cast<double>(running_count);
        running_loss = 0.0;
        running_count = 0;

        std::mt19937 eval_rng(cfg.seed + step);
        double val_loss = eval_loss(model, valid, cfg.eval_batches, eval_rng);

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();

        std::cout << "step=" << step << " elapsed_sec=" << elapsed << " train_loss=" << train_loss_avg
                  << " train_ppl=" << std::exp(train_loss_avg) << " valid_loss=" << val_loss
                  << " valid_ppl=" << std::exp(val_loss) << "\n";
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "time_main_compute_sec=" << total << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
