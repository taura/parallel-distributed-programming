#include "attention.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct VerifyArgs {
  int64_t seq_len = 32;
  int64_t dim = 64;
  int64_t heads = 8;
  uint32_t seed = 42;
  double tol = 1e-9;
};

VerifyArgs parse_args(int argc, char **argv) {
  VerifyArgs args;
  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    auto need_value = [&](const char *name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };

    if (token == "--seq") {
      args.seq_len = std::stoll(need_value("--seq"));
    } else if (token == "--dim") {
      args.dim = std::stoll(need_value("--dim"));
    } else if (token == "--heads") {
      args.heads = std::stoll(need_value("--heads"));
    } else if (token == "--seed") {
      args.seed = static_cast<uint32_t>(std::stoul(need_value("--seed")));
    } else if (token == "--tol") {
      args.tol = std::stod(need_value("--tol"));
    } else if (token == "--help") {
      std::cout << "Usage: ./attention_verify [--seq N] [--dim D] [--heads H]"
                   " [--seed S] [--tol T]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + token);
    }
  }
  if (args.seq_len <= 0 || args.dim <= 0 || args.heads <= 0) {
    throw std::runtime_error("--seq, --dim, --heads must be positive");
  }
  if (args.dim % args.heads != 0) {
    throw std::runtime_error("--dim must be divisible by --heads");
  }
  if (args.tol <= 0.0) {
    throw std::runtime_error("--tol must be positive");
  }
  return args;
}

Tensor attention_reference(const Tensor &q, const Tensor &k, const Tensor &v,
                           int64_t heads, double *head0_row0_sum) {
  if (q.shape != k.shape || q.shape != v.shape) {
    throw std::runtime_error("q, k, v must have the same shape");
  }

  const int64_t seq_len = q.shape[0];
  const int64_t dim = q.shape[1];
  const int64_t head_dim = dim / heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  Tensor out({seq_len, dim});
  std::vector<double> logits(static_cast<size_t>(seq_len));

  for (int64_t h = 0; h < heads; ++h) {
    const int64_t off = h * head_dim;
    for (int64_t i = 0; i < seq_len; ++i) {
      double row_max = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < seq_len; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += q.at2(i, off + d) * k.at2(j, off + d);
        }
        const double s = dot * scale;
        logits[static_cast<size_t>(j)] = s;
        if (s > row_max) {
          row_max = s;
        }
      }

      double denom = 0.0;
      for (int64_t j = 0; j < seq_len; ++j) {
        const double e = std::exp(logits[static_cast<size_t>(j)] - row_max);
        logits[static_cast<size_t>(j)] = e;
        denom += e;
      }

      const double inv = 1.0 / denom;
      if (h == 0 && i == 0 && head0_row0_sum != nullptr) {
        double s = 0.0;
        for (int64_t j = 0; j < seq_len; ++j) {
          s += logits[static_cast<size_t>(j)] * inv;
        }
        *head0_row0_sum = s;
      }

      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j < seq_len; ++j) {
          const double p = logits[static_cast<size_t>(j)] * inv;
          acc += p * v.at2(j, off + d);
        }
        out.at2(i, off + d) = acc;
      }
    }
  }

  return out;
}

double max_abs_diff(const Tensor &a, const Tensor &b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("shape mismatch in max_abs_diff");
  }
  double m = 0.0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    const double d = std::abs(a.data[i] - b.data[i]);
    if (d > m) {
      m = d;
    }
  }
  return m;
}

int main(int argc, char **argv) {
  try {
    const VerifyArgs args = parse_args(argc, argv);

    Tensor q({args.seq_len, args.dim});
    Tensor k({args.seq_len, args.dim});
    Tensor v({args.seq_len, args.dim});

    std::mt19937 rng(args.seed);
    fill_random(q, rng, -1.0, 1.0);
    fill_random(k, rng, -1.0, 1.0);
    fill_random(v, rng, -1.0, 1.0);

    double row0_fast = 0.0;
    double row0_ref = 0.0;
    Tensor out_fast = attention(q, k, v, args.heads, &row0_fast);
    Tensor out_ref = attention_reference(q, k, v, args.heads, &row0_ref);

    const double err = max_abs_diff(out_fast, out_ref);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "seq=" << args.seq_len << " dim=" << args.dim
              << " heads=" << args.heads << " seed=" << args.seed << "\n";
    std::cout << "max_abs_diff=" << err << "\n";
    std::cout << "fast_checksum=" << checksum(out_fast) << "\n";
    std::cout << "ref_checksum=" << checksum(out_ref) << "\n";
    std::cout << "head0_row0_sum_fast=" << row0_fast << "\n";
    std::cout << "head0_row0_sum_ref=" << row0_ref << "\n";

    if (err <= args.tol) {
      std::cout << "verification=PASS\n";
      return 0;
    }

    std::cout << "verification=FAIL\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
