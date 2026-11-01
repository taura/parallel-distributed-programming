#include "attention.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

struct Args {
  int64_t seq_len = 512;
  int64_t dim = 256;
  int64_t heads = 8;
  uint32_t seed = 42;
};

Args parse_args(int argc, char **argv) {
  Args args;
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
    } else if (token == "--help") {
      std::cout
          << "Usage: ./attention_app [--seq N] [--dim D] [--heads H] [--seed S]\n";
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
  return args;
}

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);

    Tensor q({args.seq_len, args.dim});
    Tensor k({args.seq_len, args.dim});
    Tensor v({args.seq_len, args.dim});

    std::mt19937 rng(args.seed);
    fill_random(q, rng, -1.0, 1.0);
    fill_random(k, rng, -1.0, 1.0);
    fill_random(v, rng, -1.0, 1.0);

    const auto t0 = std::chrono::steady_clock::now();
    double row0 = 0.0;
    Tensor out = attention(q, k, v, args.heads, &row0);
    const auto t1 = std::chrono::steady_clock::now();

    const double sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "seq=" << args.seq_len << " dim=" << args.dim
              << " heads=" << args.heads << " seed=" << args.seed << "\n";
    std::cout << "compute_time_sec=" << sec << "\n";
    std::cout << "output_checksum=" << checksum(out) << "\n";
    std::cout << "head0_row0_sum_after_softmax=" << row0 << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
