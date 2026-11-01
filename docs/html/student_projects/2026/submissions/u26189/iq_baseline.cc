#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Params {
  std::string input_path;
  std::string input_format = "ci16";  // ci16 or cf32
  std::size_t max_samples = 0;         // 0 = read all

  std::size_t samples = 8'000'000;
  int seed = 12345;
  int sps = 8;
  double snr_db = 12.0;
  double carrier_cyc_per_sample = 0.07;

  std::size_t taps = 64;
  std::size_t decim = 4;
  std::size_t block = 256;
  int repeats = 3;
  std::size_t check_outputs = 4096;
};

void print_usage(const char* prog) {
  std::cout
      << "Usage: " << prog << " [options]\n\n"
      << "Input source:\n"
      << "  --input PATH             Read interleaved IQ from binary file\n"
      << "  --format ci16|cf32       Format for --input (default: ci16)\n"
      << "  --max-samples N          Max samples to read (0 = all, default: 0)\n\n"
      << "Synthetic input (used when --input is omitted):\n"
      << "  --samples N              Number of generated IQ samples (default: 8000000)\n"
      << "  --seed N                 RNG seed (default: 12345)\n"
      << "  --sps N                  Samples per symbol for synthetic QPSK (default: 8)\n"
      << "  --snr-db X               Synthetic SNR in dB (default: 12.0)\n"
      << "  --carrier X              Carrier cycles per sample (default: 0.07)\n\n"
      << "Processing:\n"
      << "  --taps N                 FIR taps (default: 64)\n"
      << "  --decim N                Decimation factor (default: 4)\n"
      << "  --block N                Block size for energy reduction (default: 256)\n"
      << "  --repeats N              Repeat pipeline N times (default: 3)\n"
      << "  --check-outputs N        Number of outputs for reference check (default: 4096)\n"
      << "  --help                   Show this message\n";
}

bool parse_size(const std::string& s, std::size_t& out) {
  try {
    std::size_t idx = 0;
    const unsigned long long v = std::stoull(s, &idx, 10);
    if (idx != s.size()) return false;
    out = static_cast<std::size_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_int(const std::string& s, int& out) {
  try {
    std::size_t idx = 0;
    const long v = std::stol(s, &idx, 10);
    if (idx != s.size()) return false;
    out = static_cast<int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string& s, double& out) {
  try {
    std::size_t idx = 0;
    out = std::stod(s, &idx);
    return idx == s.size();
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char** argv, Params& p) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return false;
      }
      return true;
    };

    if (arg == "--help") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--input") {
      if (!need_value("--input")) return false;
      p.input_path = argv[++i];
    } else if (arg == "--format") {
      if (!need_value("--format")) return false;
      p.input_format = argv[++i];
      if (p.input_format != "ci16" && p.input_format != "cf32") {
        std::cerr << "--format must be ci16 or cf32\n";
        return false;
      }
    } else if (arg == "--max-samples") {
      if (!need_value("--max-samples")) return false;
      if (!parse_size(argv[++i], p.max_samples)) {
        std::cerr << "Invalid --max-samples value\n";
        return false;
      }
    } else if (arg == "--samples") {
      if (!need_value("--samples")) return false;
      if (!parse_size(argv[++i], p.samples)) {
        std::cerr << "Invalid --samples value\n";
        return false;
      }
    } else if (arg == "--seed") {
      if (!need_value("--seed")) return false;
      if (!parse_int(argv[++i], p.seed)) {
        std::cerr << "Invalid --seed value\n";
        return false;
      }
    } else if (arg == "--sps") {
      if (!need_value("--sps")) return false;
      if (!parse_int(argv[++i], p.sps)) {
        std::cerr << "Invalid --sps value\n";
        return false;
      }
    } else if (arg == "--snr-db") {
      if (!need_value("--snr-db")) return false;
      if (!parse_double(argv[++i], p.snr_db)) {
        std::cerr << "Invalid --snr-db value\n";
        return false;
      }
    } else if (arg == "--carrier") {
      if (!need_value("--carrier")) return false;
      if (!parse_double(argv[++i], p.carrier_cyc_per_sample)) {
        std::cerr << "Invalid --carrier value\n";
        return false;
      }
    } else if (arg == "--taps") {
      if (!need_value("--taps")) return false;
      if (!parse_size(argv[++i], p.taps)) {
        std::cerr << "Invalid --taps value\n";
        return false;
      }
    } else if (arg == "--decim") {
      if (!need_value("--decim")) return false;
      if (!parse_size(argv[++i], p.decim)) {
        std::cerr << "Invalid --decim value\n";
        return false;
      }
    } else if (arg == "--block") {
      if (!need_value("--block")) return false;
      if (!parse_size(argv[++i], p.block)) {
        std::cerr << "Invalid --block value\n";
        return false;
      }
    } else if (arg == "--repeats") {
      if (!need_value("--repeats")) return false;
      if (!parse_int(argv[++i], p.repeats)) {
        std::cerr << "Invalid --repeats value\n";
        return false;
      }
    } else if (arg == "--check-outputs") {
      if (!need_value("--check-outputs")) return false;
      if (!parse_size(argv[++i], p.check_outputs)) {
        std::cerr << "Invalid --check-outputs value\n";
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (p.taps < 2 || p.decim < 1 || p.block < 1 || p.repeats < 1 || p.sps < 1) {
    std::cerr << "Invalid parameters: taps>=2, decim>=1, block>=1, repeats>=1, sps>=1 required\n";
    return false;
  }
  return true;
}

std::vector<float> make_lowpass_taps(std::size_t taps, std::size_t decim) {
  std::vector<float> h(taps, 0.0f);
  const double fc = 0.45 / static_cast<double>(decim);
  const double m = 0.5 * static_cast<double>(taps - 1);
  double sum = 0.0;
  for (std::size_t n = 0; n < taps; ++n) {
    const double x = static_cast<double>(n) - m;
    const double sinc = (std::abs(x) < 1e-12)
                            ? 2.0 * fc
                            : std::sin(2.0 * kPi * fc * x) / (kPi * x);
    const double win = 0.54 - 0.46 * std::cos(2.0 * kPi * static_cast<double>(n) /
                                              static_cast<double>(taps - 1));
    const double v = sinc * win;
    h[n] = static_cast<float>(v);
    sum += v;
  }
  for (float& v : h) v = static_cast<float>(v / sum);
  return h;
}

bool load_iq_file(const Params& p, std::vector<float>& i, std::vector<float>& q) {
  std::ifstream fin(p.input_path, std::ios::binary);
  if (!fin) {
    std::cerr << "Failed to open input file: " << p.input_path << "\n";
    return false;
  }

  std::size_t estimate = 0;
  try {
    const auto bytes = std::filesystem::file_size(p.input_path);
    const std::size_t bytes_per_sample = (p.input_format == "ci16") ? 4 : 8;
    estimate = static_cast<std::size_t>(bytes / bytes_per_sample);
  } catch (...) {
  }
  if (p.max_samples > 0 && (estimate == 0 || estimate > p.max_samples)) {
    estimate = p.max_samples;
  }
  if (estimate > 0) {
    i.reserve(estimate);
    q.reserve(estimate);
  }

  if (p.input_format == "ci16") {
    int16_t pair[2] = {0, 0};
    while (fin.read(reinterpret_cast<char*>(pair), sizeof(pair))) {
      i.push_back(static_cast<float>(pair[0]) * (1.0f / 32768.0f));
      q.push_back(static_cast<float>(pair[1]) * (1.0f / 32768.0f));
      if (p.max_samples > 0 && i.size() >= p.max_samples) break;
    }
  } else {
    float pair[2] = {0.0f, 0.0f};
    while (fin.read(reinterpret_cast<char*>(pair), sizeof(pair))) {
      i.push_back(pair[0]);
      q.push_back(pair[1]);
      if (p.max_samples > 0 && i.size() >= p.max_samples) break;
    }
  }

  if (i.empty()) {
    std::cerr << "No IQ samples read from input\n";
    return false;
  }
  return true;
}

void generate_synthetic_iq(const Params& p, std::vector<float>& i, std::vector<float>& q) {
  i.resize(p.samples);
  q.resize(p.samples);

  std::mt19937 rng(static_cast<std::mt19937::result_type>(p.seed));
  std::uniform_int_distribution<int> symdist(0, 3);
  std::normal_distribution<double> gauss(0.0, 1.0);

  const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
  const double symbols_i[4] = {inv_sqrt2, inv_sqrt2, -inv_sqrt2, -inv_sqrt2};
  const double symbols_q[4] = {inv_sqrt2, -inv_sqrt2, inv_sqrt2, -inv_sqrt2};

  const double n0 = std::pow(10.0, -p.snr_db / 10.0);
  const double sigma = std::sqrt(0.5 * n0);
  const double w = 2.0 * kPi * p.carrier_cyc_per_sample;
  const double interferer_w = 2.0 * kPi * 0.173;

  int sym = 0;
  for (std::size_t n = 0; n < p.samples; ++n) {
    if (n % static_cast<std::size_t>(p.sps) == 0) {
      sym = symdist(rng);
    }
    const double t = static_cast<double>(n);
    const double c = std::cos(w * t);
    const double s = std::sin(w * t);

    const double si = symbols_i[sym];
    const double sq = symbols_q[sym];
    const double mod_i = si * c - sq * s;
    const double mod_q = si * s + sq * c;

    const double tone_i = 0.20 * std::cos(interferer_w * t);
    const double tone_q = 0.20 * std::sin(interferer_w * t);

    const double ni = sigma * gauss(rng);
    const double nq = sigma * gauss(rng);

    i[n] = static_cast<float>(mod_i + tone_i + ni);
    q[n] = static_cast<float>(mod_q + tone_q + nq);
  }
}

struct ComputeResult {
  double checksum = 0.0;
  double energy_sum = 0.0;
  std::size_t hits = 0;
  std::size_t out_samples = 0;
  double seconds = 0.0;
};

ComputeResult run_pipeline(const std::vector<float>& in_i, const std::vector<float>& in_q,
                           const std::vector<float>& taps, const Params& p) {
  const std::size_t n = in_i.size();
  const std::size_t t = taps.size();
  const std::size_t m = (n < t) ? 0 : (n - t) / p.decim + 1;
  const std::size_t blocks = m / p.block;

  std::vector<float> out_i(m, 0.0f);
  std::vector<float> out_q(m, 0.0f);

  ComputeResult r;
  r.out_samples = m;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int rep = 0; rep < p.repeats; ++rep) {
    for (std::size_t out = 0, base = 0; out < m; ++out, base += p.decim) {
      float acc_i = 0.0f;
      float acc_q = 0.0f;
      for (std::size_t k = 0; k < t; ++k) {
        const float hk = taps[k];
        acc_i += hk * in_i[base + k];
        acc_q += hk * in_q[base + k];
      }
      out_i[out] = acc_i;
      out_q[out] = acc_q;
    }

    double energy_sum = 0.0;
    for (std::size_t out = 0; out < m; ++out) {
      const double ii = out_i[out];
      const double qq = out_q[out];
      const double e = ii * ii + qq * qq;
      energy_sum += e;
      r.checksum += (ii * static_cast<double>((out % 13) + 1) +
                     qq * static_cast<double>((out % 17) + 1));
    }

    const double avg_per_sample = (m > 0) ? (energy_sum / static_cast<double>(m)) : 0.0;
    const double threshold = avg_per_sample * static_cast<double>(p.block);
    std::size_t hits = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
      double block_e = 0.0;
      const std::size_t begin = b * p.block;
      const std::size_t end = begin + p.block;
      for (std::size_t idx = begin; idx < end; ++idx) {
        const double ii = out_i[idx];
        const double qq = out_q[idx];
        block_e += ii * ii + qq * qq;
      }
      if (block_e > threshold) ++hits;
    }

    r.energy_sum += energy_sum;
    r.hits += hits;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  r.seconds = std::chrono::duration<double>(t1 - t0).count();

  return r;
}

struct CheckResult {
  std::size_t checked = 0;
  double max_abs_err_i = 0.0;
  double max_abs_err_q = 0.0;
  double rmse = 0.0;
};

CheckResult check_outputs(const std::vector<float>& in_i, const std::vector<float>& in_q,
                          const std::vector<float>& taps, const Params& p) {
  const std::size_t n = in_i.size();
  const std::size_t t = taps.size();
  const std::size_t m = (n < t) ? 0 : (n - t) / p.decim + 1;
  const std::size_t check_n = std::min(m, p.check_outputs);

  CheckResult r;
  r.checked = check_n;
  if (check_n == 0) return r;

  double se_sum = 0.0;
  for (std::size_t out = 0, base = 0; out < check_n; ++out, base += p.decim) {
    float fi = 0.0f;
    float fq = 0.0f;
    for (std::size_t k = 0; k < t; ++k) {
      const float hk = taps[k];
      fi += hk * in_i[base + k];
      fq += hk * in_q[base + k];
    }

    double di = 0.0;
    double dq = 0.0;
    for (std::size_t k = 0; k < t; ++k) {
      const double hk = static_cast<double>(taps[k]);
      di += hk * static_cast<double>(in_i[base + k]);
      dq += hk * static_cast<double>(in_q[base + k]);
    }

    const double ei = std::abs(static_cast<double>(fi) - di);
    const double eq = std::abs(static_cast<double>(fq) - dq);
    r.max_abs_err_i = std::max(r.max_abs_err_i, ei);
    r.max_abs_err_q = std::max(r.max_abs_err_q, eq);
    se_sum += ei * ei + eq * eq;
  }

  r.rmse = std::sqrt(se_sum / static_cast<double>(2 * check_n));
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  bool help_requested = false;
  for (int k = 1; k < argc; ++k) {
    if (std::string(argv[k]) == "--help") {
      help_requested = true;
      break;
    }
  }

  Params p;
  if (!parse_args(argc, argv, p)) {
    if (!help_requested) {
      std::cerr << "Use --help for options.\n";
      return 1;
    }
    return 0;
  }

  std::vector<float> i;
  std::vector<float> q;

  if (!p.input_path.empty()) {
    if (!load_iq_file(p, i, q)) return 1;
    std::cout << "Input source: file\n";
    std::cout << "  path: " << p.input_path << "\n";
    std::cout << "  format: " << p.input_format << "\n";
  } else {
    generate_synthetic_iq(p, i, q);
    std::cout << "Input source: synthetic QPSK + tone + noise\n";
    std::cout << "  seed: " << p.seed << "\n";
    std::cout << "  snr_db: " << p.snr_db << "\n";
    std::cout << "  sps: " << p.sps << "\n";
  }

  if (i.size() < p.taps) {
    std::cerr << "Need at least --taps samples, but got " << i.size() << "\n";
    return 1;
  }

  const std::vector<float> taps = make_lowpass_taps(p.taps, p.decim);
  const ComputeResult cr = run_pipeline(i, q, taps, p);
  const CheckResult ck = check_outputs(i, q, taps, p);

  const double out_samples = static_cast<double>(cr.out_samples);
  const double fir_ops = 8.0 * static_cast<double>(p.taps) * out_samples *
                         static_cast<double>(p.repeats);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Samples in: " << i.size() << "\n";
  std::cout << "Samples out per repeat: " << cr.out_samples << "\n";
  std::cout << "Taps: " << p.taps << ", Decimation: " << p.decim
            << ", Block: " << p.block << ", Repeats: " << p.repeats << "\n";
  std::cout << "Main compute time [s]: " << cr.seconds << "\n";
  std::cout << "Approx FIR throughput [GFLOP/s]: " << (fir_ops / cr.seconds) * 1e-9 << "\n";
  std::cout << "Energy sum (all repeats): " << cr.energy_sum << "\n";
  std::cout << "Block-hit count (all repeats): " << cr.hits << "\n";
  std::cout << "Checksum: " << cr.checksum << "\n";
  std::cout << "Check outputs: " << ck.checked << "\n";
  std::cout << "Check max_abs_err I: " << ck.max_abs_err_i
            << ", Q: " << ck.max_abs_err_q << "\n";
  std::cout << "Check RMSE: " << ck.rmse << "\n";

  return 0;
}
