#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Params {
  std::size_t particles = 2000;
  std::size_t frames = 2000;
  std::size_t max_lag = 400;
  double dt = 0.01;
  double diffusion = 1.0;
  std::uint64_t seed = 12345;
  std::size_t fit_start_lag = 50;
  std::size_t fit_end_lag = 200;
  std::string csv_path = "msd.csv";
  bool run_checks = true;
};

struct FitResult {
  double slope = 0.0;
  double intercept = 0.0;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("Error: " + message);
}

void print_help(const char *prog) {
  std::cout
      << "Synthetic 3D trajectory MSD + diffusion baseline (serial C++17)\n\n"
      << "Usage: " << prog << " [options]\n\n"
      << "Options:\n"
      << "  --particles N       Number of particles (default: 2000)\n"
      << "  --frames F          Number of frames per particle (default: 2000)\n"
      << "  --max-lag L         Maximum lag index for MSD (default: 400)\n"
      << "  --dt DT             Time step between frames (default: 0.01)\n"
      << "  --diffusion D       Ground-truth diffusion for synthetic Brownian data\n"
      << "                     (default: 1.0)\n"
      << "  --seed S            RNG seed (default: 12345)\n"
      << "  --fit-start A       First lag index in linear fit interval (default: 50)\n"
      << "  --fit-end B         Last lag index in linear fit interval (default: 200)\n"
      << "  --csv PATH          Output CSV path (default: msd.csv)\n"
      << "  --run-checks 0|1    Run built-in numerical checks (default: 1)\n"
      << "  --help              Show this help\n";
}

std::size_t parse_size_t(const std::string &s, const std::string &name) {
  if (s.empty()) {
    fail("invalid integer for " + name + ": " + s);
  }

  for (const char ch : s) {
    if (ch < '0' || ch > '9') {
      fail("invalid unsigned integer for " + name + ": " + s);
    }
  }

  std::size_t pos = 0;
  unsigned long long v = 0;
  try {
    v = std::stoull(s, &pos, 10);
  } catch (...) {
    fail("invalid integer for " + name + ": " + s);
  }
  if (pos != s.size()) {
    fail("invalid integer for " + name + ": " + s);
  }
  if (v > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    fail("integer out of range for " + name + ": " + s);
  }
  return static_cast<std::size_t>(v);
}

double parse_double(const std::string &s, const std::string &name) {
  std::size_t pos = 0;
  double v = 0.0;
  try {
    v = std::stod(s, &pos);
  } catch (...) {
    fail("invalid floating-point number for " + name + ": " + s);
  }
  if (pos != s.size()) {
    fail("invalid floating-point number for " + name + ": " + s);
  }
  if (!std::isfinite(v)) {
    fail("non-finite floating-point value for " + name + ": " + s);
  }
  return v;
}

Params parse_args(int argc, char **argv) {
  Params p;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (arg.rfind("--", 0) != 0) {
      fail("unexpected argument: " + arg);
    }
    if (i + 1 >= argc) {
      fail("missing value for " + arg);
    }
    const std::string value = argv[++i];
    if (arg == "--particles") {
      p.particles = parse_size_t(value, arg);
    } else if (arg == "--frames") {
      p.frames = parse_size_t(value, arg);
    } else if (arg == "--max-lag") {
      p.max_lag = parse_size_t(value, arg);
    } else if (arg == "--dt") {
      p.dt = parse_double(value, arg);
    } else if (arg == "--diffusion") {
      p.diffusion = parse_double(value, arg);
    } else if (arg == "--seed") {
      p.seed = static_cast<std::uint64_t>(parse_size_t(value, arg));
    } else if (arg == "--fit-start") {
      p.fit_start_lag = parse_size_t(value, arg);
    } else if (arg == "--fit-end") {
      p.fit_end_lag = parse_size_t(value, arg);
    } else if (arg == "--csv") {
      p.csv_path = value;
    } else if (arg == "--run-checks") {
      const std::size_t v = parse_size_t(value, arg);
      if (v > 1) {
        fail("--run-checks must be 0 or 1");
      }
      p.run_checks = (v == 1);
    } else {
      fail("unknown option: " + arg);
    }
  }

  if (p.particles == 0) fail("--particles must be >= 1");
  if (p.frames < 2) fail("--frames must be >= 2");
  if (p.max_lag == 0) fail("--max-lag must be >= 1");
  if (p.max_lag >= p.frames) fail("--max-lag must be < --frames");
  if (!(p.dt > 0.0)) fail("--dt must be > 0");
  if (!(p.diffusion >= 0.0)) fail("--diffusion must be >= 0");
  if (p.fit_start_lag < 1) fail("--fit-start must be >= 1");
  if (p.fit_end_lag <= p.fit_start_lag) fail("--fit-end must be > --fit-start");
  if (p.fit_end_lag > p.max_lag) fail("--fit-end must be <= --max-lag");

  const long double entries = static_cast<long double>(p.particles) *
                              static_cast<long double>(p.frames) * 3.0L;
  if (entries > static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
    fail("trajectory size does not fit in addressable memory");
  }

  return p;
}

std::size_t idx(std::size_t particle, std::size_t frame, std::size_t coord,
                std::size_t frames) {
  return ((particle * frames) + frame) * 3 + coord;
}

std::vector<double> generate_brownian(const Params &p, std::uint64_t seed) {
  std::vector<double> traj(p.particles * p.frames * 3, 0.0);
  if (p.diffusion == 0.0) {
    return traj;
  }

  std::mt19937_64 rng(seed);
  const double sigma = std::sqrt(2.0 * p.diffusion * p.dt);
  if (sigma == 0.0) {
    return traj;
  }
  std::normal_distribution<double> normal(0.0, sigma);

  for (std::size_t particle = 0; particle < p.particles; ++particle) {
    for (std::size_t frame = 1; frame < p.frames; ++frame) {
      for (std::size_t c = 0; c < 3; ++c) {
        const std::size_t prev = idx(particle, frame - 1, c, p.frames);
        const std::size_t cur = idx(particle, frame, c, p.frames);
        traj[cur] = traj[prev] + normal(rng);
      }
    }
  }
  return traj;
}

std::vector<double> generate_stationary(const Params &p) {
  return std::vector<double>(p.particles * p.frames * 3, 0.0);
}

std::vector<double> generate_constant_velocity(const Params &p, double vx, double vy,
                                               double vz) {
  std::vector<double> traj(p.particles * p.frames * 3, 0.0);
  for (std::size_t particle = 0; particle < p.particles; ++particle) {
    for (std::size_t frame = 0; frame < p.frames; ++frame) {
      const double t = static_cast<double>(frame) * p.dt;
      traj[idx(particle, frame, 0, p.frames)] = vx * t;
      traj[idx(particle, frame, 1, p.frames)] = vy * t;
      traj[idx(particle, frame, 2, p.frames)] = vz * t;
    }
  }
  return traj;
}

std::vector<double> compute_msd(const std::vector<double> &traj, const Params &p,
                                std::vector<std::uint64_t> *counts_out) {
  std::vector<double> msd(p.max_lag + 1, 0.0);
  std::vector<std::uint64_t> counts(p.max_lag + 1, 0);
  counts[0] = static_cast<std::uint64_t>(p.particles) * p.frames;

  for (std::size_t lag = 1; lag <= p.max_lag; ++lag) {
    const std::size_t origins = p.frames - lag;
    const std::uint64_t count = static_cast<std::uint64_t>(p.particles) * origins;
    counts[lag] = count;
    double sum = 0.0;

    for (std::size_t particle = 0; particle < p.particles; ++particle) {
      for (std::size_t t0 = 0; t0 < origins; ++t0) {
        const std::size_t i0 = idx(particle, t0, 0, p.frames);
        const std::size_t j0 = idx(particle, t0 + lag, 0, p.frames);

        const double dx = traj[j0] - traj[i0];
        const double dy = traj[j0 + 1] - traj[i0 + 1];
        const double dz = traj[j0 + 2] - traj[i0 + 2];
        sum += dx * dx + dy * dy + dz * dz;
      }
    }

    msd[lag] = sum / static_cast<double>(count);
  }

  if (counts_out != nullptr) {
    *counts_out = std::move(counts);
  }
  return msd;
}

FitResult linear_fit_msd(const std::vector<double> &msd, const Params &p) {
  const std::size_t n = p.fit_end_lag - p.fit_start_lag + 1;
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;

  for (std::size_t lag = p.fit_start_lag; lag <= p.fit_end_lag; ++lag) {
    const double x = static_cast<double>(lag) * p.dt;
    const double y = msd[lag];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }

  const double nd = static_cast<double>(n);
  const double denom = nd * sxx - sx * sx;
  if (std::abs(denom) < 1e-30) {
    fail("degenerate fit interval: all x values are equal");
  }

  FitResult out;
  out.slope = (nd * sxy - sx * sy) / denom;
  out.intercept = (sy - out.slope * sx) / nd;
  return out;
}

bool check_stationary() {
  Params p;
  p.particles = 64;
  p.frames = 128;
  p.max_lag = 32;
  p.dt = 0.01;
  p.diffusion = 1.0;
  auto traj = generate_stationary(p);
  auto msd = compute_msd(traj, p, nullptr);
  double max_abs = 0.0;
  for (std::size_t lag = 1; lag <= p.max_lag; ++lag) {
    max_abs = std::max(max_abs, std::abs(msd[lag]));
  }
  const bool ok = max_abs <= 1e-14;
  std::cout << "Check stationary: max_abs_msd=" << std::setprecision(6) << std::scientific
            << max_abs << " (tol 1e-14) => " << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

bool check_constant_velocity() {
  Params p;
  p.particles = 64;
  p.frames = 128;
  p.max_lag = 32;
  p.dt = 0.02;
  const double vx = 0.4;
  const double vy = -0.1;
  const double vz = 0.2;

  auto traj = generate_constant_velocity(p, vx, vy, vz);
  auto msd = compute_msd(traj, p, nullptr);

  double max_rel = 0.0;
  const double v2 = vx * vx + vy * vy + vz * vz;
  for (std::size_t lag = 1; lag <= p.max_lag; ++lag) {
    const double expected = v2 * std::pow(p.dt * static_cast<double>(lag), 2.0);
    const double rel = std::abs(msd[lag] - expected) / std::max(expected, 1e-30);
    max_rel = std::max(max_rel, rel);
  }
  const bool ok = max_rel <= 1e-12;
  std::cout << "Check constant velocity: max_relative_error=" << std::setprecision(6)
            << std::scientific << max_rel << " (tol 1e-12) => "
            << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

bool check_brownian_expectation() {
  Params p;
  p.particles = 300;
  p.frames = 700;
  p.max_lag = 120;
  p.dt = 0.01;
  p.diffusion = 0.8;
  p.seed = 777;

  auto traj = generate_brownian(p, p.seed);
  auto msd = compute_msd(traj, p, nullptr);

  double mean_abs_rel = 0.0;
  for (std::size_t lag = 1; lag <= p.max_lag; ++lag) {
    const double t = static_cast<double>(lag) * p.dt;
    const double expected = 6.0 * p.diffusion * t;
    mean_abs_rel += std::abs(msd[lag] - expected) / expected;
  }
  mean_abs_rel /= static_cast<double>(p.max_lag);

  const double tolerance = 0.12;
  const bool ok = mean_abs_rel <= tolerance;
  std::cout << "Check Brownian expectation: mean_abs_relative_error="
            << std::setprecision(6) << std::scientific << mean_abs_rel << " (tol "
            << tolerance << ") => " << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

void write_csv(const std::string &path, const std::vector<double> &msd,
               const std::vector<std::uint64_t> &counts, const Params &p) {
  std::ofstream out(path);
  if (!out) {
    fail("failed to open CSV for writing: " + path);
  }
  out << "lag,time,msd,count,expected_brownian_msd\n";
  out << std::setprecision(17);
  for (std::size_t lag = 0; lag <= p.max_lag; ++lag) {
    const double t = static_cast<double>(lag) * p.dt;
    const double expected = 6.0 * p.diffusion * t;
    out << lag << ',' << t << ',' << msd[lag] << ',' << counts[lag] << ',' << expected
        << '\n';
  }
}

int main(int argc, char **argv) {
  try {
    const Params p = parse_args(argc, argv);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Parameters:\n"
              << "  particles      = " << p.particles << "\n"
              << "  frames         = " << p.frames << "\n"
              << "  max_lag        = " << p.max_lag << "\n"
              << "  dt             = " << p.dt << "\n"
              << "  diffusion_true = " << p.diffusion << "\n"
              << "  seed           = " << p.seed << "\n"
              << "  fit_interval   = [" << p.fit_start_lag << ", " << p.fit_end_lag
              << "] (lag index)\n"
              << "  csv            = " << p.csv_path << "\n";

    if (p.run_checks) {
      std::cout << "\nRunning built-in checks...\n";
      const bool ok1 = check_stationary();
      const bool ok2 = check_constant_velocity();
      const bool ok3 = check_brownian_expectation();
      if (!(ok1 && ok2 && ok3)) {
        std::cerr << "At least one check failed. Exiting with nonzero status.\n";
        return 1;
      } else {
        std::cout << "All checks passed.\n\n";
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<double> traj = generate_brownian(p, p.seed);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<std::uint64_t> counts;
    const std::vector<double> msd = compute_msd(traj, p, &counts);
    const auto t2 = std::chrono::steady_clock::now();

    const FitResult fit = linear_fit_msd(msd, p);
    const double d_est = fit.slope / 6.0;

    write_csv(p.csv_path, msd, counts, p);
    const auto t3 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> gen_time = t1 - t0;
    const std::chrono::duration<double> msd_time = t2 - t1;
    const std::chrono::duration<double> io_time = t3 - t2;
    const std::chrono::duration<double> total_time = t3 - t0;

    std::cout << "Timing (seconds):\n"
              << "  trajectory_generation = " << gen_time.count() << "\n"
              << "  msd_main_computation  = " << msd_time.count() << "\n"
              << "  csv_write + fit       = " << io_time.count() << "\n"
              << "  total                 = " << total_time.count() << "\n";

    std::cout << "\nDiffusion estimate (linear fit):\n"
              << "  slope       = " << fit.slope << " (MSD units / time)\n"
              << "  intercept   = " << fit.intercept << "\n"
              << "  D_est       = " << d_est << "\n"
              << "  D_true      = " << p.diffusion << "\n"
              << "  rel_error_D = "
              << (p.diffusion > 0.0 ? std::abs(d_est - p.diffusion) / p.diffusion : 0.0)
              << "\n";

    std::cout << "\nCSV written: " << p.csv_path << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
