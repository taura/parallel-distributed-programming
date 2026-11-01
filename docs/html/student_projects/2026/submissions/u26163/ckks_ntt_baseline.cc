#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

u64 mod_pow(u64 a, u64 e, u64 mod) {
  u64 r = 1;
  while (e > 0) {
    if (e & 1ULL) {
      r = static_cast<u64>((static_cast<u128>(r) * a) % mod);
    }
    a = static_cast<u64>((static_cast<u128>(a) * a) % mod);
    e >>= 1ULL;
  }
  return r;
}

u64 mod_inv(u64 a, u64 mod) {
  return mod_pow(a, mod - 2, mod);
}

bool is_power_of_two(u64 n) {
  return n > 0 && (n & (n - 1)) == 0;
}

std::vector<u64> factorize(u64 n) {
  std::vector<u64> factors;
  for (u64 p = 2; p * p <= n; ++p) {
    if (n % p == 0) {
      factors.push_back(p);
      while (n % p == 0) {
        n /= p;
      }
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

u64 primitive_root_prime(u64 mod) {
  const u64 phi = mod - 1;
  const std::vector<u64> factors = factorize(phi);
  for (u64 g = 2; g < mod; ++g) {
    bool ok = true;
    for (u64 p : factors) {
      if (mod_pow(g, phi / p, mod) == 1) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return g;
    }
  }
  throw std::runtime_error("failed to find primitive root");
}

struct NTTContext {
  u64 n;
  u64 mod;
  u64 omega;
  u64 inv_omega;
  u64 inv_n;
  std::vector<std::size_t> bitrev;
  std::vector<u64> roots;
  std::vector<u64> iroots;
  std::vector<u64> psi_pows;
  std::vector<u64> inv_psi_pows;

  NTTContext(u64 n_, u64 mod_) : n(n_), mod(mod_) {
    if (!is_power_of_two(n)) {
      throw std::runtime_error("N must be a power of two");
    }
    if ((mod - 1) % (2 * n) != 0) {
      throw std::runtime_error("modulus does not support negacyclic NTT for this N");
    }

    const u64 g = primitive_root_prime(mod);
    const u64 psi = mod_pow(g, (mod - 1) / (2 * n), mod);
    if (mod_pow(psi, n, mod) != mod - 1) {
      throw std::runtime_error("failed to construct 2N-th primitive root");
    }

    omega = static_cast<u64>((static_cast<u128>(psi) * psi) % mod);
    inv_omega = mod_inv(omega, mod);
    inv_n = mod_inv(n, mod);

    bitrev.resize(n);
    int lg = 0;
    for (u64 x = n; x > 1; x >>= 1ULL) {
      ++lg;
    }
    for (u64 i = 0; i < n; ++i) {
      u64 x = i;
      u64 y = 0;
      for (int b = 0; b < lg; ++b) {
        y = (y << 1) | (x & 1ULL);
        x >>= 1ULL;
      }
      bitrev[i] = static_cast<std::size_t>(y);
    }

    roots.resize(n);
    iroots.resize(n);
    roots[0] = 1;
    iroots[0] = 1;
    for (u64 i = 1; i < n; ++i) {
      roots[i] = static_cast<u64>((static_cast<u128>(roots[i - 1]) * omega) % mod);
      iroots[i] = static_cast<u64>((static_cast<u128>(iroots[i - 1]) * inv_omega) % mod);
    }

    psi_pows.resize(n);
    inv_psi_pows.resize(n);
    psi_pows[0] = 1;
    inv_psi_pows[0] = 1;
    const u64 inv_psi = mod_inv(psi, mod);
    for (u64 i = 1; i < n; ++i) {
      psi_pows[i] = static_cast<u64>((static_cast<u128>(psi_pows[i - 1]) * psi) % mod);
      inv_psi_pows[i] =
          static_cast<u64>((static_cast<u128>(inv_psi_pows[i - 1]) * inv_psi) % mod);
    }
  }

  void ntt(std::vector<u64>& a, bool inverse) const {
    for (u64 i = 0; i < n; ++i) {
      const std::size_t j = bitrev[i];
      if (i < j) {
        std::swap(a[i], a[j]);
      }
    }

    for (u64 len = 2; len <= n; len <<= 1ULL) {
      const u64 half = len >> 1ULL;
      const u64 step = n / len;
      for (u64 i = 0; i < n; i += len) {
        for (u64 j = 0; j < half; ++j) {
          const u64 tw = inverse ? iroots[j * step] : roots[j * step];
          const u64 u = a[i + j];
          const u64 v = static_cast<u64>((static_cast<u128>(a[i + j + half]) * tw) % mod);
          u64 x = u + v;
          if (x >= mod) {
            x -= mod;
          }
          u64 y = (u >= v) ? (u - v) : (u + mod - v);
          a[i + j] = x;
          a[i + j + half] = y;
        }
      }
    }

    if (inverse) {
      for (u64 i = 0; i < n; ++i) {
        a[i] = static_cast<u64>((static_cast<u128>(a[i]) * inv_n) % mod);
      }
    }
  }

  void multiply_negacyclic(const std::vector<u64>& a,
                          const std::vector<u64>& b,
                          std::vector<u64>& out) const {
    std::vector<u64> fa(n), fb(n);
    for (u64 i = 0; i < n; ++i) {
      fa[i] = static_cast<u64>((static_cast<u128>(a[i]) * psi_pows[i]) % mod);
      fb[i] = static_cast<u64>((static_cast<u128>(b[i]) * psi_pows[i]) % mod);
    }

    ntt(fa, false);
    ntt(fb, false);

    for (u64 i = 0; i < n; ++i) {
      fa[i] = static_cast<u64>((static_cast<u128>(fa[i]) * fb[i]) % mod);
    }

    ntt(fa, true);
    out.resize(n);
    for (u64 i = 0; i < n; ++i) {
      out[i] = static_cast<u64>((static_cast<u128>(fa[i]) * inv_psi_pows[i]) % mod);
    }
  }
};

struct Args {
  u64 logn = 14;
  u64 mods = 2;
  u64 reps = 5;
  u64 seed = 12345;
  u64 verify_coeffs = 64;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need_value = [&](const std::string& key) {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return std::string(argv[++i]);
    };

    if (s == "--logn") {
      args.logn = std::stoull(need_value(s));
    } else if (s == "--mods") {
      args.mods = std::stoull(need_value(s));
    } else if (s == "--reps") {
      args.reps = std::stoull(need_value(s));
    } else if (s == "--seed") {
      args.seed = std::stoull(need_value(s));
    } else if (s == "--verify-coeffs") {
      args.verify_coeffs = std::stoull(need_value(s));
    } else if (s == "--help") {
      std::cout
          << "Usage: ./ckks_ntt_baseline [--logn L] [--mods M] [--reps R] [--seed S] "
             "[--verify-coeffs K]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + s);
    }
  }
  return args;
}

u64 naive_negacyclic_coeff(const std::vector<u64>& a,
                          const std::vector<u64>& b,
                          u64 k,
                          u64 mod) {
  const u64 n = static_cast<u64>(a.size());
  u64 acc = 0;
  for (u64 i = 0; i <= k; ++i) {
    acc = (acc + static_cast<u64>((static_cast<u128>(a[i]) * b[k - i]) % mod)) % mod;
  }
  for (u64 i = k + 1; i < n; ++i) {
    const u64 term = static_cast<u64>((static_cast<u128>(a[i]) * b[n + k - i]) % mod);
    acc = (acc >= term) ? (acc - term) : (acc + mod - term);
  }
  return acc;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const u64 n = 1ULL << args.logn;

    const std::vector<u64> candidate_primes = {
        2013265921ULL, 1811939329ULL, 469762049ULL,
        1224736769ULL, 1004535809ULL, 998244353ULL,
    };

    std::vector<u64> moduli;
    for (u64 q : candidate_primes) {
      if ((q - 1) % (2 * n) == 0) {
        moduli.push_back(q);
      }
      if (moduli.size() == args.mods) {
        break;
      }
    }
    if (moduli.size() < args.mods) {
      throw std::runtime_error(
          "not enough built-in NTT primes for this N and --mods. Try smaller --logn or --mods.");
    }

    std::mt19937_64 rng(args.seed);
    std::vector<NTTContext> contexts;
    contexts.reserve(moduli.size());

    std::vector<std::vector<u64>> a_all(moduli.size(), std::vector<u64>(n));
    std::vector<std::vector<u64>> b_all(moduli.size(), std::vector<u64>(n));
    std::vector<std::vector<u64>> c_all(moduli.size(), std::vector<u64>(n));

    for (std::size_t m = 0; m < moduli.size(); ++m) {
      const u64 q = moduli[m];
      contexts.emplace_back(n, q);
      std::uniform_int_distribution<u64> dist(0, q - 1);
      for (u64 i = 0; i < n; ++i) {
        a_all[m][i] = dist(rng);
        b_all[m][i] = dist(rng);
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (u64 rep = 0; rep < args.reps; ++rep) {
      for (std::size_t m = 0; m < moduli.size(); ++m) {
        contexts[m].multiply_negacyclic(a_all[m], b_all[m], c_all[m]);
        if (rep + 1 < args.reps) {
          std::rotate(a_all[m].begin(), a_all[m].begin() + 1, a_all[m].end());
          std::rotate(b_all[m].begin(), b_all[m].begin() + 3, b_all[m].end());
        }
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    const u64 check_mod = moduli[0];
    const u64 kmax = std::min<u64>(args.verify_coeffs, n);
    u64 mismatches = 0;
    for (u64 k = 0; k < kmax; ++k) {
      const u64 ref = naive_negacyclic_coeff(a_all[0], b_all[0], k, check_mod);
      if (ref != c_all[0][k]) {
        ++mismatches;
      }
    }

    u64 checksum = 0;
    for (std::size_t m = 0; m < moduli.size(); ++m) {
      for (u64 i = 0; i < n; ++i) {
        checksum ^= (c_all[m][i] + 0x9e3779b97f4a7c15ULL + (checksum << 6) + (checksum >> 2));
      }
    }

    std::cout << "CKKS-style negacyclic NTT baseline (serial)\n";
    std::cout << "N=" << n << " (logn=" << args.logn << "), moduli=" << moduli.size()
              << ", reps=" << args.reps << "\n";
    std::cout << "moduli:";
    for (u64 q : moduli) {
      std::cout << ' ' << q;
    }
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "compute_time_sec=" << elapsed_s << "\n";
    std::cout << "verify_first_k=" << kmax << ", mismatches=" << mismatches << "\n";
    std::cout << "checksum=" << checksum << "\n";

    return (mismatches == 0) ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
