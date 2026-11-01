#include "ckks.h"
#include "ntt.h"
#include "polynomial.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace ckks {

static constexpr double PI = 3.14159265358979323846;

static int64_t normalize(int64_t x, int64_t q) {
    return (x % q + q) % q;
}

static std::vector<int64_t> add(const std::vector<int64_t>& a, const std::vector<int64_t>& b, int64_t q) {
    std::vector<int64_t> c(a.size());
    for (size_t i = 0; i < a.size(); ++i) c[i] = normalize(a[i] + b[i], q);
    return c;
}

static std::vector<int64_t> sample_small(size_t N, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(-1, 1);
    std::vector<int64_t> a(N);
    for (auto& x : a) x = dist(rng);
    return a;
}

static std::vector<int64_t> sample_uniform(size_t N, int64_t q, std::mt19937_64& rng) {
    std::uniform_int_distribution<int64_t> dist(0, q - 1);
    std::vector<int64_t> a(N);
    for (auto& x : a) x = dist(rng);
    return a;
}

// ----- Constructor -----
CKKS::CKKS(int64_t N, int64_t MOD, int64_t PRIM_ROOT)
    : N_(N), MOD_(MOD), PRIM_ROOT_(PRIM_ROOT) {
    ntt::set_params(N, MOD, PRIM_ROOT);
}

// ----- Encoding -----
std::vector<int64_t> CKKS::encode(const std::vector<double>& plain, double scale) const {
    if (plain.size() > static_cast<size_t>(N_ / 2)) throw std::invalid_argument("Too many slots");

    std::vector<std::complex<double>> values(N_);

    // Canonical embedding の共役対称性
    for (size_t i = 0; i < plain.size(); ++i) {
        values[i] = plain[i];
        values[N_ - 1 - i] = plain[i];
    }

    std::vector<int64_t> poly(N_);

    // 逆 canonical embedding（直接計算）
    for (int64_t j = 0; j < N_; ++j) {
        std::complex<double> sum = 0;
        for (int64_t k = 0; k < N_; ++k) {
            double angle = -PI * (2 * k + 1) * j / N_;
            sum += values[k] * std::polar(1.0, angle);
        }
        double coeff = sum.real() / N_;
        poly[j] = normalize(static_cast<int64_t>(std::llround(coeff * scale)), MOD_);
    }

    return poly;
}

// ----- Decoding -----
std::vector<double> CKKS::decode(const std::vector<int64_t>& poly, double scale, size_t slots) const {
    std::vector<double> result(slots);

    for (size_t k = 0; k < slots; ++k) {
        std::complex<double> sum = 0;
        for (int64_t j = 0; j < N_; ++j) {
            int64_t coeff = poly[j] > MOD_ / 2 ? poly[j] - MOD_ : poly[j];
            double angle = PI * (2 * k + 1) * j / N_;
            sum += static_cast<double>(coeff) * std::polar(1.0, angle);
        }
        result[k] = sum.real() / scale;
    }

    return result;
}

// ----- Key Generation -----
void CKKS::generateKeyPair() {
    std::random_device rd;
    std::mt19937_64 rng(rd());

    secret_key_ = sample_small(N_, rng);
    public_key_a_ = sample_uniform(N_, MOD_, rng);
    auto e = sample_small(N_, rng);

    auto as = polynomial::multiply(public_key_a_, secret_key_);
    public_key_b_.resize(N_);

    for (int64_t i = 0; i < N_; ++i) {
        public_key_b_[i] = normalize(-as[i] + e[i], MOD_);
    }
}

// ----- Encryption -----
Ciphertext CKKS::encrypt(const std::vector<double>& plain, double scale) const {
    if (secret_key_.empty()) throw std::runtime_error("Generate keys first");

    std::random_device rd;
    std::mt19937_64 rng(rd());

    auto m = encode(plain, scale);
    auto u = sample_small(N_, rng);
    auto e0 = sample_small(N_, rng);
    auto e1 = sample_small(N_, rng);

    auto bu = polynomial::multiply(public_key_b_, u);
    auto au = polynomial::multiply(public_key_a_, u);

    Ciphertext ct;
    ct.c0.resize(N_);
    ct.c1.resize(N_);
    ct.scale = scale;

    for (int64_t i = 0; i < N_; ++i) {
        ct.c0[i] = normalize(bu[i] + e0[i] + m[i], MOD_);
        ct.c1[i] = normalize(au[i] + e1[i], MOD_);
    }

    return ct;
}

// ----- Decryption -----
std::vector<double> CKKS::decrypt(const Ciphertext& cipher, size_t slots) const {
    if (secret_key_.empty()) throw std::runtime_error("Generate keys first");

    auto c1s = polynomial::multiply(cipher.c1, secret_key_);
    auto m = add(cipher.c0, c1s, MOD_);

    return decode(m, cipher.scale, slots);
}

} // namespace ckks