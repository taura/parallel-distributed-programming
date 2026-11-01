#include "ckks.h"
#include "ntt.h"
#include "polynomial.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdexcept>

int main() {
    const int64_t N = 1024;
    const int64_t MOD = 2013265921;
    const int64_t PRIM_ROOT = 31;
    const double SCALE = static_cast<double>(1LL << 20);

    ntt::set_params(N, MOD, PRIM_ROOT);

    // ----- Test 1: NTT / INTT -----
    std::vector<int64_t> a(N);
    for (int64_t i = 0; i < N; ++i) a[i] = i % MOD;

    auto original = a;
    ntt::ntt(a, false);
    ntt::ntt(a, true);

    bool ntt_ok = (a == original);
    std::cout << "[NTT] " << (ntt_ok ? "PASS" : "FAIL") << "\n";

    // ----- Test 2: Negacyclic Polynomial Multiplication -----
    std::vector<int64_t> p(N, 0), q(N, 0);
    p[N - 1] = 1; // X^(N-1)
    q[1] = 1;     // X

    auto product = polynomial::multiply(p, q);

    bool poly_ok = (product[0] == MOD - 1);
    for (int64_t i = 1; i < N; ++i) poly_ok &= (product[i] == 0);

    std::cout << "[Polynomial] " << (poly_ok ? "PASS" : "FAIL") << "\n";

    // ----- Test 3: CKKS Encrypt / Decrypt -----
    ckks::CKKS scheme(N, MOD, PRIM_ROOT);
    scheme.generateKeyPair();

    std::vector<double> plain = {1.2, -0.5, 3.0, 0.75, -2.25};

    auto t0 = std::chrono::high_resolution_clock::now();
    ckks::Ciphertext cipher = scheme.encrypt(plain, SCALE);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto recovered = scheme.decrypt(cipher, plain.size());
    auto t2 = std::chrono::high_resolution_clock::now();

    double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double dec_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "[CKKS] Encryption time : " << enc_ms << " ms\n";
    std::cout << "[CKKS] Decryption time : " << dec_ms << " ms\n";

    double max_error = 0.0;
    for (size_t i = 0; i < plain.size(); ++i) {
        double error = std::abs(plain[i] - recovered[i]);
        max_error = std::max(max_error, error);
        std::cout << "  " << plain[i] << " -> " << recovered[i] << " (error=" << error << ")\n";
    }

    const double TOLERANCE = 1e-2;
    bool ckks_ok = max_error < TOLERANCE;

    std::cout << "[CKKS] " << (ckks_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "Maximum error: " << max_error << "\n";

    return (ntt_ok && poly_ok && ckks_ok) ? 0 : 1;
}