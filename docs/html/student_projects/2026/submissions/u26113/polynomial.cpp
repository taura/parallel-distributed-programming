#include "polynomial.h"
#include "ntt.h"
#include <stdexcept>

namespace polynomial {

std::vector<int64_t> multiply(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    if (a.size() != b.size() || a.empty()) throw std::invalid_argument("Invalid polynomial size");

    const int64_t N = a.size();
    const int64_t MOD = ntt::g_MOD;

    if (N != ntt::g_N) throw std::invalid_argument("Polynomial size must equal N");
    if ((MOD - 1) % (2 * N) != 0) throw std::invalid_argument("MOD must satisfy 2N | (MOD - 1)");

    int64_t psi = ntt::mod_pow(ntt::g_PRIM_ROOT, (MOD - 1) / (2 * N), MOD);
    int64_t psi_inv = 0;
    if (!ntt::mod_inv(psi, MOD, psi_inv)) throw std::runtime_error("psi has no inverse");

    std::vector<int64_t> A(N), B(N);
    int64_t w = 1;
    for (int64_t i = 0; i < N; ++i) {
        int64_t ai = (a[i] % MOD + MOD) % MOD;
        int64_t bi = (b[i] % MOD + MOD) % MOD;
        A[i] = ntt::mod_mul(ai, w, MOD);
        B[i] = ntt::mod_mul(bi, w, MOD);
        w = ntt::mod_mul(w, psi, MOD);
    }

    // NTT
    ntt::ntt(A, false);
    ntt::ntt(B, false);

    // 逐点乗算
    std::vector<int64_t> C(N);
    for (int64_t i = 0; i < N; ++i) C[i] = ntt::mod_mul(A[i], B[i], MOD);

    // INTT
    ntt::ntt(C, true);

    w = 1;
    for (int64_t i = 0; i < N; ++i) {
        C[i] = ntt::mod_mul(C[i], w, MOD);
        w = ntt::mod_mul(w, psi_inv, MOD);
    }

    return C;
}

} // namespace polynomial