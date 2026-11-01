#include "ntt.h"
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ntt {

// ----- グローバルパラメータ -----
int64_t g_N = 0;
int64_t g_MOD = 0;
int64_t g_PRIM_ROOT = 0;

void set_params(int64_t N, int64_t MOD, int64_t PRIM_ROOT) {
    if (N <= 0 || (N & (N - 1)) != 0) throw std::invalid_argument("N must be a power of two");
    if (MOD <= 2 || (MOD - 1) % N != 0) throw std::invalid_argument("MOD must satisfy N | (MOD - 1)");
    g_N = N;
    g_MOD = MOD;
    g_PRIM_ROOT = PRIM_ROOT;
}

// ----- 基本演算 -----
int64_t mod_mul(int64_t a, int64_t b, int64_t MOD) {
    return static_cast<int64_t>((__int128)a * b % MOD);
}

int64_t mod_pow(int64_t a, uint64_t e, int64_t MOD) {
    int64_t r = 1;
    while (e) {
        if (e & 1) r = mod_mul(r, a, MOD);
        a = mod_mul(a, a, MOD);
        e >>= 1;
    }
    return r;
}

std::pair<int64_t, int64_t> egcd(int64_t a, int64_t b) {
    if (b == 0) return {a, 1};
    auto p = egcd(b, a % b);
    return {p.first, p.second - (a / b) * p.second};
}

bool mod_inv(int64_t a, int64_t MOD, int64_t& out) {
    int64_t t = 0, new_t = 1;
    int64_t r = MOD, new_r = (a % MOD + MOD) % MOD;

    while (new_r != 0) {
        int64_t q = r / new_r;

        int64_t tmp = t - q * new_t;
        t = new_t;
        new_t = tmp;

        tmp = r - q * new_r;
        r = new_r;
        new_r = tmp;
    }

    if (r != 1) return false;
    out = (t % MOD + MOD) % MOD;
    return true;
}

// ----- ビット反転並べ替え -----
void bit_reverse_copy(std::vector<int64_t>& a) {
    const int64_t N = g_N;
    for (int64_t i = 1, j = 0; i < N; ++i) {
        int64_t bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// ----- NTT / INTT -----
void ntt(std::vector<int64_t>& a, bool invert) {
    if (static_cast<int64_t>(a.size()) != g_N) throw std::invalid_argument("Polynomial size must equal N");

    bit_reverse_copy(a);

    // 原始 N 乗根
    int64_t root = mod_pow(g_PRIM_ROOT, (g_MOD - 1) / g_N, g_MOD);

    // INTTでは逆元を使用
    if (invert) {
        int64_t root_inv = 0;
        if (!mod_inv(root, g_MOD, root_inv)) throw std::runtime_error("Root has no inverse");
        root = root_inv;
    }

    // Cooley-Tukey NTT
    for (int64_t len = 2; len <= g_N; len <<= 1) {
        int64_t wlen = mod_pow(root, g_N / len, g_MOD);

        for (int64_t i = 0; i < g_N; i += len) {
            int64_t w = 1;
            for (int64_t j = 0; j < len / 2; ++j) {
                int64_t u = a[i + j];
                int64_t v = mod_mul(a[i + j + len / 2], w, g_MOD);
                a[i + j] = (u + v) % g_MOD;
                a[i + j + len / 2] = (u - v + g_MOD) % g_MOD;
                w = mod_mul(w, wlen, g_MOD);
            }
        }
    }

    // INTTでは N^{-1} を掛ける
    if (invert) {
        int64_t ninv = 0;
        if (!mod_inv(g_N, g_MOD, ninv)) throw std::runtime_error("N has no inverse");
        for (auto& x : a) x = mod_mul(x, ninv, g_MOD);
    }
}

} // namespace ntt