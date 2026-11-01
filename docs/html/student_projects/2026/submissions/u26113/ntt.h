#pragma once
#include <vector>
#include <cstdint>
#include <utility>

namespace ntt {

// ----- グローバルパラメータ -----
extern int64_t g_N;          // 多項式次数 (2^k)
extern int64_t g_MOD;        // 素数法 (q)
extern int64_t g_PRIM_ROOT;  // 原始根

void set_params(int64_t N, int64_t MOD, int64_t PRIM_ROOT);

// ----- 基本演算 -----
int64_t mod_mul(int64_t a, int64_t b, int64_t MOD);
int64_t mod_pow(int64_t a, uint64_t e, int64_t MOD);
std::pair<int64_t, int64_t> egcd(int64_t a, int64_t b);
bool mod_inv(int64_t a, int64_t MOD, int64_t &out);

// ----- ビット反転並べ替え -----
void bit_reverse_copy(std::vector<int64_t>& a);

// ----- NTT / INTT -----
void ntt(std::vector<int64_t>& a, bool invert = false);

} // namespace ntt