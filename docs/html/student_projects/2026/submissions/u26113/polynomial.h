#pragma once
#include <vector>
#include <cstdint>

namespace polynomial {

// Z_q[X]/(X^N + 1) 上の多項式乗算
// 入力は長さ N の係数ベクトル
// 内部で Twist → NTT → 逐点乗算 → INTT → Untwist を実行
// ntt::set_params() による事前設定が必要
std::vector<int64_t> multiply(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

} // namespace polynomial