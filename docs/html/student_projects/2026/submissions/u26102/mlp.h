#ifndef MLP_H
#define MLP_H

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

struct MLP {
    int hidden;
    std::vector<double> w1, b1, w2;
    double b2;

    explicit MLP(int h, unsigned int seed = 1234)
        : hidden(h), w1(h), b1(h), w2(h), b2(0.0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (int j = 0; j < hidden; ++j) {
            w1[j] = dist(rng);
            b1[j] = 0.0;
            w2[j] = dist(rng);
        }
    }

    // 1次元入力 -> hidden 次元の隠れ層 -> 1次元出力
    void forward(const std::vector<double>& x,
                 std::vector<double>& out_h,
                 double& out_y) const {
        out_h.assign(hidden, 0.0);
        for (int j = 0; j < hidden; ++j) {
            const double z = b1[j] + w1[j] * x[0];
            out_h[j] = std::max(0.0, z);
        }

        out_y = b2;
        for (int j = 0; j < hidden; ++j) {
            out_y += w2[j] * out_h[j];
        }
    }

    // 勾配を加算する（重みの更新は呼び出し側で実施）
    void backward(const std::vector<double>& x,
                  const std::vector<double>& h,
                  double y_hat,
                  double y_true,
                  std::vector<double>& gradW1,
                  std::vector<double>& gradB1,
                  std::vector<double>& gradW2,
                  double& gradB2) const {
        const double dL_dy = 2.0 * (y_hat - y_true);

        gradB2 += dL_dy;
        for (int j = 0; j < hidden; ++j) {
            gradW2[j] += dL_dy * h[j];

            const double z = b1[j] + w1[j] * x[0];
            const double relu_grad = (z > 0.0) ? 1.0 : 0.0;
            const double dz = dL_dy * w2[j] * relu_grad;

            gradW1[j] += dz * x[0];
            gradB1[j] += dz;
        }
    }
};

#endif // MLP_H
