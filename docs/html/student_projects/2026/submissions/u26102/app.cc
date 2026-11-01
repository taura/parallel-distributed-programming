#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include "mlp.h"

namespace {

void set_pixel(std::vector<unsigned char>& img, int w, int h,
               int x, int y,
               unsigned char r, unsigned char g, unsigned char b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    const int idx = 3 * (y * w + x);
    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

void draw_line(std::vector<unsigned char>& img, int w, int h,
               int x0, int y0, int x1, int y1,
               unsigned char r, unsigned char g, unsigned char b) {
    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    while (true) {
        set_pixel(img, w, h, x0, y0, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    int n = 20000;
    int epochs = 20;
    int batch = 256;
    int hidden = 32;
    double lr = 0.01;
    double noise_std = 0.05;
    unsigned int seed = 1234;
    std::string out_ppm = "regression.ppm";

    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) epochs = std::atoi(argv[2]);
    if (argc > 3) batch = std::atoi(argv[3]);
    if (argc > 4) hidden = std::atoi(argv[4]);
    if (argc > 5) lr = std::atof(argv[5]);
    if (argc > 6) noise_std = std::atof(argv[6]);
    if (argc > 7) seed = static_cast<unsigned int>(std::strtoul(argv[7], nullptr, 10));
    if (argc > 8) out_ppm = argv[8];

    if (n <= 1 || epochs <= 0 || batch <= 0 || hidden <= 0 || lr <= 0.0) {
        std::cerr << "Usage: ./app [N] [epochs] [batch] [hidden] [lr] [noise_std] [seed] [out_ppm]\n";
        return 1;
    }

    const double pi = 3.14159265358979323846;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_x(0.0, 1.0);
    std::normal_distribution<double> dist_noise(0.0, noise_std);

    std::vector<double> x_all(n), y_all(n);
    for (int i = 0; i < n; ++i) {
        const double x = dist_x(rng);
        const double y = std::sin(2.0 * pi * x) + dist_noise(rng);
        x_all[i] = x;
        y_all[i] = y;
    }

    const int train_n = (n * 8) / 10;
    const int test_n = n - train_n;

    std::vector<double> trainX(train_n), trainY(train_n), testX(test_n), testY(test_n);
    for (int i = 0; i < train_n; ++i) {
        trainX[i] = x_all[i];
        trainY[i] = y_all[i];
    }
    for (int i = 0; i < test_n; ++i) {
        testX[i] = x_all[train_n + i];
        testY[i] = y_all[train_n + i];
    }

    MLP net(hidden, seed + 1);
    std::vector<double> gradW1(hidden), gradB1(hidden), gradW2(hidden);
    double gradB2 = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int e = 0; e < epochs; ++e) {
        std::vector<int> idx(trainX.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(),
                     std::mt19937(seed + static_cast<unsigned int>(e) + 10));

        for (size_t p = 0; p < idx.size(); p += static_cast<size_t>(batch)) {
            std::fill(gradW1.begin(), gradW1.end(), 0.0);
            std::fill(gradB1.begin(), gradB1.end(), 0.0);
            std::fill(gradW2.begin(), gradW2.end(), 0.0);
            gradB2 = 0.0;

            const size_t lim = std::min(static_cast<size_t>(batch), idx.size() - p);

            for (size_t i = 0; i < lim; ++i) {
                const int id = idx[p + i];
                std::vector<double> x(1, trainX[id]);
                std::vector<double> h;
                double y_hat;
                net.forward(x, h, y_hat);

                net.backward(x, h, y_hat, trainY[id], gradW1, gradB1, gradW2, gradB2);
            }

            const double scale = 1.0 / static_cast<double>(lim);
            for (int j = 0; j < hidden; ++j) {
                net.w1[j] -= lr * gradW1[j] * scale;
                net.b1[j] -= lr * gradB1[j] * scale;
                net.w2[j] -= lr * gradW2[j] * scale;
            }
            net.b2 -= lr * gradB2 * scale;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double train_time = std::chrono::duration<double>(t1 - t0).count();

    double mse = 0.0;
    for (size_t i = 0; i < testX.size(); ++i) {
        std::vector<double> h;
        double y_pred;
        std::vector<double> x(1, testX[i]);
        net.forward(x, h, y_pred);
        const double err = y_pred - testY[i];
        mse += err * err;
    }
    mse /= testX.size();

    std::cout << "N=" << n
              << " epochs=" << epochs
              << " batch=" << batch
              << " hidden=" << hidden
              << " lr=" << lr
              << " noise_std=" << noise_std
              << " seed=" << seed << "\n";
    std::cout << "Test MSE = " << mse << "\n";
    std::cout << "Train time = " << train_time << " s\n";

    constexpr int width = 1024;
    constexpr int height = 512;
    constexpr double y_min = -1.5;
    constexpr double y_max = 1.5;

    auto x_to_px = [](double x) {
        return static_cast<int>(x * (width - 1));
    };
    auto y_to_py = [](double y) {
        const double t = (y - y_min) / (y_max - y_min);
        return static_cast<int>((1.0 - t) * (height - 1));
    };
    auto predict = [&net](double x) {
        std::vector<double> h;
        std::vector<double> in(1, x);
        double y;
        net.forward(in, h, y);
        return y;
    };

    std::vector<unsigned char> img(3 * width * height, 255);

    const int y0 = y_to_py(0.0);
    if (0 <= y0 && y0 < height) {
        for (int x = 0; x < width; ++x) set_pixel(img, width, height, x, y0, 220, 220, 220);
    }

    for (int i = 0; i < test_n; ++i) {
        const int px = x_to_px(testX[i]);
        const int py = y_to_py(testY[i]);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                set_pixel(img, width, height, px + dx, py + dy, 60, 60, 60);
            }
        }
    }

    for (int x = 0; x < width - 1; ++x) {
        const double x0n = static_cast<double>(x) / static_cast<double>(width - 1);
        const double x1n = static_cast<double>(x + 1) / static_cast<double>(width - 1);

        const double y0_true = std::sin(2.0 * pi * x0n);
        const double y1_true = std::sin(2.0 * pi * x1n);
        draw_line(img, width, height,
                  x, y_to_py(y0_true),
                  x + 1, y_to_py(y1_true),
                  70, 110, 240);

        const double y0_pred = predict(x0n);
        const double y1_pred = predict(x1n);
        draw_line(img, width, height,
                  x, y_to_py(y0_pred),
                  x + 1, y_to_py(y1_pred),
                  230, 60, 60);
    }

    std::ofstream ofs(out_ppm, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    ofs.close();

    std::cout << "Wrote image: " << out_ppm << "\n";
    return 0;
}
