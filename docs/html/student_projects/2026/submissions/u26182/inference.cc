// Serial C++17 baseline: learned classification of noisy seven-segment digits.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr int inputs = 64, classes = 10;

struct Options {
    int hidden = 256, samples = 4096, batch = 64, repeats = 100;
    int train_steps = 20000;
    uint32_t seed = 42;
    bool check = false;
};

void usage() {
    std::cout << "Usage: ./inference [options]\n"
              << "  --hidden N       hidden neurons (default 256)\n"
              << "  --samples N      held-out images (default 4096)\n"
              << "  --batch N        images per forward call (default 64)\n"
              << "  --repeats N      timed dataset passes (default 100)\n"
              << "  --train-steps N  SGD updates before timing (default 20000)\n"
              << "  --seed N         random seed (default 42)\n"
              << "  --check          check against double-precision reference\n";
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check") { o.check = true; continue; }
        if (key != "--hidden" && key != "--samples" && key != "--batch" &&
            key != "--repeats" && key != "--train-steps" && key != "--seed")
            throw std::runtime_error("unknown option: " + key);
        if (++i == argc) throw std::runtime_error("missing value for " + key);
        const std::string text = argv[i];
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("expected a nonnegative integer for " + key);
        const auto value = std::stoull(text);
        if (key == "--seed") {
            if (value > UINT32_MAX) throw std::runtime_error("seed exceeds uint32 range");
            o.seed = static_cast<uint32_t>(value);
        } else {
            // Practical bounds also keep array-size arithmetic manageable.
            if (value > 1000000 || (value == 0 && key != "--train-steps"))
                throw std::runtime_error(key + " must be 1..1000000 (train steps may be 0)");
            const int n = static_cast<int>(value);
            if (key == "--hidden") o.hidden = n;
            if (key == "--samples") o.samples = n;
            if (key == "--batch") o.batch = n;
            if (key == "--repeats") o.repeats = n;
            if (key == "--train-steps") o.train_steps = n;
        }
    }
    return o;
}

// Explicit conversion makes random inputs reproducible across C++ libraries.
float uniform(std::mt19937& rng) {
    return static_cast<float>(rng() >> 8) * (1.0f / 16777216.0f);
}

void make_image(int digit, std::mt19937& rng, float* image) {
    // Bits: top, upper-right, lower-right, bottom, lower-left, upper-left, middle.
    constexpr unsigned masks[classes] = {
        0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f
    };
    bool ink[inputs] = {};
    auto pixel = [&](int row, int col) { ink[row * 8 + col] = true; };
    for (int s = 0; s < 7; ++s) {
        if (!(masks[digit] & (1u << s))) continue;
        if (s == 0 || s == 3 || s == 6) {
            const int row = s == 0 ? 0 : (s == 3 ? 6 : 3);
            for (int col = 2; col <= 5; ++col) pixel(row, col);
        } else {
            const int col = (s == 1 || s == 2) ? 6 : 1;
            const int start = (s == 1 || s == 5) ? 1 : 4;
            for (int row = start; row < start + 2; ++row) pixel(row, col);
        }
    }
    const float brightness = 0.65f + 0.35f * uniform(rng);
    for (int i = 0; i < inputs; ++i) {
        const bool erased = uniform(rng) < 0.025f;
        const float signal = ink[i] && !erased ? brightness : 0.0f;
        image[i] = std::clamp(signal + 0.20f * uniform(rng), 0.0f, 1.0f);
    }
}

void softmax(float* logits) {
    const float largest = *std::max_element(logits, logits + classes);
    float sum = 0.0f;
    for (int k = 0; k < classes; ++k) {
        logits[k] = std::exp(logits[k] - largest);
        sum += logits[k];
    }
    for (int k = 0; k < classes; ++k) logits[k] /= sum;
}

struct Network {
    int width;
    // Row-major weights: [hidden][input] and [class][hidden].
    std::vector<float> w1, b1, w2, b2;
    Network(int h, std::mt19937& rng)
        : width(h), w1(size_t(h) * inputs), b1(h, 0.0f),
          w2(size_t(classes) * h), b2(classes, 0.0f) {
        const float scale1 = std::sqrt(6.0f / inputs);
        const float scale2 = std::sqrt(6.0f / h);
        for (float& w : w1) w = (2.0f * uniform(rng) - 1.0f) * scale1;
        for (float& w : w2) w = (2.0f * uniform(rng) - 1.0f) * scale2;
    }

    // Allocation-free serial forward pass. Buffers hold count rows.
    void forward(const float* x, int count, float* hidden, float* probabilities) const {
        for (int n = 0; n < count; ++n) {
            for (int j = 0; j < width; ++j) {
                float sum = b1[j];
                for (int i = 0; i < inputs; ++i)
                    sum += w1[size_t(j) * inputs + i] * x[size_t(n) * inputs + i];
                hidden[size_t(n) * width + j] = std::max(0.0f, sum);
            }
        }
        for (int n = 0; n < count; ++n) {
            float* p = probabilities + size_t(n) * classes;
            for (int k = 0; k < classes; ++k) {
                float sum = b2[k];
                for (int j = 0; j < width; ++j)
                    sum += w2[size_t(k) * width + j] * hidden[size_t(n) * width + j];
                p[k] = sum;
            }
            softmax(p);
        }
    }
};

void train(Network& net, int steps, std::mt19937& rng) {
    std::vector<float> x(inputs), h(net.width), p(classes), dh(net.width);
    for (int step = 0; step < steps; ++step) {
        const int label = static_cast<int>(rng() % classes);
        make_image(label, rng, x.data());
        net.forward(x.data(), 1, h.data(), p.data());
        // Cross-entropy gradient with respect to the logits.
        p[label] -= 1.0f;
        for (int j = 0; j < net.width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < classes; ++k)
                sum += net.w2[size_t(k) * net.width + j] * p[k];
            dh[j] = h[j] > 0.0f ? sum : 0.0f;
        }
        const float rate = 0.02f / (1.0f + 4.0f * step / std::max(1, steps));
        // Compute all gradients that depend on old weights before updating.
        for (int k = 0; k < classes; ++k) {
            for (int j = 0; j < net.width; ++j)
                net.w2[size_t(k) * net.width + j] -= rate * p[k] * h[j];
            net.b2[k] -= rate * p[k];
        }
        for (int j = 0; j < net.width; ++j) {
            for (int i = 0; i < inputs; ++i)
                net.w1[size_t(j) * inputs + i] -= rate * dh[j] * x[i];
            net.b1[j] -= rate * dh[j];
        }
    }
}

// Separate scalar reference uses input-major accumulation and double arithmetic.
// This checks indexing, activation and softmax, beyond checking only labels.
std::vector<double> reference(const Network& net, const float* x) {
    std::vector<double> h(net.b1.begin(), net.b1.end());
    std::vector<double> p(net.b2.begin(), net.b2.end());
    for (int i = 0; i < inputs; ++i)
        for (int j = 0; j < net.width; ++j)
            h[j] += double(x[i]) * net.w1[size_t(j) * inputs + i];
    for (double& value : h) value = std::max(0.0, value);
    for (int j = 0; j < net.width; ++j)
        for (int k = 0; k < classes; ++k)
            p[k] += h[j] * net.w2[size_t(k) * net.width + j];
    const double largest = *std::max_element(p.begin(), p.end());
    double sum = 0.0;
    for (double& value : p) { value = std::exp(value - largest); sum += value; }
    for (double& value : p) value /= sum;
    return p;
}

double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        const Options o = parse(argc, argv);
        std::mt19937 model_rng(o.seed), train_rng(o.seed ^ 0x9e3779b9u);
        std::mt19937 test_rng(o.seed ^ 0x85ebca6bu);
        Network net(o.hidden, model_rng);
        auto begin = Clock::now();
        train(net, o.train_steps, train_rng);
        const double training_seconds = seconds(begin);

        std::vector<float> x(size_t(o.samples) * inputs);
        std::vector<int> labels(o.samples);
        for (int n = 0; n < o.samples; ++n) {
            labels[n] = n % classes;
            make_image(labels[n], test_rng, x.data() + size_t(n) * inputs);
        }
        const int batch = std::min(o.batch, o.samples);
        std::vector<float> h(size_t(batch) * o.hidden);
        std::vector<float> p(size_t(o.samples) * classes);
        auto pass = [&]() {
            for (int start = 0; start < o.samples; start += batch) {
                const int count = std::min(batch, o.samples - start);
                net.forward(x.data() + size_t(start) * inputs, count, h.data(),
                            p.data() + size_t(start) * classes);
            }
        };
        pass(); // Untimed full-dataset warm-up; also prepares code and data caches.
        double total_seconds = 0.0, best_seconds = 1.0e300;
        double checksum = 0.0, loss = 0.0;
        int correct = 0;
        // Volatile publication keeps each repetition's full output observable.
        volatile double observed_checksum = 0.0;
        for (int rep = 0; rep < o.repeats; ++rep) {
            begin = Clock::now();
            pass();
            const double elapsed = seconds(begin);
            total_seconds += elapsed;
            best_seconds = std::min(best_seconds, elapsed);
            checksum = 0.0;
            loss = 0.0;
            correct = 0;
            // Validation and aggregation are outside the inference timer.
            for (int n = 0; n < o.samples; ++n) {
                const float* row = p.data() + size_t(n) * classes;
                double sum = 0.0;
                for (int k = 0; k < classes; ++k) {
                    if (!std::isfinite(row[k]) || row[k] < 0.0f || row[k] > 1.0f)
                        throw std::runtime_error("invalid output probability");
                    sum += row[k];
                    checksum += double(row[k]) * (k + 1) * (1 + n % 97);
                }
                if (std::abs(sum - 1.0) > 1.0e-5)
                    throw std::runtime_error("probabilities do not sum to one");
                correct += std::max_element(row, row + classes) - row == labels[n];
                loss -= std::log(std::max(double(row[labels[n]]), 1.0e-30));
            }
            observed_checksum = checksum;
        }
        (void)observed_checksum;
        std::cout << std::fixed << std::setprecision(6)
                  << "hidden=" << o.hidden << " samples=" << o.samples
                  << " batch=" << batch << " repeats=" << o.repeats
                  << " train_steps=" << o.train_steps << " seed=" << o.seed << '\n'
                  << "parameters=" << net.w1.size() + net.w2.size() + net.b1.size() + net.b2.size() << '\n'
                  << "training_seconds=" << training_seconds << '\n'
                  << "inference_total_seconds=" << total_seconds << '\n'
                  << "mean_pass_seconds=" << total_seconds / o.repeats << '\n'
                  << "best_pass_seconds=" << best_seconds << '\n'
                  << "images_per_second=" << double(o.samples) * o.repeats / total_seconds << '\n'
                  << "amortized_us_per_image=" << total_seconds * 1.0e6 / (double(o.samples) * o.repeats) << '\n'
                  << "accuracy=" << double(correct) / o.samples << '\n'
                  << "cross_entropy=" << loss / o.samples << '\n'
                  << "checksum=" << checksum << '\n';
        if (o.check) {
            double error = 0.0, batch_error = 0.0;
            std::vector<float> single_h(o.hidden), single_p(classes);
            const int checked = std::min(o.samples, 16);
            for (int n = 0; n < checked; ++n) {
                const float* image = x.data() + size_t(n) * inputs;
                const auto expected = reference(net, image);
                net.forward(image, 1, single_h.data(), single_p.data());
                for (int k = 0; k < classes; ++k) {
                    const double actual = p[size_t(n) * classes + k];
                    error = std::max(error, std::abs(actual - expected[k]));
                    batch_error = std::max(batch_error, std::abs(actual - single_p[k]));
                }
            }
            std::cout << std::scientific << "reference_max_abs_error=" << error
                      << " batch1_max_abs_error=" << batch_error << '\n';
            if (error > 1.0e-4 || batch_error > 1.0e-6)
                throw std::runtime_error("reference check failed");
            std::cout << "reference_check=PASS checked_images=" << checked << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
