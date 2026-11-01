#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int default_width = 3600;
constexpr int default_height = 2400;
constexpr int default_max_iter = 77000;
constexpr int smoothing_steps = 2;

// Return the first escape iteration, or 0 if no escape was observed.
// Optionally save |z|^2 after two extra iterations for smoother coloring.
int escape_iteration(double cr, double ci, int max_iter,
                     double* escaped_norm_squared = nullptr) {
    if (escaped_norm_squared) {
        *escaped_norm_squared = 0.0;
    }
    double zr = 0.0;
    double zi = 0.0;
    for (int n = 1; n <= max_iter; ++n) {
        const double next_r = zr * zr - zi * zi + cr;
        const double next_i = 2.0 * zr * zi + ci;
        zr = next_r;
        zi = next_i;
        if (zr * zr + zi * zi > 4.0) {
            if (escaped_norm_squared) {
                for (int extra = 0; extra < smoothing_steps; ++extra) {
                    const double next_r = zr * zr - zi * zi + cr;
                    const double next_i = 2.0 * zr * zi + ci;
                    zr = next_r;
                    zi = next_i;
                }
                *escaped_norm_squared = zr * zr + zi * zi;
            }
            return n;
        }
    }
    return 0;
}

// Only call for escaped pixels. The stored norm includes smoothing_steps.
double smooth_escape_value(int n, double norm_squared) {
    const double log_abs_z = 0.5 * std::log(norm_squared);
    return n + smoothing_steps + 1.0 - std::log2(log_abs_z);
}

int positive_integer(const char* text, int upper_bound) {
    const std::string input(text);
    std::size_t used = 0;
    const long long value = std::stoll(input, &used);
    if (used != input.size() || value < 1 || value > upper_bound) {
        throw std::invalid_argument("parameter out of range: " + input);
    }
    return static_cast<int>(value);
}

int self_test() {
    struct Test { double cr; double ci; int limit; int expected; };
    const Test tests[] = {
        {0.0, 0.0, 100, 0}, {-1.0, 0.0, 100, 0},
        {-2.0, 0.0, 100, 0}, {1.0, 0.0, 100, 3},
        {2.0, 0.0, 100, 2}, {0.0, 2.0, 100, 2},
        {1.0, 0.0, 2, 0}, {1.0, 0.0, 3, 3}
    };
    for (const auto& test : tests) {
        double norm_squared = 0.0;
        if (escape_iteration(test.cr, test.ci, test.limit, &norm_squared)
            != test.expected) {
            std::cerr << "Known-point test failed\n";
            return 1;
        }
        if ((test.expected == 0 && norm_squared != 0.0) ||
            (test.expected != 0 &&
             (!(norm_squared > 4.0) ||
              !std::isfinite(smooth_escape_value(test.expected, norm_squared))))) {
            std::cerr << "Smooth-value validity test failed\n";
            return 1;
        }
    }
    // c=1: first escape at z3=5, then z4=26 and z5=677.
    double known_norm = 0.0;
    escape_iteration(1.0, 0.0, 3, &known_norm);
    if (known_norm != 458329.0) {
        std::cerr << "Post-escape orbit test failed\n";
        return 1;
    }
    // With c=0, the continuation z -> z^2 makes the smooth value invariant.
    const double before = smooth_escape_value(3, 16.0);
    const double after = smooth_escape_value(4, 256.0);
    if (std::abs(before - after) > 1e-12) {
        std::cerr << "Smooth-value iteration-offset test failed\n";
        return 1;
    }
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) {
            const double cr = -2.5 + 3.5 * (x + 0.5) / 40;
            const double ci = 1.5 * (y + 0.5) / 30;
            double upper_norm = 0.0;
            double lower_norm = 0.0;
            if (escape_iteration(cr, ci, 200, &upper_norm) !=
                escape_iteration(cr, -ci, 200, &lower_norm) ||
                upper_norm != lower_norm) {
                std::cerr << "Conjugate-symmetry test failed\n";
                return 1;
            }
        }
    }
    std::cout << "PASS: 8 known-point/iteration-limit tests and "
                 "1200 conjugate-symmetry tests; smooth-coloring tests\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            return self_test();
        }
        if ((argc == 2 && std::string(argv[1]) == "--help") || argc > 5) {
            std::cout << "Usage: " << argv[0]
                      << " [width=" << default_width << "] [height="
                      << default_height << "] [max_iter=" << default_max_iter
                      << "] [output=mandelbrot.ppm]\n"
                         "       --self-test\n"
                         "Dimensions: 1..16384; max_iter: 1..1000000\n"
                         "Defaults target about 5 minutes on Miyabi (estimate).\n";
            return argc > 5 ? 1 : 0;
        }
        const int width = argc > 1 ? positive_integer(argv[1], 16384) : default_width;
        const int height = argc > 2 ? positive_integer(argv[2], 16384) : default_height;
        const int max_iter = argc > 3 ? positive_integer(argv[3], 1000000) : default_max_iter;
        const std::string output = argc > 4 ? argv[4] : "mandelbrot.ppm";
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        std::vector<int> escape(pixels);
        std::vector<double> escaped_norm_squared(pixels);

        // Pixel centers, with square pixels and symmetry about the real axis.
        const double step = 3.5 / width;
        const auto start = std::chrono::steady_clock::now();
        for (int y = 0; y < height; ++y) {
            const double ci = ((height - 1) * 0.5 - y) * step;
            for (int x = 0; x < width; ++x) {
                const double cr = -2.5 + (x + 0.5) * step;
                const std::size_t p = static_cast<std::size_t>(y) * width + x;
                escape[p] = escape_iteration(cr, ci, max_iter,
                                             &escaped_norm_squared[p]);
            }
        }
        const auto finish = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(finish - start).count();

        // Statistics and image encoding are outside the main computation timer.
        std::uint64_t total_iterations = 0;
        std::size_t nonescaped = 0;
        std::size_t symmetry_mismatches = 0;
        std::size_t smooth_symmetry_mismatches = 0;
        for (std::size_t p = 0; p < pixels; ++p) {
            nonescaped += (escape[p] == 0);
            total_iterations += escape[p] == 0 ? max_iter : escape[p];
        }
        for (int y = 0; y < height / 2; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t upper = static_cast<std::size_t>(y) * width + x;
                const std::size_t lower = static_cast<std::size_t>(height - 1 - y) * width + x;
                symmetry_mismatches += escape[upper] != escape[lower];
                smooth_symmetry_mismatches +=
                    escaped_norm_squared[upper] != escaped_norm_squared[lower];
            }
        }

        std::ofstream image(output, std::ios::binary);
        if (!image) {
            throw std::runtime_error("cannot open output: " + output);
        }
        image << "P6\n" << width << ' ' << height << "\n255\n";
        for (std::size_t p = 0; p < pixels; ++p) {
            const int n = escape[p];
            unsigned char rgb[3] = {0, 0, 0};
            if (n != 0) {
                // Continuous escape values remove integer-iteration color bands.
                const double mu = smooth_escape_value(n, escaped_norm_squared[p]);
                const double t = std::log1p(mu > 0.0 ? mu : 0.0);
                for (int channel = 0; channel < 3; ++channel) {
                    rgb[channel] = static_cast<unsigned char>(
                        127.5 * (1.0 + std::cos(3.0 * t + 2.0 * channel)));
                }
            }
            image.write(reinterpret_cast<const char*>(rgb), 3);
        }
        image.close();
        if (!image) {
            throw std::runtime_error("failed to write output: " + output);
        }
        const double postprocess_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - finish).count();
        std::cout << "width=" << width << " height=" << height
                  << " max_iter=" << max_iter << '\n'
                  << std::fixed << std::setprecision(6)
                  << "compute_seconds=" << seconds << '\n'
                  << "postprocess_seconds=" << postprocess_seconds << '\n'
                  << "total_iterations=" << total_iterations << '\n'
                  << "coloring_extra_iterations="
                  << static_cast<std::uint64_t>(pixels - nonescaped) * smoothing_steps << '\n'
                  << "nonescaped_pixels=" << nonescaped << '/' << pixels << '\n'
                  << "symmetry_mismatches=" << symmetry_mismatches << '\n'
                  << "smooth_symmetry_mismatches=" << smooth_symmetry_mismatches << '\n'
                  << "coloring=smooth\n"
                  << "image=" << output << '\n';
        return symmetry_mismatches == 0 && smooth_symmetry_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
