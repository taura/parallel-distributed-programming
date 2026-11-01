
#include "wavelet_matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t parse_size(const char* text) {
    const std::string input(text);

    if (input.empty() ||
        input.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected a positive integer");
    }

    const auto value = std::stoull(input);
    const auto result = static_cast<std::size_t>(value);

    if (result == 0 || result != value) {
        throw std::out_of_range("Argument is zero or too large");
    }

    return result;
}

struct Timing {
    double build = 0;
    double access = 0;
    double rank = 0;
    double select = 0;
};

struct Statistics {
    double mean = 0;
    double minimum = 0;
    double stddev = 0;
};

Statistics summarize(const std::vector<double>& samples) {
    Statistics result;

    for (double x : samples) {
        result.mean += x;
    }
    result.mean /= samples.size();

    result.minimum = *std::min_element(
        samples.begin(), samples.end());

    double variance = 0;
    for (double x : samples) {
        const double diff = x - result.mean;
        variance += diff * diff;
    }

    variance /= samples.size();
    result.stddev = std::sqrt(variance);

    return result;
}

void report(const std::string& name,
            const std::vector<double>& samples) {
    const auto s = summarize(samples);

    std::cout << name
              << ": mean=" << s.mean
              << " ms, min=" << s.minimum
              << " ms, stddev=" << s.stddev
              << " ms\n";
}

unsigned actual_bit_width(const std::vector<std::uint32_t>& data) {
    const auto max_value = *std::max_element(
        data.begin(), data.end());

    unsigned width = 1;
    auto value = max_value;

    while (value >>= 1) {
        ++width;
    }

    return width;
}

int main(int argc, char** argv) {
    try {
        // Parse positional arguments and optional --build-only.
        std::vector<std::string> args;
        bool build_only = false;

        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--build-only") {
                build_only = true;
            } else {
                args.emplace_back(argv[i]);
            }
        }

        if (args.size() > 6) {
            std::cerr
                << "Usage: " << argv[0]
                << " [N] [repeats] [bit_width]"
                << " [access_count] [rank_count] [select_count]"
                << " [--build-only]\n";
            return 1;
        }

        auto argument = [&](std::size_t index, std::size_t fallback) {
            return index < args.size()
                ? parse_size(args[index].c_str())
                : fallback;
        };

        const std::size_t n = argument(0, 100'000'000);
        const std::size_t repeats = argument(1, 3);
        const std::size_t bit_width = argument(2, 32);

        if (bit_width > 32) {
            throw std::out_of_range("bit_width must be 1-32");
        }

        const std::size_t access_count =
            argument(3, 100'000);
        const std::size_t rank_count =
            argument(4, 100'000);
        const std::size_t select_count =
            argument(5, 100'000);

        const std::uint32_t max_value =
            bit_width == 32
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(
                      (std::uint64_t{1} << bit_width) - 1);

        std::cout << "=== Configuration ===\n"
                  << "N: " << n << '\n'
                  << "Repeats: " << repeats << '\n'
                  << "Bit width: " << bit_width << '\n'
                  << "Mode: "
                  << (build_only ? "build-only" : "full")
                  << "\nGenerating input...\n";

        // Generate input once, outside timed regions.
        std::mt19937_64 rng(42);

        std::uniform_int_distribution<std::uint32_t>
            value_dist(0, max_value);

        std::vector<std::uint32_t> data(n);

        for (auto& value : data) {
            value = value_dist(rng);
        }

        const unsigned height = actual_bit_width(data);

        std::uniform_int_distribution<std::size_t>
            position_dist(0, n - 1);

        std::uniform_int_distribution<std::size_t>
            end_dist(0, n);

        // Query pools are only allocated when needed.
        std::vector<std::size_t> access_positions;
        std::vector<std::uint32_t> rank_values;
        std::vector<std::size_t> rank_ends;
        std::vector<std::size_t> select_positions;
        std::vector<std::uint32_t> select_values;
        std::vector<std::size_t> select_k;

        if (!build_only) {
            access_positions.resize(access_count);
            rank_values.resize(rank_count);
            rank_ends.resize(rank_count);
            select_positions.resize(select_count);
            select_values.resize(select_count);
            select_k.resize(select_count);

            for (auto& pos : access_positions) {
                pos = position_dist(rng);
            }

            for (std::size_t q = 0; q < rank_count; ++q) {
                // Alternate present and uniform random values.
                rank_values[q] = (q % 2 == 0)
                    ? data[position_dist(rng)]
                    : value_dist(rng);

                rank_ends[q] = end_dist(rng);
            }

            for (std::size_t q = 0; q < select_count; ++q) {
                select_positions[q] = position_dist(rng);
                select_values[q] = data[select_positions[q]];
            }
        }

        // Benchmark measurements.
        std::vector<double> build_times;
        std::vector<double> access_times;
        std::vector<double> rank_times;
        std::vector<double> select_times;

        build_times.reserve(repeats);

        if (!build_only) {
            access_times.reserve(repeats);
            rank_times.reserve(repeats);
            select_times.reserve(repeats);
        }

        bool access_ok = true;
        bool rank_ok = true;
        bool select_ok = true;

        for (std::size_t run = 0; run < repeats; ++run) {
            Timing timing;

            // Construction timing: constructor only.
            auto start = Clock::now();
            WaveletMatrix matrix(data);
            auto end = Clock::now();

            timing.build = elapsed_ms(start, end);
            build_times.push_back(timing.build);

            // Lightweight construction verification.
            constexpr std::size_t verify_samples = 1000;
            const std::size_t checks = std::min(n, verify_samples);

            for (std::size_t q = 0; q < checks; ++q) {
                const std::size_t pos =
                    (checks == n) ? q : position_dist(rng);

                if (matrix.access(pos) != data[pos]) {
                    access_ok = false;
                }
            }

            if (!build_only) {
                // Prepare valid select arguments.
                for (std::size_t q = 0; q < select_count; ++q) {
                    select_k[q] = matrix.rank(
                        select_values[q],
                        select_positions[q] + 1);
                }

                std::uint64_t access_checksum = 0;
                std::uint64_t rank_checksum = 0;
                std::uint64_t select_checksum = 0;

                // ACCESS timing.
                start = Clock::now();

                for (std::size_t pos : access_positions) {
                    access_checksum += matrix.access(pos);
                }

                end = Clock::now();
                timing.access = elapsed_ms(start, end);
                access_times.push_back(timing.access);

                // RANK timing.
                start = Clock::now();

                for (std::size_t q = 0; q < rank_count; ++q) {
                    rank_checksum += matrix.rank(
                        rank_values[q], rank_ends[q]);
                }

                end = Clock::now();
                timing.rank = elapsed_ms(start, end);
                rank_times.push_back(timing.rank);

                // SELECT timing (verification excluded).
                std::vector<std::size_t> select_results(select_count);
                std::size_t missing = 0;

                start = Clock::now();

                for (std::size_t q = 0; q < select_count; ++q) {
                    const auto pos = matrix.select(
                        select_values[q], select_k[q]);

                    if (pos) {
                        select_checksum += *pos;
                        select_results[q] = *pos;
                    } else {
                        ++missing;
                    }
                }

                end = Clock::now();
                timing.select = elapsed_ms(start, end);
                select_times.push_back(timing.select);

                if (missing != 0) {
                    select_ok = false;
                }

                // Verify access outside timing.
                for (std::size_t pos : access_positions) {
                    if (matrix.access(pos) != data[pos]) {
                        access_ok = false;
                    }
                }

                // Verify rank using a local consistency property.
                for (std::size_t q = 0; q < rank_count; ++q) {
                    const auto value = rank_values[q];
                    const auto right = rank_ends[q];

                    if (right == 0) {
                        if (matrix.rank(value, 0) != 0) {
                            rank_ok = false;
                        }
                        continue;
                    }

                    const auto a = matrix.rank(value, right);
                    const auto b = matrix.rank(value, right - 1);
                    const std::size_t expected =
                        data[right - 1] == value ? 1 : 0;

                    if (a < b || a - b != expected) {
                        rank_ok = false;
                    }
                }

                // Stronger reference verification on small inputs.
                if (n <= 10'000) {
                    const std::size_t samples =
                        std::min<std::size_t>(rank_count, 100);

                    for (std::size_t q = 0; q < samples; ++q) {
                        std::size_t expected = 0;

                        for (std::size_t i = 0;
                             i < rank_ends[q]; ++i) {
                            if (data[i] == rank_values[q]) {
                                ++expected;
                            }
                        }

                        if (matrix.rank(rank_values[q],
                                        rank_ends[q]) != expected) {
                            rank_ok = false;
                        }
                    }
                }

                // Verify select outside timing.
                for (std::size_t q = 0; q < select_count; ++q) {
                    if (select_results[q] != select_positions[q]) {
                        select_ok = false;
                    }
                }

                std::cout << "Run " << run + 1 << '/' << repeats
                          << ": build=" << timing.build
                          << " ms, access=" << timing.access
                          << " ms, rank=" << timing.rank
                          << " ms, select=" << timing.select
                          << " ms\n"
                          << "  Checksums: "
                          << access_checksum << ' '
                          << rank_checksum << ' '
                          << select_checksum << '\n';
            } else {
                std::cout << "Run " << run + 1 << '/' << repeats
                          << ": build=" << timing.build << " ms\n";
            }

            if (!access_ok || !rank_ok || !select_ok) {
                std::cerr << "Verification failed\n";
                return 1;
            }
        }

        // Report construction statistics.
        const Statistics build_stats = summarize(build_times);
        const double build_seconds = build_stats.mean / 1000.0;

        const double throughput =
            static_cast<double>(n) / build_seconds / 1e6;

        const double level_throughput =
            throughput * height;

        std::cout << "\n=== Construction Performance ===\n";
        report("Build", build_times);

        std::cout << "Actual height: " << height << '\n'
                  << "Throughput: " << throughput
                  << " M elements/s\n"
                  << "Level throughput: " << level_throughput
                  << " M element-levels/s\n";

        if (!build_only) {
            std::cout << "\n=== Query Performance ===\n";
            report("Access", access_times);
            report("Rank", rank_times);
            report("Select", select_times);

            std::cout << "Average query latency:\n"
                      << "  Access: "
                      << summarize(access_times).mean * 1000.0
                             / access_count
                      << " us/query\n"
                      << "  Rank: "
                      << summarize(rank_times).mean * 1000.0
                             / rank_count
                      << " us/query\n"
                      << "  Select: "
                      << summarize(select_times).mean * 1000.0
                             / select_count
                      << " us/query\n";
        }

        std::cout << "\n=== Verification ===\n"
                  << "Access: " << (access_ok ? "PASS" : "FAIL") << '\n'
                  << "Rank: "
                  << (build_only ? "SKIPPED" :
                      (rank_ok ? "PASS" : "FAIL")) << '\n'
                  << "Select: "
                  << (build_only ? "SKIPPED" :
                      (select_ok ? "PASS" : "FAIL")) << '\n';

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
