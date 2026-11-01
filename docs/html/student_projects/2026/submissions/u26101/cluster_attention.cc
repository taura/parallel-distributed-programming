// Serial clustering-routed causal attention with blockwise online softmax.
// Build: clang++ -O3 -Wall -Wextra -std=c++17 cluster_attention.cc -o cluster_attention
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Options {
    int tokens = 4096, dim = 32, clusters = 8, iterations = 12;
    int tile = 64, passes = 1, compare_dense = 1, batch = 256;
    std::uint32_t seed = 42;
    double noise = 0.9, skew = 1.3;
};

int argument(char* text) {
    std::size_t end = 0;
    int number = std::stoi(text, &end);
    if (text[end] != '\0') throw std::invalid_argument("not an integer");
    return number;
}

double real_argument(char* text) {
    std::size_t end = 0;
    double number = std::stod(text, &end);
    if (text[end] != '\0' || !std::isfinite(number))
        throw std::invalid_argument("not a finite number");
    return number;
}

// Unbalanced, persistent topics with overlapping, noisy feature distributions.
// The payload channel makes attention's numerical output independently testable.
void make_input(const Options& o, std::uint32_t seed, std::vector<float>& x,
                std::vector<float>& values, std::vector<int>& truth) {
    std::mt19937 rng(seed);
    std::vector<double> frequencies(o.clusters);
    for (int t = 0; t < o.clusters; ++t)
        frequencies[t] = 1.0 / std::pow(double(t + 1), o.skew);
    std::discrete_distribution<int> new_topic(frequencies.begin(), frequencies.end());
    std::bernoulli_distribution stay(0.6);
    std::normal_distribution<float> noise(0.0f, float(o.noise));
    std::uniform_real_distribution<float> payload(-1.0f, 1.0f);
    const int vd = o.clusters + 1;
    int topic = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (i == 0 || !stay(rng)) topic = new_topic(rng);
        truth[i] = topic;
        for (int d = 0; d < o.dim; ++d)
            x[i * o.dim + d] = noise(rng) + (d == topic ? 1.5f : 0.0f) +
                               (d == (topic + 1) % o.clusters ? 0.8f : 0.0f);
        values[i * vd + topic] = 1.0f;
        values[i * vd + o.clusters] = payload(rng);
    }
}

// Lloyd's k-means; centers are flat arrays, and empty clusters retain their center.
std::vector<int> kmeans(const std::vector<float>& x, int n, const Options& o,
                        std::vector<float>& centers) {
    const int d = o.dim, c = o.clusters;
    // Deterministic farthest-first initialization; does not use topic labels.
    std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
    int selected = 0;
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < d; ++j) centers[k * d + j] = x[selected * d + j];
        for (int i = 0; i < n; ++i) {
            double dist = 0;
            for (int j = 0; j < d; ++j) {
                double diff = double(x[i * d + j]) - centers[k * d + j];
                dist += diff * diff;
            }
            nearest[i] = std::min(nearest[i], dist);
        }
        selected = static_cast<int>(std::max_element(nearest.begin(), nearest.end()) - nearest.begin());
    }

    std::vector<int> labels(n, -1), counts(c);
    std::vector<float> sums(static_cast<std::size_t>(c) * d);
    auto assign = [&]() {
        int changes = 0;
        for (int i = 0; i < n; ++i) {
            double best = std::numeric_limits<double>::infinity();
            int chosen = 0;
            for (int k = 0; k < c; ++k) {
                double dist = 0;
                for (int j = 0; j < d; ++j) {
                    double diff = double(x[i * d + j]) - centers[k * d + j];
                    dist += diff * diff;
                }
                if (dist < best) { best = dist; chosen = k; }
            }
            changes += labels[i] != chosen;
            labels[i] = chosen;
        }
        return changes;
    };

    for (int step = 0; step < o.iterations; ++step) {
        int changes = assign();
        if (changes == 0) break;
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n; ++i) {
            int k = labels[i];
            ++counts[k];
            for (int j = 0; j < d; ++j) sums[k * d + j] += x[i * d + j];
        }
        for (int k = 0; k < c; ++k)
            if (counts[k] != 0)
                for (int j = 0; j < d; ++j)
                    centers[k * d + j] = sums[k * d + j] / counts[k];
    }
    assign(); // Match labels to the last updated centers.
    return labels;
}

int nearest_center(const std::vector<float>& x, int i, const Options& o,
                   const std::vector<float>& centers) {
    double best = std::numeric_limits<double>::infinity();
    int chosen = 0;
    for (int k = 0; k < o.clusters; ++k) {
        double dist = 0;
        for (int j = 0; j < o.dim; ++j) {
            double diff = double(x[i * o.dim + j]) - centers[k * o.dim + j];
            dist += diff * diff;
        }
        if (dist < best) { best = dist; chosen = k; }
    }
    return chosen;
}

struct Routing {
    std::vector<std::vector<int>> members;
    std::vector<int> rank;
    std::uint64_t pairs = 0;
};

void route_batch(const std::vector<float>& x, int begin, int end, const Options& o,
                 const std::vector<float>& centers, std::vector<int>& labels, Routing& r) {
    // Centers are frozen throughout this batch. Rank limits each query to past keys.
    for (int i = begin; i < end; ++i) {
        int cluster = nearest_center(x, i, o, centers);
        labels[i] = cluster;
        r.rank[i] = static_cast<int>(r.members[cluster].size());
        r.members[cluster].push_back(i);
        r.pairs += static_cast<std::uint64_t>(r.rank[i]);
    }
}

void update_centers(const std::vector<float>& x, const std::vector<int>& labels,
                    int begin, int end, const Options& o,
                    std::vector<float>& centers, std::vector<int>& seen) {
    std::vector<double> sums(static_cast<std::size_t>(o.clusters) * o.dim, 0.0);
    std::vector<int> counts(o.clusters, 0);
    for (int i = begin; i < end; ++i) {
        int k = labels[i];
        ++counts[k];
        for (int j = 0; j < o.dim; ++j)
            sums[k * o.dim + j] += x[i * o.dim + j];
    }
    for (int k = 0; k < o.clusters; ++k) {
        if (counts[k] == 0) continue;
        for (int j = 0; j < o.dim; ++j)
            centers[k * o.dim + j] = float((seen[k] * double(centers[k * o.dim + j]) +
                                           sums[k * o.dim + j]) / (seen[k] + counts[k]));
        seen[k] += counts[k];
    }
}

// The same online-softmax computation for both masks: cluster-routed causal
// attention or dense causal attention. Neither materializes an N x N matrix.
void attention(const std::vector<float>& x, const std::vector<float>& values,
               const std::vector<int>& labels, const Routing& r,
               const Options& o, int begin, int end, bool dense, std::vector<float>& out) {
    const int d = o.dim, vd = o.clusters + 1;
    const double scale = 1.0 / std::sqrt(double(d));
    std::vector<double> scores(o.tile), numerator(vd), tile_num(vd);
    for (int i = begin; i < end; ++i) {
        const auto& keys = r.members[labels[i]];
        const int eligible = dense ? i : r.rank[i]; // strictly earlier keys
        if (eligible == 0) {
            for (int j = 0; j < vd; ++j) out[static_cast<std::size_t>(i) * vd + j] = 0;
            continue;
        }
        std::fill(numerator.begin(), numerator.end(), 0.0);
        double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
        for (int start = 0; start < eligible; start += o.tile) {
            int count = std::min(o.tile, eligible - start);
            double tile_max = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < count; ++t) {
                int key = dense ? start + t : keys[start + t];
                double dot = 0;
                for (int j = 0; j < d; ++j)
                    dot += double(x[i * d + j]) * x[key * d + j];
                scores[t] = dot * scale;
                tile_max = std::max(tile_max, scores[t]);
            }
            std::fill(tile_num.begin(), tile_num.end(), 0.0);
            double tile_sum = 0;
            for (int t = 0; t < count; ++t) {
                double weight = std::exp(scores[t] - tile_max);
                tile_sum += weight;
                int key = dense ? start + t : keys[start + t];
                for (int j = 0; j < vd; ++j)
                    tile_num[j] += weight * values[key * vd + j];
            }
            double next_max = std::max(maximum, tile_max);
            double old_factor = denominator == 0 ? 0 : std::exp(maximum - next_max);
            double tile_factor = std::exp(tile_max - next_max);
            denominator = denominator * old_factor + tile_sum * tile_factor;
            for (int j = 0; j < vd; ++j)
                numerator[j] = numerator[j] * old_factor + tile_num[j] * tile_factor;
            maximum = next_max;
        }
        for (int j = 0; j < vd; ++j)
            out[static_cast<std::size_t>(i) * vd + j] = float(numerator[j] / denominator);
    }
}

// Independent, two-pass softmax reference for a few queries (verification only).
double reference_error(const std::vector<float>& x, const std::vector<float>& values,
                       const std::vector<int>& labels, const Routing& r,
                       const Options& o, bool dense, const std::vector<float>& out) {
    const int vd = o.clusters + 1;
    const double scale = 1.0 / std::sqrt(double(o.dim));
    double error = 0;
    for (int i : {0, o.tokens / 4, o.tokens / 2, o.tokens - 1}) {
        const auto& keys = r.members[labels[i]];
        int eligible = dense ? i : r.rank[i];
        if (eligible == 0) {
            for (int j = 0; j < vd; ++j)
                error = std::max(error, std::abs(double(out[i * vd + j])));
            continue;
        }
        std::vector<double> scores(eligible), expected(vd, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < eligible; ++t) {
            int key = dense ? t : keys[t];
            double dot = 0;
            for (int j = 0; j < o.dim; ++j)
                dot += double(x[i * o.dim + j]) * x[key * o.dim + j];
            scores[t] = dot * scale;
            maximum = std::max(maximum, scores[t]);
        }
        double sum = 0;
        for (int t = 0; t < eligible; ++t) {
            double weight = std::exp(scores[t] - maximum);
            sum += weight;
            int key = dense ? t : keys[t];
            for (int j = 0; j < vd; ++j)
                expected[j] += weight * values[key * vd + j];
        }
        for (int j = 0; j < vd; ++j)
            error = std::max(error, std::abs(expected[j] / sum - out[i * vd + j]));
    }
    return error;
}

struct Accuracy {
    double fraction;
    int evaluated;
};

Accuracy topic_accuracy(const std::vector<float>& output,
                        const std::vector<int>& truth, const Routing& r,
                        int clusters, bool dense) {
    int correct = 0, evaluated = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (dense ? i == 0 : r.rank[i] == 0) continue;
        ++evaluated;
        int predicted = 0;
        for (int k = 1; k < clusters; ++k)
            if (output[i * (clusters + 1) + k] > output[i * (clusters + 1) + predicted])
                predicted = k;
        correct += predicted == truth[i];
    }
    return {evaluated == 0 ? 0.0 : double(correct) / evaluated, evaluated};
}

int main(int argc, char** argv) {
    try {
        if (argc > 12) throw std::invalid_argument("too many arguments");
        Options o;
        if (argc > 1) o.tokens = argument(argv[1]);
        if (argc > 2) o.dim = argument(argv[2]);
        if (argc > 3) o.clusters = argument(argv[3]);
        if (argc > 4) o.iterations = argument(argv[4]);
        if (argc > 5) o.tile = argument(argv[5]);
        if (argc > 6) o.passes = argument(argv[6]);
        if (argc > 7) {
            int seed = argument(argv[7]);
            if (seed < 0) throw std::invalid_argument("seed must be nonnegative");
            o.seed = static_cast<std::uint32_t>(seed);
        }
        if (argc > 8) o.compare_dense = argument(argv[8]);
        if (argc > 9) o.batch = argument(argv[9]);
        if (argc > 10) o.noise = real_argument(argv[10]);
        if (argc > 11) o.skew = real_argument(argv[11]);
        if (o.tokens < 2 || o.dim < 1 || o.clusters < 2 || o.clusters > o.dim ||
            o.clusters > o.tokens || o.iterations < 1 || o.tile < 1 || o.passes < 1 || o.batch < 1 ||
            o.noise <= 0 || o.skew < 0 || (o.compare_dense != 0 && o.compare_dense != 1))
            throw std::invalid_argument("require tokens>=clusters>=2, dim>=clusters, positive iterations/tile/passes/batch/noise, skew>=0, compare_dense=0 or 1");

        std::vector<float> x(static_cast<std::size_t>(o.tokens) * o.dim);
        std::vector<float> values(static_cast<std::size_t>(o.tokens) * (o.clusters + 1));
        std::vector<int> truth(o.tokens);
        make_input(o, o.seed, x, values, truth);

        // An independent calibration sequence initializes k-means; this never
        // reads a future token from the evaluation sequence.
        int calibration_n = std::max(64, o.clusters * 32);
        std::vector<float> calibration_x(static_cast<std::size_t>(calibration_n) * o.dim);
        std::vector<float> calibration_values(static_cast<std::size_t>(calibration_n) * (o.clusters + 1));
        std::vector<int> calibration_truth(calibration_n);
        make_input(o, o.seed ^ 0x9e3779b9U, calibration_x, calibration_values, calibration_truth);

        std::vector<float> centers(static_cast<std::size_t>(o.clusters) * o.dim);
        auto calibration_start = Clock::now();
        auto calibration_labels = kmeans(calibration_x, calibration_n, o, centers);
        auto calibration_done = Clock::now();
        std::vector<int> seen(o.clusters, 0);
        for (int label : calibration_labels) ++seen[label];

        std::vector<int> labels(o.tokens, -1);
        Routing routing{std::vector<std::vector<int>>(o.clusters), std::vector<int>(o.tokens), 0};

        std::vector<float> output(values.size());
        std::vector<float> dense_output;
        if (o.compare_dense) dense_output.resize(values.size());
        double kmeans_sec = std::chrono::duration<double>(calibration_done - calibration_start).count();
        double routing_sec = 0, sparse_sec = 0, dense_sec = 0;
        for (int begin = 0; begin < o.tokens; begin += o.batch) {
            int end = std::min(o.tokens, begin + o.batch);
            auto t0 = Clock::now();
            route_batch(x, begin, end, o, centers, labels, routing);
            auto t1 = Clock::now();
            for (int pass = 0; pass < o.passes; ++pass)
                attention(x, values, labels, routing, o, begin, end, false, output);
            auto t2 = Clock::now();
            if (o.compare_dense)
                for (int pass = 0; pass < o.passes; ++pass)
                    attention(x, values, labels, routing, o, begin, end, true, dense_output);
            auto t3 = Clock::now();
            // Only the next batch sees updated centers. Old key assignments stay fixed.
            update_centers(x, labels, begin, end, o, centers, seen);
            auto t4 = Clock::now();
            routing_sec += std::chrono::duration<double>(t1 - t0).count();
            sparse_sec += std::chrono::duration<double>(t2 - t1).count();
            dense_sec += std::chrono::duration<double>(t3 - t2).count();
            kmeans_sec += std::chrono::duration<double>(t4 - t3).count();
        }

        int pure = 0;
        std::vector<int> counts(static_cast<std::size_t>(o.clusters) * o.clusters);
        for (int i = 0; i < o.tokens; ++i)
            ++counts[labels[i] * o.clusters + truth[i]];
        for (int k = 0; k < o.clusters; ++k)
            pure += *std::max_element(counts.begin() + k * o.clusters,
                                      counts.begin() + (k + 1) * o.clusters);

        double error = reference_error(x, values, labels, routing, o, false, output);
        double dense_error = o.compare_dense ?
            reference_error(x, values, labels, routing, o, true, dense_output) : 0;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "tokens=" << o.tokens << " dim=" << o.dim << " clusters=" << o.clusters
                  << " iterations=" << o.iterations << " tile=" << o.tile
                  << " passes=" << o.passes << " seed=" << o.seed
                  << " compare_dense=" << o.compare_dense << " batch=" << o.batch
                  << " noise=" << o.noise << " skew=" << o.skew << '\n';
        std::cout << "kmeans_seconds=" << kmeans_sec << " routing_seconds=" << routing_sec
                  << " sparse_attention_seconds=" << sparse_sec;
        if (o.compare_dense) std::cout << " dense_attention_seconds=" << dense_sec
                                       << " dense_over_sparse_attention_time=" << dense_sec / sparse_sec;
        std::cout << " total_compute_seconds=" << kmeans_sec + routing_sec + sparse_sec +
                     (o.compare_dense ? dense_sec : 0) << '\n';
        std::uint64_t dense_pairs = std::uint64_t(o.tokens) * (o.tokens - 1) / 2;
        std::cout << "causal_pairs_per_pass=" << routing.pairs
                  << " dense_pairs=" << dense_pairs
                  << " density=" << double(routing.pairs) /
                     dense_pairs << '\n';
        Accuracy sparse_acc = topic_accuracy(output, truth, routing, o.clusters, false);
        std::cout << "cluster_purity=" << double(pure) / o.tokens
                  << " sparse_evaluated_queries=" << sparse_acc.evaluated
                  << " sparse_retrieval_accuracy=" << sparse_acc.fraction
                  << " sparse_reference_max_abs_error=" << error << '\n';
        std::vector<int> topic_counts(o.clusters);
        for (int topic : truth) ++topic_counts[topic];
        std::cout << "topic_counts=";
        for (int k = 0; k < o.clusters; ++k)
            std::cout << (k == 0 ? "" : ",") << topic_counts[k];
        std::cout << '\n';
        if (o.compare_dense) {
            Accuracy dense_acc = topic_accuracy(dense_output, truth, routing, o.clusters, true);
            std::cout << "dense_evaluated_queries=" << dense_acc.evaluated
                      << " dense_retrieval_accuracy=" << dense_acc.fraction
                       << " dense_reference_max_abs_error=" << dense_error << '\n';
        }
        int last = o.tokens - 1;
        std::cout << "last_query_topic=" << truth[last] << " last_output_payload="
                  << output[last * (o.clusters + 1) + o.clusters] << '\n';
        int prefix = std::min(o.tokens, o.batch) - 1;
        std::cout << "first_batch_last_payload="
                  << output[prefix * (o.clusters + 1) + o.clusters] << '\n';
        return error <= 1e-5 && dense_error <= 1e-5 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'
                  << "Usage: ./cluster_attention [tokens [dim [clusters [iterations [tile [passes [seed [compare_dense [batch [noise [skew]]]]]]]]]]]\n";
        return 1;
    }
}
