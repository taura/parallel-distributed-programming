// Monte Carlo logical error rate of the surface-code 3D decoder.
// Phenomenological noise: data edges and interior temporal edges flip with prob. p.
#include "fsm.hpp"
#include "graph.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace qec;

// Toggle the syndrome of every existing endpoint node of an edge.
template <class EdgeInfo> static void flip_edge(const EdgeInfo& e, std::vector<bool>& syndrome) {
    if (e.upper_flat)
        syndrome[*e.upper_flat] = !syndrome[*e.upper_flat];
    if (e.lower_flat)
        syndrome[*e.lower_flat] = !syndrome[*e.lower_flat];
}

// Decode one syndrome; return the logical parity of the correction (row y == 0).
static bool decode_logical(HwDecoderFsm& fsm, const std::vector<bool>& syndrome, std::size_t d) {
    const DecoderOutput out = fsm.run(syndrome);
    bool parity = false;
    for (std::size_t x = 0; x < d; ++x)
        parity ^= out.corrections[0][x];
    return parity;
}

// Flip each single edge in turn; every one must be decoded without a logical error.
static bool selftest() {
    bool ok = true;
    for (std::size_t d : {3, 5, 7}) {
        const std::size_t rounds = d;
        GraphTopology graph(d, rounds);
        HwDecoderFsm fsm(graph, 0, rounds, false, false);
        int tested = 0, failed = 0;
        for (const auto& e : graph.data_edges) {
            std::vector<bool> syn(graph.params.node_count, false);
            flip_edge(e, syn);
            ++tested;
            if (decode_logical(fsm, syn, d) != (e.edge.y == 0)) {
                std::printf("FAIL d=%zu data t=%d y=%d x=%d\n", d, e.edge.t, e.edge.y, e.edge.x);
                ++failed;
            }
        }
        for (const auto& e : graph.temp_edges) {
            if (e.edge.t < 1 || e.edge.t >= rounds)
                continue;
            std::vector<bool> syn(graph.params.node_count, false);
            flip_edge(e, syn);
            ++tested;
            if (decode_logical(fsm, syn, d)) {
                std::printf("FAIL d=%zu temporal t=%d y=%d x=%d\n", d, e.edge.t, e.edge.y,
                            e.edge.x);
                ++failed;
            }
        }
        std::printf("selftest d=%zu: %d edges, %d failed\n", d, tested, failed);
        ok = ok && failed == 0;
    }
    return ok;
}

// Mix (seed, shot) into one 64-bit seed (splitmix64), so every shot has its own RNG stream.
static uint64_t shot_seed(uint64_t seed, uint64_t shot) {
    uint64_t z = seed * 0x9E3779B97F4A7C15ULL + shot + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int main(int argc, char** argv) {
    std::size_t d = 5, rounds = 0, shots = 100000;
    double p = 0.02;
    uint64_t seed = 1;
    bool run_selftest = false;

    for (int i = 1; i < argc; ++i) {
        const bool has_val = i + 1 < argc;
        if (!std::strcmp(argv[i], "--selftest"))
            run_selftest = true;
        else if (!std::strcmp(argv[i], "--d") && has_val)
            d = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--rounds") && has_val)
            rounds = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--p") && has_val)
            p = std::strtod(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--shots") && has_val)
            shots = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--seed") && has_val)
            seed = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::fprintf(stderr, "bad argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (run_selftest)
        return selftest() ? 0 : 1;

    if (rounds == 0)
        rounds = d;
    if (d < 3 || d % 2 == 0 || rounds < 1 || rounds > 200 || p < 0 || p > 1) {
        std::fprintf(stderr, "need odd d >= 3, 1 <= rounds <= 200, 0 <= p <= 1\n");
        return 2;
    }

    GraphTopology graph(d, rounds);
    HwDecoderFsm fsm(graph, 0, rounds, false, false); // one window, no seams, reused for all shots

    using clock = std::chrono::steady_clock;
    double sample_sec = 0, decode_sec = 0;
    std::size_t logical_errors = 0, nonempty = 0, total_defects = 0;
    std::vector<bool> syndrome(graph.params.node_count);

    for (std::size_t shot = 0; shot < shots; ++shot) {
        // Sample noise: toggle syndromes and track the true logical parity.
        auto t0 = clock::now();
        std::mt19937_64 rng(shot_seed(seed, shot)); // per-shot stream: shots are independent
        std::bernoulli_distribution flip(p);
        syndrome.assign(syndrome.size(), false);
        bool true_logical = false;
        for (const auto& e : graph.data_edges) {
            if (!flip(rng))
                continue;
            flip_edge(e, syndrome);
            if (e.edge.y == 0)
                true_logical = !true_logical;
        }
        for (const auto& e : graph.temp_edges) {
            if (e.edge.t < 1 || e.edge.t >= rounds)
                continue; // boundary edges are unused
            if (flip(rng))
                flip_edge(e, syndrome);
        }
        std::size_t defects = 0;
        for (bool b : syndrome)
            defects += b;
        auto t1 = clock::now();

        // Decode and compare the logical parity.
        const bool decoded_logical = decode_logical(fsm, syndrome, d);
        auto t2 = clock::now();

        logical_errors += decoded_logical != true_logical;
        nonempty += defects > 0;
        total_defects += defects;
        sample_sec += std::chrono::duration<double>(t1 - t0).count();
        decode_sec += std::chrono::duration<double>(t2 - t1).count();
    }

    std::printf("d,rounds,p,shots,seed,logical_errors,logical_error_rate,"
                "nonempty_syndrome_shots,mean_defects_per_shot,sample_sec,decode_sec\n");
    std::printf("%zu,%zu,%g,%zu,%llu,%zu,%g,%zu,%g,%.6f,%.6f\n", d, rounds, p, shots,
                static_cast<unsigned long long>(seed), logical_errors,
                shots ? double(logical_errors) / shots : 0.0, nonempty,
                shots ? double(total_defects) / shots : 0.0, sample_sec, decode_sec);
    return 0;
}
