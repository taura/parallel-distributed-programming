// code/pattern_scanner.cc
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static uint64_t xorshift64(uint64_t &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Fill data with bytes in [0, 127] only.
static void fill_background(uint8_t *data, size_t n, uint64_t &seed) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (uint8_t)(xorshift64(seed) & 0x7F);
    }
}

// Pattern bytes are in [128, 255], so background can never match accidentally.
static void make_pattern(uint8_t *pat, size_t m, uint64_t &seed) {
    for (size_t i = 0; i < m; i++) {
        pat[i] = (uint8_t)(0x80 | (xorshift64(seed) & 0x7F));
    }
    pat[0] = 0xFE; // fixed first byte for readability
}

// Inject known matches at deterministic non-overlapping positions.
static std::vector<size_t> inject_pattern(uint8_t *data, size_t n,
                                          const uint8_t *pat, size_t m,
                                          size_t requested_hits) {
    std::vector<size_t> expected;
    if (n < m || m == 0) return expected;

    size_t max_hits = (n - m) / (m + 1) + 1;
    size_t hits = requested_hits < max_hits ? requested_hits : max_hits;

    expected.reserve(hits);
    for (size_t k = 0; k < hits; k++) {
        size_t pos = k * (m + 1); // guaranteed non-overlap
        std::memcpy(data + pos, pat, m);
        expected.push_back(pos);
    }
    return expected;
}

// Naive serial scan: O(n * m)
static std::vector<size_t> scan_naive(const uint8_t *data, size_t n,
                                      const uint8_t *pat, size_t m) {
    std::vector<size_t> hits;
    if (m == 0 || n < m) return hits;

    for (size_t i = 0; i + m <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < m; j++) {
            if (data[i + j] != pat[j]) {
                ok = false;
                break;
            }
        }
        if (ok) hits.push_back(i);
    }
    return hits;
}

static bool verify_hits(const uint8_t *data, size_t n,
                        const uint8_t *pat, size_t m,
                        const std::vector<size_t> &expected,
                        const std::vector<size_t> &found) {
    if (expected.size() != found.size()) return false;

    // Check exact positions (both are sorted by construction).
    for (size_t i = 0; i < expected.size(); i++) {
        if (expected[i] != found[i]) return false;
    }

    // Extra safety: each found position really matches bytes.
    for (size_t idx = 0; idx < found.size(); idx++) {
        size_t p = found[idx];
        if (p + m > n) return false;
        for (size_t j = 0; j < m; j++) {
            if (data[p + j] != pat[j]) return false;
        }
    }
    return true;
}

static uint64_t checksum_positions(const std::vector<size_t> &v) {
    uint64_t h = 1469598103934665603ULL; // FNV offset
    for (size_t i = 0; i < v.size(); i++) {
        h ^= (uint64_t)v[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    // Defaults: adjust to get seconds -> minutes.
    size_t data_mb = 256;
    size_t pat_len = 8;
    size_t injected_hits = 20000;
    uint64_t seed = 0x12345678ABCDEF00ULL;

    if (argc > 1) data_mb = std::strtoull(argv[1], nullptr, 10);
    if (argc > 2) pat_len = std::strtoull(argv[2], nullptr, 10);
    if (argc > 3) injected_hits = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) seed = std::strtoull(argv[4], nullptr, 10);

    size_t n = data_mb * 1024ULL * 1024ULL;
    if (pat_len == 0 || n < pat_len) {
        std::cerr << "Invalid sizes: data must be >= pattern length, pattern > 0\n";
        return 1;
    }

    uint8_t *data = (uint8_t *)std::malloc(n);
    uint8_t *pat = (uint8_t *)std::malloc(pat_len);
    if (!data || !pat) {
        std::cerr << "Allocation failed\n";
        std::free(data);
        std::free(pat);
        return 1;
    }

    uint64_t rng = seed;
    fill_background(data, n, rng);
    make_pattern(pat, pat_len, rng);
    std::vector<size_t> expected = inject_pattern(data, n, pat, pat_len, injected_hits);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<size_t> found = scan_naive(data, n, pat, pat_len);
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    bool ok = verify_hits(data, n, pat, pat_len, expected, found);

    double gib = (double)n / (1024.0 * 1024.0 * 1024.0);
    double gibps = gib / sec;

    std::cout << "data_mb=" << data_mb
              << " pattern_len=" << pat_len
              << " injected_hits=" << expected.size()
              << " seed=" << seed << "\n";
    std::cout << "found_hits=" << found.size()
              << " verify=" << (ok ? "OK" : "FAIL")
              << " checksum=0x" << std::hex << checksum_positions(found) << std::dec << "\n";
    std::cout << "scan_time_sec=" << sec
              << " throughput_GiB_per_sec=" << gibps << "\n";

    std::free(data);
    std::free(pat);
    return ok ? 0 : 2;
}