// Serial SHA-256 Proof of Work baseline. Standard library only, C++17.
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Digest = std::array<std::uint32_t, 8>;

std::uint32_t rotr(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

Digest sha256(const std::string& message) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    Digest h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    // Append 1 bit, zero padding, and the original bit length (big endian).
    std::vector<std::uint8_t> data(message.begin(), message.end());
    const std::uint64_t bit_length = std::uint64_t(message.size()) * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) {
        data.push_back(0);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<std::uint8_t>(bit_length >> shift));
    }

    for (std::size_t offset = 0; offset < data.size(); offset += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const std::size_t p = offset + 4 * i;
            w[i] = (std::uint32_t(data[p]) << 24)
                 | (std::uint32_t(data[p + 1]) << 16)
                 | (std::uint32_t(data[p + 2]) << 8)
                 | std::uint32_t(data[p + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18)
                          ^ (w[i - 15] >> 3);
            const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19)
                          ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto a = h[0], b = h[1], c = h[2], d = h[3];
        auto e = h[4], f = h[5], g = h[6], z = h[7];
        for (int i = 0; i < 64; ++i) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto t1 = z + s1 + choose + k[i] + w[i];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            z = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += z;
    }
    return h;
}

std::string hex_digest(const Digest& h) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto word : h) {
        out << std::setw(8) << word;
    }
    return out.str();
}

unsigned leading_zero_bits(const Digest& h) {
    unsigned count = 0;
    for (auto word : h) {
        for (int bit = 31; bit >= 0; --bit) {
            if ((word >> bit) & 1u) {
                return count;
            }
            ++count;
        }
    }
    return count;
}

void self_test() {
    // Known SHA-256 vectors, including a message needing two padded blocks.
    // https://www.nist.gov/itl/ai/ai-standards-and-guidelines-group/nsrl-test-data
    const std::array<std::string, 3> messages = {
        "", "abc", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    };
    const std::array<std::string, 3> expected = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    };
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (hex_digest(sha256(messages[i])) != expected[i]) {
            throw std::runtime_error("SHA-256 self-test failed");
        }
    }
    Digest h{};
    if (leading_zero_bits(h) != 256) {
        throw std::runtime_error("zero-bit test failed");
    }
    for (unsigned n = 0; n < 256; ++n) {
        h.fill(0);
        h[n / 32] = std::uint32_t(1) << (31 - n % 32);
        if (leading_zero_bits(h) != n) {
            throw std::runtime_error("zero-bit test failed");
        }
    }
}

std::uint64_t parse_uint(const std::string& s) {
    if (s.empty()) {
        throw std::invalid_argument("empty number");
    }
    std::uint64_t n = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("expected unsigned decimal integer: " + s);
        }
        const unsigned digit = c - '0';
        if (n > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::invalid_argument("integer too large: " + s);
        }
        n = n * 10 + digit;
    }
    return n;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: ./pow [solve|bench [prefix [bits [limit]]]]\n"
                      << "       ./pow --self-test\n"
                      << "Defaults: solve ctf-demo: 20 1000000000\n"
                      << "Hash input: prefix + decimal nonce (no separator/newline added).\n"
                      << "Nonces: 0 through limit-1. bits: 0..256; limit: positive.\n"
                      << "solve stops at first match; bench scans the entire range.\n";
            return 0;
        }
        self_test();
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            std::cout << "self_test=PASS\n";
            return 0;
        }
        if (argc > 5) {
            throw std::invalid_argument("too many arguments; use --help");
        }
        const std::string mode = argc > 1 ? argv[1] : "solve";
        const std::string prefix = argc > 2 ? argv[2] : "ctf-demo:";
        const auto bits = argc > 3 ? parse_uint(argv[3]) : 20;
        const auto limit = argc > 4 ? parse_uint(argv[4]) : 1000000000;
        if (mode != "solve" && mode != "bench") {
            throw std::invalid_argument("mode must be solve or bench");
        }
        if (bits > 256 || limit == 0) {
            throw std::invalid_argument("bits must be <=256 and limit must be >0");
        }

        std::uint64_t attempts = 0, matches = 0, first_nonce = 0;
        Digest first_hash{};
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint64_t nonce = 0; nonce < limit; ++nonce) {
            const Digest h = sha256(prefix + std::to_string(nonce));
            ++attempts;
            if (leading_zero_bits(h) >= bits) {
                if (matches == 0) {
                    first_nonce = nonce;
                    first_hash = h;
                }
                ++matches;
                if (mode == "solve") {
                    break;
                }
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        std::cout << "self_test=PASS\nmode=" << mode << " bits=" << bits
                  << " limit=" << limit << "\nattempts=" << attempts
                  << " matches=" << matches << '\n'
                  << std::fixed << std::setprecision(6) << "seconds=" << seconds
                  << "\nhashes_per_second=" << (seconds > 0 ? attempts / seconds : 0) << '\n';
        if (matches > 0) {
            const std::string input = prefix + std::to_string(first_nonce);
            const Digest checked = sha256(input);
            if (checked != first_hash || leading_zero_bits(checked) < bits) {
                throw std::runtime_error("solution verification failed");
            }
            std::cout << "nonce=" << first_nonce << "\ninput=" << std::quoted(input)
                      << "\nsha256=" << hex_digest(first_hash)
                      << "\nleading_zero_bits=" << leading_zero_bits(first_hash)
                      << "\nverification=PASS\n";
        } else {
            std::cout << "No solution in the tested range. Increase limit or lower bits.\n";
        }
        // An exhausted range is a valid computation, not a program error.
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
