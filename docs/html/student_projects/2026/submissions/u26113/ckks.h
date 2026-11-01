#pragma once
#include <vector>
#include <complex>
#include <cstdint>

namespace ckks {

struct Ciphertext {
    std::vector<int64_t> c0;
    std::vector<int64_t> c1;
    double scale;
};

class CKKS {
private:
    int64_t N_, MOD_, PRIM_ROOT_;
    std::vector<int64_t> secret_key_;
    std::vector<int64_t> public_key_b_;
    std::vector<int64_t> public_key_a_;

    std::vector<int64_t> encode(const std::vector<double>& plain, double scale) const;
    std::vector<double> decode(const std::vector<int64_t>& poly, double scale, size_t slots) const;

public:
    CKKS(int64_t N, int64_t MOD, int64_t PRIM_ROOT);
    void generateKeyPair();
    Ciphertext encrypt(const std::vector<double>& plain, double scale) const;
    std::vector<double> decrypt(const Ciphertext& cipher, size_t slots) const;
};

} // namespace ckks