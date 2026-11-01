#pragma once

#include "bit_vector.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

class WaveletMatrix {
public:
  explicit WaveletMatrix(const std::vector<std::uint32_t> &data);

  std::uint32_t access(std::size_t i) const;
  std::size_t rank(std::uint32_t value, std::size_t end) const;
  std::optional<std::size_t> select(std::uint32_t value, std::size_t k) const;

  std::size_t size() const noexcept;

private:
  std::size_t size_;
  unsigned height_;
  std::uint32_t max_value_;

  std::vector<BitVector> levels_;
  std::vector<std::size_t> zeros_;

  std::pair<std::size_t, std::size_t>
  descend(std::uint32_t value, std::size_t l, std::size_t r) const;
};
