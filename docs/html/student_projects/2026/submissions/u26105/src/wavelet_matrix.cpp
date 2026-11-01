#include "wavelet_matrix.hpp"

#include <bit>
#include <stdexcept>

WaveletMatrix::WaveletMatrix(const std::vector<std::uint32_t> &data)
    : size_(data.size()), height_(1), max_value_(0) {
  for (std::size_t i = 0; i < data.size(); ++i) {
    // it's better to use std::max_element when the data scale is small.
    if (data[i] > max_value_)
      max_value_ = data[i];
  }

  if (max_value_ != 0)
    height_ = static_cast<unsigned>(std::bit_width(max_value_));

  levels_.reserve(height_);
  zeros_.reserve(height_);

  std::vector<std::uint32_t> current = data;
  std::vector<std::uint32_t> next(size_);

  for (unsigned level = 0; level < height_; ++level) {
    const unsigned shift = height_ - 1 - level;

    levels_.emplace_back(size_);
    auto &bits = levels_.back();

    std::size_t zero_count = 0;

    for (std::size_t i = 0; i < size_; ++i) {
      const bool bit = ((current[i] >> shift) & 1U) != 0;

      bits.set(i, bit);

      if (!bit)
        ++zero_count;
    }

    bits.build_rank();
    zeros_.push_back(zero_count);

    std::size_t zero_pos = 0;
    std::size_t one_pos = zero_count;

    for (std::size_t i = 0; i < size_; ++i) {
      if (bits.access(i))
        next[one_pos++] = current[i];
      else
        next[zero_pos++] = current[i];
    }

    current.swap(next);
  }
}

std::uint32_t WaveletMatrix::access(std::size_t i) const {
  if (i >= size_)
    throw std::out_of_range("WaveletMatrix::access");

  std::uint32_t value = 0;

  for (unsigned level = 0; level < height_; ++level) {
    const auto &bits = levels_[level];
    const bool bit = bits.access(i);

    value = (value << 1) | static_cast<std::uint32_t>(bit);

    if (bit)
      i = zeros_[level] + bits.rank1(i);
    else
      i = bits.rank0(i);
  }

  return value;
}

std::pair<std::size_t, std::size_t>
WaveletMatrix::descend(std::uint32_t value, std::size_t l,
                       std::size_t r) const {
  for (unsigned level = 0; level < height_; ++level) {
    const unsigned shift = height_ - 1 - level;
    const bool bit = ((value >> shift) & 1U) != 0;
    const auto &bits = levels_[level];

    if (bit) {
      l = zeros_[level] + bits.rank1(l);
      r = zeros_[level] + bits.rank1(r);
    } else {
      l = bits.rank0(l);
      r = bits.rank0(r);
    }
  }

  return {l, r};
}

std::size_t WaveletMatrix::rank(std::uint32_t value, std::size_t end) const {
  if (end > size_)
    throw std::out_of_range("WaveletMatrix::rank");

  if (value > max_value_)
    return 0;

  const auto [l, r] = descend(value, 0, end);
  return r - l;
}

std::optional<std::size_t> WaveletMatrix::select(std::uint32_t value,
                                                 std::size_t k) const {
  if (k == 0 || value > max_value_)
    return std::nullopt;

  const auto [l, r] = descend(value, 0, size_);

  if (k > r - l)
    return std::nullopt;

  std::size_t pos = l + k - 1;

  for (unsigned depth = height_; depth > 0; --depth) {
    const unsigned level = depth - 1;
    const unsigned shift = height_ - 1 - level;
    const bool bit = ((value >> shift) & 1U) != 0;
    const auto &bits = levels_[level];

    const auto previous =
        bit ? bits.select1(pos - zeros_[level] + 1) : bits.select0(pos + 1);

    if (!previous)
      throw std::logic_error("Invalid WaveletMatrix mapping");

    pos = *previous;
  }

  return pos;
}

std::size_t WaveletMatrix::size() const noexcept { return size_; }
