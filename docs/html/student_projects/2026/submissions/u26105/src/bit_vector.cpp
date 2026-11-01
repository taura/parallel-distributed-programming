#include "bit_vector.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <bit>

BitVector::BitVector(std::size_t size)
    : size_(size), words_(size / 64 + (size % 64 != 0), 0) {}

void BitVector::set(std::size_t i, bool value) {
  if (i >= size_)
    throw std::out_of_range("BitVector::set");

  const std::size_t word = i / 64;
  const std::size_t offset = i % 64;
  const std::uint64_t mask = std::uint64_t{1} << offset;

  if (value) {
    words_[word] |= mask;
  } else {
    words_[word] &= ~mask;
  }
}

void BitVector::build_rank() {
  prefix_.assign(words_.size() + 1, 0);

  for (std::size_t j = 0; j < words_.size(); ++j) {
    prefix_[j + 1] = prefix_[j] + std::popcount(words_[j]);
  }

  rank_ready_ = true;
}

bool BitVector::access(std::size_t i) const {
  if (i >= size_)
    throw std::out_of_range("BitVector::access");

  return (words_[i / 64] >> (i % 64) & std::uint64_t{1}) != 0;
}

std::size_t BitVector::rank1(std::size_t end) const {
  if (end > size_)
    throw std::out_of_range("BitVector::rank0");

  if (!rank_ready_)
    throw std::logic_error("Call build_rank() before rank/select");

  const std::size_t word = end / 64;
  const std::size_t offset = end % 64;

  std::size_t result = prefix_[word];

  if (offset != 0) {
    const auto mask = (std::uint64_t{1} << offset) - 1;
    result += std::popcount(words_[word] & mask);
  }

  return result;
}

std::size_t BitVector::rank0(std::size_t end) const { return end - rank1(end); }

std::optional<std::size_t> BitVector::select(bool bit, std::size_t k) const {
  const std::size_t total = bit ? rank1(size_) : rank0(size_);

  if (k == 0 || k > total)
    return std::nullopt;

  std::size_t lo = 0;
  std::size_t hi = size_;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const std::size_t count = bit ? rank1(mid + 1) : rank0(mid + 1);

    if (count >= k) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  return lo;
}

std::optional<std::size_t> BitVector::select1(std::size_t k) const {
  return select(true, k);
}

std::optional<std::size_t> BitVector::select0(std::size_t k) const {
  return select(false, k);
}

std::size_t BitVector::size() const noexcept { return size_; }
