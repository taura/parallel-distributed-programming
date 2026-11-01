#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>

class BitVector {
public:
  explicit BitVector(std::size_t size);

  void set(std::size_t i, bool value);
  void build_rank();
  bool access(std::size_t i) const;

  std::size_t rank0(std::size_t end) const;
  std::size_t rank1(std::size_t end) const;

  std::optional<std::size_t> select0(std::size_t k) const;
  std::optional<std::size_t> select1(std::size_t k) const;


  std::size_t size() const noexcept;

private:
  std::size_t size_;
  std::vector<std::uint64_t> words_;
  bool rank_ready_ = false;

  std::vector<std::size_t> prefix_;

  std::optional<std::size_t> select(bool bit, std::size_t k) const;
};
