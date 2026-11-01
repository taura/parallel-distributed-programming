#include "wavelet_matrix.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  const std::vector<std::uint32_t> data{3, 1, 4, 1, 5, 7, 2, 6};

  WaveletMatrix matrix(data);

  assert(matrix.size() == data.size());

  for (std::size_t i = 0; i < data.size(); ++i) {
    assert(matrix.access(i) == data[i]);
  }

  assert(matrix.rank(1, 0) == 0);
  assert(matrix.rank(1, 3) == 1);
  assert(matrix.rank(1, 4) == 2);
  assert(matrix.rank(1, 8) == 2);
  assert(matrix.rank(0, 8) == 0);
  assert(matrix.rank(9, 8) == 0);

  assert(matrix.select(1, 1) == 1);
  assert(matrix.select(1, 2) == 3);
  assert(matrix.select(3, 1) == 0);
  assert(matrix.select(6, 1) == 7);

  assert(!matrix.select(1, 0).has_value());
  assert(!matrix.select(1, 3).has_value());
  assert(!matrix.select(0, 1).has_value());

  WaveletMatrix zeros(std::vector<std::uint32_t>{0, 0, 0});
  assert(zeros.access(2) == 0);
  assert(zeros.rank(0, 3) == 3);
  assert(zeros.select(0, 3) == 2);

  WaveletMatrix empty(std::vector<std::uint32_t>{});
  assert(empty.size() == 0);
  assert(empty.rank(0, 0) == 0);
  assert(!empty.select(0, 1).has_value());

  std::cout << "All WaveletMatrix tests passed!\n";
}
