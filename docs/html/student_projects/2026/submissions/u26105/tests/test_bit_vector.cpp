#include "bit_vector.hpp"

#include <cassert>
#include <iostream>

int main() {
    BitVector bits(8);

    assert(bits.size() == 8);
    assert(bits.access(0) == false);

    bits.set(1, true);
    bits.set(3, true);
    bits.set(4, true);
    bits.set(7, true);

    assert(bits.access(1) == true);
    assert(bits.access(2) == false);

    bits.build_rank();

    assert(bits.rank1(0) == 0);
    assert(bits.rank1(5) == 3);
    assert(bits.rank0(5) == 2);
    assert(bits.rank1(8) == 4);
    assert(bits.rank0(8) == 4);

    assert(bits.select1(1) == 1);
    assert(bits.select1(3) == 4);
    assert(bits.select0(1) == 0);
    assert(bits.select0(3) == 5);

    assert(!bits.select1(0).has_value());
    assert(!bits.select1(5).has_value());
    assert(!bits.select0(5).has_value());

    bits.set(3, false);
    bits.build_rank();

    assert(bits.access(3) == false);
    assert(bits.rank1(8) == 3);
    assert(bits.select1(2) == 4);

    BitVector large(65);
    large.set(64, true);
    large.build_rank();

    assert(large.access(64) == true);
    assert(large.rank1(64) == 0);
    assert(large.rank1(65) == 1);
    assert(large.select1(1) == 64);

    std::cout << "All tests passed!\n";
}
