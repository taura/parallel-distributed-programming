#pragma once

#include "sv_types.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace qec {

// =========================================================================
// Type-1 Scanner (merge_scanner_t1.sv)
// =========================================================================

class MergeScanner {
public:
    MergeScanner() = default;

    std::vector<EdgeIdx> scan_all(const std::vector<std::vector<bool>>& data_zdr_row,
                                  const std::vector<std::vector<bool>>& temp_zdr_col,
                                  const std::function<bool(const EdgeIdx&)>& edge_merge_required,
                                  std::size_t window_rounds, std::size_t distance,
                                  std::size_t row_max, std::size_t col_max);

    std::vector<EdgeIdx> scan_all_flat(const std::vector<bool>& data_zdr_row, std::size_t data_cols,
                                       const std::vector<bool>& temp_zdr_col, std::size_t temp_cols,
                                       const std::function<bool(const EdgeIdx&)>& edge_merge_required,
                                       std::size_t window_rounds, std::size_t distance,
                                       std::size_t row_max, std::size_t col_max);
};

// =========================================================================
// Type-2 Scanner (merge_scanner_t2.sv)
// =========================================================================

class MergeScanner2 {
public:
    MergeScanner2() = default;

    std::vector<EdgeCoord2> scan_all(const std::vector<bool>& zdr_row,
                                     const std::vector<std::vector<bool>>& merge_required,
                                     std::size_t distance);
};

} // namespace qec
