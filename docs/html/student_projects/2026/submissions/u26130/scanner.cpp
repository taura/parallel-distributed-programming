#include "scanner.hpp"

#include <optional>

namespace qec {

// =========================================================================
// Priority Encoder Helpers (matching find_first_one in SV)
// =========================================================================

static std::optional<std::size_t> find_first_one(const std::vector<bool>& vec) {
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (vec[i])
            return i;
    }
    return std::nullopt;
}

static std::optional<std::size_t> find_first_one_after(const std::vector<bool>& vec,
                                                       std::size_t after) {
    for (std::size_t i = after + 1; i < vec.size(); ++i) {
        if (vec[i])
            return i;
    }
    return std::nullopt;
}

static std::optional<std::size_t> find_next_row(const std::vector<bool>& zdr_row,
                                                std::size_t current_y) {
    for (std::size_t i = current_y + 1; i < zdr_row.size(); ++i) {
        if (zdr_row[i])
            return i;
    }
    return std::nullopt;
}

static bool all_false(const std::vector<bool>& v) {
    for (const bool b : v) {
        if (b)
            return false;
    }
    return true;
}

// =========================================================================
// MergeScanner (Type-1)
// =========================================================================

std::vector<EdgeIdx>
MergeScanner::scan_all(const std::vector<std::vector<bool>>& data_zdr_row,
                       const std::vector<std::vector<bool>>& temp_zdr_col,
                       const std::function<bool(const EdgeIdx&)>& edge_merge_required,
                       std::size_t window_rounds, std::size_t distance, std::size_t row_max,
                       std::size_t col_max) {
    std::vector<EdgeIdx> results;

    // Phase 1: Temporal edges (i=0..window_rounds inclusive)
    for (std::size_t i = 0; i <= window_rounds; ++i) {
        if (i < temp_zdr_col.size() && all_false(temp_zdr_col[i])) {
            continue;
        }
        for (std::size_t col = 0; col < col_max; ++col) {
            if (i < temp_zdr_col.size() && col < temp_zdr_col[i].size() && !temp_zdr_col[i][col]) {
                continue;
            }
            for (std::size_t row = 0; row < row_max; ++row) {
                const auto edge = EdgeIdx::temporal(
                    static_cast<uint8_t>(i), static_cast<uint8_t>(row), static_cast<uint8_t>(col));
                if (edge_merge_required(edge)) {
                    results.push_back(edge);
                }
            }
        }
    }

    // Phase 2: Data edges (t=0..window_rounds-1)
    for (std::size_t t = 0; t < window_rounds; ++t) {
        if (t < data_zdr_row.size() && all_false(data_zdr_row[t])) {
            continue;
        }
        for (std::size_t y = 0; y < distance; ++y) {
            if (t < data_zdr_row.size() && y < data_zdr_row[t].size() && !data_zdr_row[t][y]) {
                continue;
            }
            for (std::size_t x = 0; x < distance; ++x) {
                const auto edge = EdgeIdx::data(static_cast<uint8_t>(t), static_cast<uint8_t>(y),
                                                static_cast<uint8_t>(x));
                if (edge_merge_required(edge)) {
                    results.push_back(edge);
                }
            }
        }
    }

    return results;
}

std::vector<EdgeIdx>
MergeScanner::scan_all_flat(const std::vector<bool>& data_zdr_row, std::size_t data_cols,
                            const std::vector<bool>& temp_zdr_col, std::size_t temp_cols,
                            const std::function<bool(const EdgeIdx&)>& edge_merge_required,
                            std::size_t window_rounds, std::size_t distance, std::size_t row_max,
                            std::size_t col_max) {
    std::vector<EdgeIdx> results;

    // Phase 1: Temporal edges (i=0..window_rounds inclusive)
    for (std::size_t i = 0; i <= window_rounds; ++i) {
        bool row_all_false = true;
        for (std::size_t c = 0; c < temp_cols; ++c) {
            if (temp_zdr_col[i * temp_cols + c]) {
                row_all_false = false;
                break;
            }
        }
        if (row_all_false) {
            continue;
        }
        for (std::size_t col = 0; col < col_max; ++col) {
            if (!temp_zdr_col[i * temp_cols + col]) {
                continue;
            }
            for (std::size_t row = 0; row < row_max; ++row) {
                const auto edge = EdgeIdx::temporal(
                    static_cast<uint8_t>(i), static_cast<uint8_t>(row), static_cast<uint8_t>(col));
                if (edge_merge_required(edge)) {
                    results.push_back(edge);
                }
            }
        }
    }

    // Phase 2: Data edges (t=0..window_rounds-1)
    for (std::size_t t = 0; t < window_rounds; ++t) {
        bool row_all_false = true;
        for (std::size_t c = 0; c < data_cols; ++c) {
            if (data_zdr_row[t * data_cols + c]) {
                row_all_false = false;
                break;
            }
        }
        if (row_all_false) {
            continue;
        }
        for (std::size_t y = 0; y < distance; ++y) {
            if (!data_zdr_row[t * data_cols + y]) {
                continue;
            }
            for (std::size_t x = 0; x < distance; ++x) {
                const auto edge = EdgeIdx::data(static_cast<uint8_t>(t), static_cast<uint8_t>(y),
                                                static_cast<uint8_t>(x));
                if (edge_merge_required(edge)) {
                    results.push_back(edge);
                }
            }
        }
    }

    return results;
}

// =========================================================================
// MergeScanner2 (Type-2)
// =========================================================================

std::vector<EdgeCoord2>
MergeScanner2::scan_all(const std::vector<bool>& zdr_row,
                        const std::vector<std::vector<bool>>& merge_required,
                        std::size_t /*distance*/) {
    std::vector<EdgeCoord2> results;

    if (all_false(zdr_row)) {
        return results;
    }

    const auto current_y_opt = find_first_one(zdr_row);
    if (!current_y_opt.has_value()) {
        return results;
    }
    std::size_t current_y = current_y_opt.value();

    while (true) {
        auto current_x_opt = find_first_one(merge_required[current_y]);
        if (!current_x_opt.has_value()) {
            auto next_y_opt = find_next_row(zdr_row, current_y);
            if (!next_y_opt.has_value())
                break;
            current_y = next_y_opt.value();
            continue;
        }
        std::size_t current_x = current_x_opt.value();

        while (true) {
            results.push_back(
                EdgeCoord2{static_cast<uint8_t>(current_y), static_cast<uint8_t>(current_x)});

            auto next_x_opt = find_first_one_after(merge_required[current_y], current_x);
            if (!next_x_opt.has_value())
                break;
            current_x = next_x_opt.value();
        }

        auto next_y_opt = find_next_row(zdr_row, current_y);
        if (!next_y_opt.has_value())
            break;
        current_y = next_y_opt.value();
    }

    return results;
}

} // namespace qec
