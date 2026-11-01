#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace qec {

// =========================================================================
// Detector
// =========================================================================

struct Detector {
    std::size_t id;
    int64_t x;
    int64_t y;
    int64_t t;

    Detector() : id(0), x(0), y(0), t(0) {}
    Detector(std::size_t id, int64_t x, int64_t y, int64_t t) : id(id), x(x), y(y), t(t) {}

    [[nodiscard]] int64_t graph_distance(const Detector& other) const {
        const auto dx = std::abs(x - other.x);
        const auto dy = std::abs(y - other.y);
        const auto dt = std::abs(t - other.t);
        return std::max(dx, dy) + dt;
    }

    [[nodiscard]] int64_t distance_to_boundary_y(int64_t boundary_y) const {
        return std::abs(y - boundary_y);
    }

    [[nodiscard]] int64_t distance_to_boundary_x(int64_t boundary_x) const {
        return std::abs(x - boundary_x);
    }

    bool operator==(const Detector& other) const {
        return id == other.id && x == other.x && y == other.y && t == other.t;
    }
};

} // namespace qec
