#include "graph.hpp"

namespace qec {

// =========================================================================
// GraphTopology (Type-1, 3D)
// =========================================================================

GraphTopology::GraphTopology(std::size_t distance, std::size_t window_rounds)
    : params(distance, window_rounds), data_edges(build_data_edges(params)),
      temp_edges(build_temp_edges(params)) {}

std::size_t GraphTopology::total_edge_count() const {
    return data_edges.size() + temp_edges.size();
}

// ---- Edge-to-Node Mapping (matching graph_pkg.sv exactly) ---------------

NodeIdx GraphTopology::data_edge_upper_node(uint8_t y, uint8_t x, uint8_t t, std::size_t /*d*/) {
    if (y == 0) {
        return std::nullopt;
    }
    const auto row = (((int)y + (int)x) % 2 == 0) ? x : static_cast<uint8_t>(x + 1);
    const auto col = static_cast<uint8_t>(((int)y - 1) / 2);
    return NodeCoord{.t = t, .row = row, .col = col};
}

NodeIdx GraphTopology::data_edge_lower_node(uint8_t y, uint8_t x, uint8_t t, std::size_t d) {
    if (static_cast<std::size_t>(y) == d - 1) {
        return std::nullopt;
    }
    const auto row = (((int)y + (int)x) % 2 == 0) ? static_cast<uint8_t>(x + 1) : x;
    const auto col = static_cast<uint8_t>((int)y / 2);
    return NodeCoord{.t = t, .row = row, .col = col};
}

NodeIdx GraphTopology::temp_edge_upper_node(uint8_t t, uint8_t row, uint8_t col) {
    if (t == 0) {
        return std::nullopt;
    }
    return NodeCoord{.t = static_cast<uint8_t>(t - 1), .row = row, .col = col};
}

NodeIdx GraphTopology::temp_edge_lower_node(uint8_t t, uint8_t row, uint8_t col,
                                            std::size_t window_rounds) {
    if (static_cast<std::size_t>(t) >= window_rounds) {
        return std::nullopt;
    }
    return NodeCoord{.t = t, .row = row, .col = col};
}

// ---- Build Functions ----------------------------------------------------

std::vector<DataEdgeInfo> GraphTopology::build_data_edges(const HwDecoderParams& params) {
    const auto d = params.distance;
    const auto m = params.m;
    const auto col_max = params.col_max;
    std::vector<DataEdgeInfo> edges;
    edges.reserve(params.window_rounds * d * d);

    for (std::size_t t = 0; t < params.window_rounds; ++t) {
        for (std::size_t y = 0; y < d; ++y) {
            for (std::size_t x = 0; x < d; ++x) {
                const auto edge = EdgeIdx::data(static_cast<uint8_t>(t), static_cast<uint8_t>(y),
                                                static_cast<uint8_t>(x));
                const auto upper = data_edge_upper_node(
                    static_cast<uint8_t>(y), static_cast<uint8_t>(x), static_cast<uint8_t>(t), d);
                const auto lower = data_edge_lower_node(
                    static_cast<uint8_t>(y), static_cast<uint8_t>(x), static_cast<uint8_t>(t), d);

                std::optional<std::size_t> upper_flat;
                if (upper.has_value()) {
                    upper_flat = upper->flat_index(m, col_max);
                }
                std::optional<std::size_t> lower_flat;
                if (lower.has_value()) {
                    lower_flat = lower->flat_index(m, col_max);
                }

                edges.push_back({.edge = edge,
                                 .upper_node = upper,
                                 .lower_node = lower,
                                 .upper_flat = upper_flat,
                                 .lower_flat = lower_flat});
            }
        }
    }

    return edges;
}

std::vector<TempEdgeInfo> GraphTopology::build_temp_edges(const HwDecoderParams& params) {
    const auto m = params.m;
    const auto col_max = params.col_max;
    const auto wr = params.window_rounds;
    std::vector<TempEdgeInfo> edges;
    edges.reserve((wr + 1) * params.row_max * params.col_max);

    for (std::size_t i = 0; i <= wr; ++i) {
        for (std::size_t row = 0; row < params.row_max; ++row) {
            for (std::size_t col = 0; col < params.col_max; ++col) {
                const auto edge = EdgeIdx::temporal(
                    static_cast<uint8_t>(i), static_cast<uint8_t>(row), static_cast<uint8_t>(col));
                const auto upper = temp_edge_upper_node(
                    static_cast<uint8_t>(i), static_cast<uint8_t>(row), static_cast<uint8_t>(col));
                const auto lower =
                    temp_edge_lower_node(static_cast<uint8_t>(i), static_cast<uint8_t>(row),
                                         static_cast<uint8_t>(col), wr);

                std::optional<std::size_t> upper_flat;
                if (upper.has_value()) {
                    upper_flat = upper->flat_index(m, col_max);
                }
                std::optional<std::size_t> lower_flat;
                if (lower.has_value()) {
                    lower_flat = lower->flat_index(m, col_max);
                }

                edges.push_back({.edge = edge,
                                 .upper_node = upper,
                                 .lower_node = lower,
                                 .upper_flat = upper_flat,
                                 .lower_flat = lower_flat});
            }
        }
    }

    return edges;
}

// ---- Accessors ----------------------------------------------------------

const DataEdgeInfo& GraphTopology::get_data_edge(std::size_t t, std::size_t y,
                                                 std::size_t x) const {
    const auto d = params.distance;
    const auto idx = t * d * d + y * d + x;
    return data_edges[idx];
}

const TempEdgeInfo& GraphTopology::get_temp_edge(std::size_t i, std::size_t row,
                                                 std::size_t col) const {
    const auto rm = params.row_max;
    const auto cm = params.col_max;
    const auto idx = i * rm * cm + row * cm + col;
    return temp_edges[idx];
}

std::size_t GraphTopology::data_edge_flat_idx(std::size_t t, std::size_t y, std::size_t x) const {
    const auto d = params.distance;
    return t * d * d + y * d + x;
}

std::size_t GraphTopology::temp_edge_flat_idx(std::size_t i, std::size_t row,
                                              std::size_t col) const {
    const auto rm = params.row_max;
    const auto cm = params.col_max;
    return i * rm * cm + row * cm + col;
}

// =========================================================================
// GraphTopology2 (Type-2, 2D)
// =========================================================================

GraphTopology2::GraphTopology2(std::size_t distance)
    : distance(distance), m((distance + 1) * (distance - 1) / 2), row_max(distance + 1),
      col_max((distance - 1) / 2), data_edges(build_data_edges(distance, col_max)) {}

NodeIdx2 GraphTopology2::data_edge_upper_node(uint8_t y, uint8_t x, std::size_t /*d*/) {
    if (y == 0) {
        return std::nullopt;
    }
    const auto row = (((int)y + (int)x) % 2 == 0) ? x : static_cast<uint8_t>(x + 1);
    const auto col = static_cast<uint8_t>(((int)y - 1) / 2);
    return NodeCoord2{.row = row, .col = col};
}

NodeIdx2 GraphTopology2::data_edge_lower_node(uint8_t y, uint8_t x, std::size_t d) {
    if (static_cast<std::size_t>(y) == d - 1) {
        return std::nullopt;
    }
    const auto row = (((int)y + (int)x) % 2 == 0) ? static_cast<uint8_t>(x + 1) : x;
    const auto col = static_cast<uint8_t>((int)y / 2);
    return NodeCoord2{.row = row, .col = col};
}

std::vector<DataEdgeInfo2> GraphTopology2::build_data_edges(std::size_t d, std::size_t col_max) {
    std::vector<DataEdgeInfo2> edges;
    edges.reserve(d * d);

    for (std::size_t y = 0; y < d; ++y) {
        for (std::size_t x = 0; x < d; ++x) {
            const auto edge =
                EdgeCoord2{.y = static_cast<uint8_t>(y), .x = static_cast<uint8_t>(x)};
            const auto upper =
                data_edge_upper_node(static_cast<uint8_t>(y), static_cast<uint8_t>(x), d);
            const auto lower =
                data_edge_lower_node(static_cast<uint8_t>(y), static_cast<uint8_t>(x), d);

            std::optional<std::size_t> upper_flat;
            if (upper.has_value()) {
                upper_flat = upper->flat_index(col_max);
            }
            std::optional<std::size_t> lower_flat;
            if (lower.has_value()) {
                lower_flat = lower->flat_index(col_max);
            }

            edges.push_back({.edge = edge,
                             .upper_node = upper,
                             .lower_node = lower,
                             .upper_flat = upper_flat,
                             .lower_flat = lower_flat});
        }
    }

    return edges;
}

const DataEdgeInfo2& GraphTopology2::get_data_edge(std::size_t y, std::size_t x) const {
    const auto idx = y * distance + x;
    return data_edges[idx];
}

std::size_t GraphTopology2::data_edge_flat_idx(std::size_t y, std::size_t x) const {
    return y * distance + x;
}

// =========================================================================
// Stim <-> SV Coordinate Conversion
// =========================================================================

std::size_t edge_flat_index(const EdgeIdx& edge, const HwDecoderParams& params) {
    const auto d = params.distance;
    if (edge.is_temporal) {
        const auto data_count = params.window_rounds * d * d;
        return data_count + static_cast<std::size_t>(edge.t) * params.row_max * params.col_max +
               static_cast<std::size_t>(edge.x) * params.col_max + // x = row for temporal
               static_cast<std::size_t>(edge.y);                   // y = col for temporal
    }
    return static_cast<std::size_t>(edge.t) * d * d + static_cast<std::size_t>(edge.y) * d +
           static_cast<std::size_t>(edge.x);
}

NodeIdx stim_to_node(int64_t stim_x, int64_t stim_y, const HwDecoderParams& params) {
    if (stim_x < 0 || stim_x % 2 != 0) {
        return std::nullopt;
    }
    const auto row = static_cast<uint8_t>(stim_x / 2);
    uint8_t col;

    if (row % 2 == 0) {
        if (stim_y < 4 || (stim_y - 4) % 4 != 0) {
            return std::nullopt;
        }
        col = static_cast<uint8_t>((stim_y - 4) / 4);
    } else {
        if (stim_y < 2 || (stim_y - 2) % 4 != 0) {
            return std::nullopt;
        }
        col = static_cast<uint8_t>((stim_y - 2) / 4);
    }

    if (static_cast<std::size_t>(row) < params.row_max &&
        static_cast<std::size_t>(col) < params.col_max) {
        return NodeCoord{.t = 0, .row = row, .col = col};
    }
    return std::nullopt;
}

std::vector<bool> fired_to_syndrome(const std::vector<std::size_t>& fired,
                                    const std::vector<Detector>& detectors,
                                    const HwDecoderParams& params, std::size_t window_start,
                                    std::size_t window_end) {
    const auto num_nodes = (window_end - window_start) * params.m;
    std::vector<bool> syndrome(num_nodes, false);

    for (std::size_t det_id : fired) {
        if (det_id >= detectors.size())
            continue;
        const auto& det = detectors[det_id];
        const auto round = static_cast<std::size_t>(det.t / 2);
        if (round >= window_start && round < window_end) {
            const auto local_t = round - window_start;
            const auto node_opt = stim_to_node(det.x, det.y, params);
            if (node_opt.has_value()) {
                const auto& node = node_opt.value();
                const auto flat = local_t * params.m +
                                  static_cast<std::size_t>(node.row) * params.col_max +
                                  static_cast<std::size_t>(node.col);
                if (flat < num_nodes) {
                    syndrome[flat] = syndrome[flat] ^ true;
                }
            }
        }
    }

    return syndrome;
}

} // namespace qec
