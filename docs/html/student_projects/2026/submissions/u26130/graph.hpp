#pragma once

#include "common.hpp"
#include "sv_types.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

namespace qec {

// =========================================================================
// Precomputed Edge Info (Type-1, 3D)
// =========================================================================

struct DataEdgeInfo {
    EdgeIdx edge;
    NodeIdx upper_node;
    NodeIdx lower_node;
    std::optional<std::size_t> upper_flat;
    std::optional<std::size_t> lower_flat;
};

struct TempEdgeInfo {
    EdgeIdx edge;
    NodeIdx upper_node;
    NodeIdx lower_node;
    std::optional<std::size_t> upper_flat;
    std::optional<std::size_t> lower_flat;
};

// =========================================================================
// Precomputed Edge Info (Type-2, 2D)
// =========================================================================

struct DataEdgeInfo2 {
    EdgeCoord2 edge;
    NodeIdx2 upper_node;
    NodeIdx2 lower_node;
    std::optional<std::size_t> upper_flat;
    std::optional<std::size_t> lower_flat;
};

// =========================================================================
// Type-1 Graph Topology (3D)
// =========================================================================

class GraphTopology {
public:
    HwDecoderParams params;
    std::vector<DataEdgeInfo> data_edges;
    std::vector<TempEdgeInfo> temp_edges;

    GraphTopology(std::size_t distance, std::size_t window_rounds);

    std::size_t total_edge_count() const;

    static NodeIdx data_edge_upper_node(uint8_t y, uint8_t x, uint8_t t, std::size_t d);
    static NodeIdx data_edge_lower_node(uint8_t y, uint8_t x, uint8_t t, std::size_t d);
    static NodeIdx temp_edge_upper_node(uint8_t t, uint8_t row, uint8_t col);
    static NodeIdx temp_edge_lower_node(uint8_t t, uint8_t row, uint8_t col,
                                        std::size_t window_rounds);

    const DataEdgeInfo& get_data_edge(std::size_t t, std::size_t y, std::size_t x) const;
    const TempEdgeInfo& get_temp_edge(std::size_t i, std::size_t row, std::size_t col) const;
    std::size_t data_edge_flat_idx(std::size_t t, std::size_t y, std::size_t x) const;
    std::size_t temp_edge_flat_idx(std::size_t i, std::size_t row, std::size_t col) const;

private:
    static std::vector<DataEdgeInfo> build_data_edges(const HwDecoderParams& params);
    static std::vector<TempEdgeInfo> build_temp_edges(const HwDecoderParams& params);
};

// =========================================================================
// Type-2 Graph Topology (2D)
// =========================================================================

class GraphTopology2 {
public:
    std::size_t distance;
    std::size_t m;
    std::size_t row_max;
    std::size_t col_max;
    std::vector<DataEdgeInfo2> data_edges;

    explicit GraphTopology2(std::size_t distance);

    static NodeIdx2 data_edge_upper_node(uint8_t y, uint8_t x, std::size_t d);
    static NodeIdx2 data_edge_lower_node(uint8_t y, uint8_t x, std::size_t d);

    const DataEdgeInfo2& get_data_edge(std::size_t y, std::size_t x) const;
    std::size_t data_edge_flat_idx(std::size_t y, std::size_t x) const;

private:
    static std::vector<DataEdgeInfo2> build_data_edges(std::size_t d, std::size_t col_max);
};

// =========================================================================
// Stim <-> SV Coordinate Conversion
// =========================================================================

std::vector<bool> fired_to_syndrome(const std::vector<std::size_t>& fired,
                                    const std::vector<Detector>& detectors,
                                    const HwDecoderParams& params, std::size_t window_start,
                                    std::size_t window_end);

NodeIdx stim_to_node(int64_t stim_x, int64_t stim_y, const HwDecoderParams& params);

// =========================================================================
// Edge flat-index (single source of truth)
//
// Maps a Type-1 EdgeIdx to its position in the flat edge-register array.
// Data edges occupy [0, window_rounds*d*d); temporal edges follow. Temporal
// edges store row in `x` and col in `y` (swapped, see EdgeIdx::temporal), so
// the swap lives here and nowhere else.
// =========================================================================

std::size_t edge_flat_index(const EdgeIdx& edge, const HwDecoderParams& params);

} // namespace qec
