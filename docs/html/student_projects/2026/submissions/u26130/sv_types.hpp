#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace qec {

// =========================================================================
// Node Coordinate (3D structured addressing) — matches node_idx_t
// =========================================================================

struct NodeCoord {
    uint8_t t;
    uint8_t row;
    uint8_t col;

    std::size_t flat_index(std::size_t m, std::size_t col_max) const {
        return static_cast<std::size_t>(t) * m + static_cast<std::size_t>(row) * col_max +
               static_cast<std::size_t>(col);
    }

    bool operator==(const NodeCoord& o) const {
        return t == o.t && row == o.row && col == o.col;
    }
};

using NodeIdx = std::optional<NodeCoord>;

// =========================================================================
// Edge Index Type (3D with temporal flag) — matches edge_idx_t
//
// EdgeIdx cannot use std::variant easily because the SV code stores
// temporal edges with swapped y/x fields, and both branches share the
// same field names throughout the codebase. Kept as a struct with an
// is_temporal discriminator.
// =========================================================================

struct EdgeIdx {
    bool is_temporal;
    uint8_t t;
    uint8_t y;
    uint8_t x;

    static EdgeIdx data(uint8_t t, uint8_t y, uint8_t x) {
        return EdgeIdx{false, t, y, x};
    }

    static EdgeIdx temporal(uint8_t t, uint8_t row, uint8_t col) {
        return EdgeIdx{true, t, col, row}; // swapped storage
    }

    bool operator==(const EdgeIdx& o) const {
        return is_temporal == o.is_temporal && t == o.t && y == o.y && x == o.x;
    }
};

using OptEdgeIdx = std::optional<EdgeIdx>;

// =========================================================================
// Type-2 Node Coordinate (2D: row × col, no time) — matches node_idx_t2
// =========================================================================

struct NodeCoord2 {
    uint8_t row;
    uint8_t col;

    std::size_t flat_index(std::size_t col_max) const {
        return static_cast<std::size_t>(row) * col_max + static_cast<std::size_t>(col);
    }

    bool operator==(const NodeCoord2& o) const {
        return row == o.row && col == o.col;
    }
};

using NodeIdx2 = std::optional<NodeCoord2>;

// =========================================================================
// Type-2 Edge Coordinate (2D: y × x, data edges only) — matches edge_idx_t2
// =========================================================================

struct EdgeCoord2 {
    uint8_t y;
    uint8_t x;

    std::size_t flat_index(std::size_t d) const {
        return static_cast<std::size_t>(y) * d + static_cast<std::size_t>(x);
    }

    bool operator==(const EdgeCoord2& o) const {
        return y == o.y && x == o.x;
    }
};

using EdgeIdx2 = std::optional<EdgeCoord2>;

// =========================================================================
// Root Parity (2 bits in SV: [1]=frozen, [0]=odd)
// =========================================================================

struct RootParity {
    bool frozen;
    bool odd;

    RootParity() : frozen(false), odd(false) {}
    RootParity(bool frozen, bool odd) : frozen(frozen), odd(odd) {}

    static RootParity merge(const RootParity& a, const RootParity& b) {
        const auto val_a = (static_cast<uint8_t>(a.frozen) << 1) | static_cast<uint8_t>(a.odd);
        const auto val_b = (static_cast<uint8_t>(b.frozen) << 1) | static_cast<uint8_t>(b.odd);
        const auto sum = static_cast<uint8_t>(val_a + val_b);
        return RootParity{((sum >> 1) & 1) != 0, (sum & 1) != 0};
    }

    bool operator==(const RootParity& o) const {
        return frozen == o.frozen && odd == o.odd;
    }
};

// =========================================================================
// FSM State — matches decoder_state_t
// =========================================================================

enum class DecoderState : uint8_t {
    Idle = 0,
    Init = 1,
    Grow = 2,
    Check = 3,
    Merge = 4,
    Peel = 5,
    Output = 6,
    Done = 7,
};

// =========================================================================
// Per-Node Register Set — matches node_ctrl_t1
// =========================================================================

struct NodeReg {
    bool initial_parity;
    std::size_t root_id;
    RootParity root_parity;
    uint32_t subtree_size;
    OptEdgeIdx edge_merged_from;
    std::size_t cluster_next;

    explicit NodeReg(std::size_t node_flat_idx)
        : initial_parity(false), root_id(node_flat_idx), root_parity(), subtree_size(1),
          edge_merged_from(std::nullopt), cluster_next(node_flat_idx) {}
};

// =========================================================================
// Per-Node Register Set (Type-2) — matches node_ctrl_t2
// =========================================================================

struct NodeReg2 {
    bool initial_parity;
    std::size_t root_id;
    RootParity root_parity;
    uint32_t subtree_size;
    EdgeIdx2 edge_merged_from;
    std::size_t cluster_next;

    explicit NodeReg2(std::size_t node_flat_idx)
        : initial_parity(false), root_id(node_flat_idx), root_parity(), subtree_size(1),
          edge_merged_from(std::nullopt), cluster_next(node_flat_idx) {}
};

// =========================================================================
// Edge Chaining Direction Constants (matching SV 2'b00/2'b01/2'b10)
// =========================================================================

inline constexpr uint8_t CHAIN_DIR_NONE = 0;
inline constexpr uint8_t CHAIN_DIR_FROM_LOWER = 1; // 2'b01
inline constexpr uint8_t CHAIN_DIR_FROM_UPPER = 2; // 2'b10

// =========================================================================
// Per-Edge Register Set — matches edge_ctrl_t1/t2
// =========================================================================

struct EdgeReg {
    bool edge_u_tree_state;
    bool edge_l_tree_state;
    bool merge_required;
    uint8_t edge_chained_direction;

    EdgeReg()
        : edge_u_tree_state(false), edge_l_tree_state(false), merge_required(false),
          edge_chained_direction(CHAIN_DIR_NONE) {}
};

// =========================================================================
// Merge Query Types — matches merge_query_t / merge_query_t2
// =========================================================================

struct MergeQuery {
    EdgeIdx edge_id;
    std::size_t node_u_flat;
    std::size_t node_l_flat;
};

struct MergeQuery2 {
    EdgeCoord2 edge_id;
    std::size_t node_u_flat;
    std::size_t node_l_flat;
};

// =========================================================================
// Peel Entry Types — for clusters_met FIFO
// =========================================================================

struct PeelEntry {
    EdgeIdx edge_id;
    OptEdgeIdx node_u_edge_merged_from;
    OptEdgeIdx node_l_edge_merged_from;
};

struct PeelEntry2 {
    EdgeCoord2 edge_id;
    EdgeIdx2 node_u_edge_merged_from;
    EdgeIdx2 node_l_edge_merged_from;
};

// =========================================================================
// Decoder Configuration
// =========================================================================

struct HwDecoderParams {
    std::size_t distance;
    std::size_t m;
    std::size_t row_max;
    std::size_t col_max;
    std::size_t window_rounds;
    std::size_t step_rounds;
    std::size_t buffer_rounds;
    std::size_t node_count;

    HwDecoderParams(std::size_t distance, std::size_t window_rounds)
        : distance(distance), m((distance + 1) * (distance - 1) / 2), row_max(distance + 1),
          col_max((distance - 1) / 2), window_rounds(window_rounds),
          step_rounds((distance + 1) / 2), buffer_rounds((distance + 1) / 2),
          node_count(window_rounds * m) {}

    static HwDecoderParams standard(std::size_t distance) {
        std::size_t step = (distance + 1) / 2;
        std::size_t wr = step + 2 * step;
        return HwDecoderParams(distance, wr);
    }
};

// =========================================================================
// Window Type — matches window_type_t
// =========================================================================

enum class WindowType : uint8_t {
    First = 0,
    Middle = 1,
    Last = 2,
};

// =========================================================================
// Decoder Output
// =========================================================================

struct DecoderOutput {
    std::vector<std::vector<bool>> corrections;
    std::optional<std::vector<std::vector<bool>>> left_seam;
    std::optional<std::vector<std::vector<bool>>> right_seam;
};

struct DecoderOutput2 {
    std::vector<std::vector<bool>> corrections;
};

} // namespace qec
