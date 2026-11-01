#pragma once

#include "graph.hpp"
#include "sv_types.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace qec {

// =========================================================================
// Edge Growth Results
// =========================================================================

struct GrowthResult {
    bool edge_u_state;
    bool edge_l_state;
    bool merge_required;
};

struct EdgeGrowthResult {
    bool edge_u_state;
    bool edge_l_state;
    bool merge_required;
    uint8_t chain_direction;
};

GrowthResult growth_case_table(bool node_u_odd, bool edge_u_state, bool edge_l_state,
                               bool node_l_odd);

EdgeGrowthResult evaluate_edge_growth(std::optional<std::size_t> upper_flat,
                                      std::optional<std::size_t> lower_flat, const EdgeReg& reg,
                                      const std::vector<NodeReg>& nodes, bool is_temporal);

// =========================================================================
// Type-1 Decoder FSM (3D)
// =========================================================================

class HwDecoderFsm {
public:
    const GraphTopology& graph;

    std::vector<NodeReg> nodes;
    std::vector<EdgeReg> edges;

    std::vector<std::vector<bool>> corrections;
    std::vector<std::vector<bool>> left_seam;
    std::vector<std::vector<bool>> right_seam;

    std::size_t core_start;
    std::size_t core_end;
    bool has_left_seam;
    bool has_right_seam;

    HwDecoderFsm(const GraphTopology& graph, std::size_t core_start, std::size_t core_end,
                 bool has_left_seam, bool has_right_seam);

    void reset(const std::vector<bool>& syndrome, std::size_t core_start, std::size_t core_end,
               bool has_left_seam, bool has_right_seam);

    DecoderOutput run(const std::vector<bool>& syndrome);

private:
    struct GrowNewState {
        bool new_u;
        bool new_l;
        bool merge_req;
        uint8_t chain_dir;
    };
    std::vector<GrowNewState> grow_new_states_;
    std::vector<bool> check_data_zdr_row_;
    std::size_t check_data_zdr_row_cols_;
    std::vector<bool> check_temp_zdr_col_;
    std::size_t check_temp_zdr_col_cols_;

    void init(const std::vector<bool>& syndrome);
    bool grow();
    [[nodiscard]] std::vector<MergeQuery> check();
};

// =========================================================================
// Type-2 Decoder FSM (2D)
// =========================================================================

class HwDecoderFsm2 {
public:
    const GraphTopology2& graph;

    std::vector<NodeReg2> nodes;
    std::vector<EdgeReg> edges;

    std::vector<std::vector<bool>> corrections;

    explicit HwDecoderFsm2(const GraphTopology2& graph);

    DecoderOutput2 run_fsm(const std::vector<bool>& syndrome);

private:
    void init(const std::vector<bool>& syndrome);
    bool grow();
    [[nodiscard]] std::vector<MergeQuery2> check() const;
};

} // namespace qec
