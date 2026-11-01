#pragma once

#include "graph.hpp"
#include "sv_types.hpp"

#include <cstddef>
#include <vector>

namespace qec {

struct PeelOutput {
    std::vector<std::vector<bool>> corrections;
    std::vector<std::vector<bool>> left_seam;
    std::vector<std::vector<bool>> right_seam;
};

struct PeelOutput2 {
    std::vector<std::vector<bool>> corrections;
};

PeelOutput run_peeling_t1(const std::vector<PeelEntry>& peel_entries,
                          const std::vector<EdgeReg>& edge_regs,
                          const std::vector<NodeReg>& node_regs, const GraphTopology& graph,
                          std::size_t core_start, std::size_t core_end, bool has_left_seam,
                          bool has_right_seam);

PeelOutput2 run_peeling_t2(const std::vector<PeelEntry2>& peel_entries,
                           const std::vector<EdgeReg>& edge_regs,
                           const std::vector<NodeReg2>& node_regs, std::size_t distance,
                           std::size_t col_max);

bool is_edge_in_core(const EdgeIdx& edge, std::size_t core_start, std::size_t core_end);

} // namespace qec
