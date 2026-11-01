#include "peeling.hpp"

namespace qec {

// =========================================================================
// Internal helpers
// =========================================================================

static bool is_node_on_left_seam(const NodeIdx& node, std::size_t core_start, bool has_left_seam) {
    return has_left_seam && node.has_value() && (static_cast<std::size_t>(node->t) == core_start);
}

static bool is_node_on_right_seam(const NodeIdx& node, std::size_t core_end, bool has_right_seam) {
    return has_right_seam && node.has_value() && (static_cast<std::size_t>(node->t) == core_end);
}

static const EdgeReg& get_edge_reg(const EdgeIdx& edge, const std::vector<EdgeReg>& edge_regs,
                                   const GraphTopology& graph) {
    return edge_regs[edge_flat_index(edge, graph.params)];
}

static OptEdgeIdx get_edge_chained_to(const OptEdgeIdx& edge, const std::vector<EdgeReg>& edge_regs,
                                      const std::vector<NodeReg>& node_regs,
                                      const GraphTopology& graph) {

    if (!edge.has_value()) {
        return std::nullopt;
    }

    const auto& reg = get_edge_reg(*edge, edge_regs, graph);

    if (reg.edge_chained_direction == CHAIN_DIR_FROM_UPPER) {
        const auto node_u = edge->is_temporal
                                ? GraphTopology::temp_edge_upper_node(edge->t, edge->x, edge->y)
                                : GraphTopology::data_edge_upper_node(edge->y, edge->x, edge->t,
                                                                      graph.params.distance);
        if (!node_u.has_value()) {
            return std::nullopt;
        }
        const auto flat = node_u->flat_index(graph.params.m, graph.params.col_max);
        return node_regs[flat].edge_merged_from;
    } else if (reg.edge_chained_direction == CHAIN_DIR_FROM_LOWER) {
        const auto node_l = edge->is_temporal
                                ? GraphTopology::temp_edge_lower_node(edge->t, edge->x, edge->y,
                                                                      graph.params.window_rounds)
                                : GraphTopology::data_edge_lower_node(edge->y, edge->x, edge->t,
                                                                      graph.params.distance);
        if (!node_l.has_value()) {
            return std::nullopt;
        }
        const auto flat = node_l->flat_index(graph.params.m, graph.params.col_max);
        return node_regs[flat].edge_merged_from;
    }

    return std::nullopt;
}

static void process_edge_t1(const OptEdgeIdx& edge, std::vector<std::vector<bool>>& corrections,
                            std::vector<std::vector<bool>>& left_seam,
                            std::vector<std::vector<bool>>& right_seam, const GraphTopology& graph,
                            std::size_t core_start, std::size_t core_end, bool has_left_seam,
                            bool has_right_seam) {

    if (!edge.has_value()) {
        return;
    }

    if (!is_edge_in_core(*edge, core_start, core_end)) {
        return;
    }

    if (edge->is_temporal) {
        const auto node_u = GraphTopology::temp_edge_upper_node(edge->t, edge->x, edge->y);
        const auto node_l = GraphTopology::temp_edge_lower_node(edge->t, edge->x, edge->y,
                                                                graph.params.window_rounds);

        if (is_node_on_left_seam(node_u, core_start, has_left_seam)) {
            left_seam[node_u->row][node_u->col] = !left_seam[node_u->row][node_u->col];
        }
        if (is_node_on_right_seam(node_l, core_end, has_right_seam)) {
            right_seam[node_l->row][node_l->col] = !right_seam[node_l->row][node_l->col];
        }
    } else {
        const auto node_u =
            GraphTopology::data_edge_upper_node(edge->y, edge->x, edge->t, graph.params.distance);
        const auto node_l =
            GraphTopology::data_edge_lower_node(edge->y, edge->x, edge->t, graph.params.distance);

        corrections[edge->y][edge->x] = !corrections[edge->y][edge->x];

        if (is_node_on_left_seam(node_u, core_start, has_left_seam)) {
            left_seam[node_u->row][node_u->col] = !left_seam[node_u->row][node_u->col];
        }
        if (is_node_on_left_seam(node_l, core_start, has_left_seam)) {
            left_seam[node_l->row][node_l->col] = !left_seam[node_l->row][node_l->col];
        }
    }
}

static void walk_chain_t1(const OptEdgeIdx& start_edge, const std::vector<EdgeReg>& edge_regs,
                          const std::vector<NodeReg>& node_regs,
                          std::vector<std::vector<bool>>& corrections,
                          std::vector<std::vector<bool>>& left_seam,
                          std::vector<std::vector<bool>>& right_seam, const GraphTopology& graph,
                          std::size_t core_start, std::size_t core_end, bool has_left_seam,
                          bool has_right_seam) {

    OptEdgeIdx current_edge = start_edge;

    while (true) {
        if (!current_edge.has_value()) {
            break;
        }

        process_edge_t1(current_edge, corrections, left_seam, right_seam, graph, core_start,
                        core_end, has_left_seam, has_right_seam);

        current_edge = get_edge_chained_to(current_edge, edge_regs, node_regs, graph);
    }
}

// =========================================================================
// Type-2 internal helpers
// =========================================================================

static EdgeIdx2 get_edge_chained_to_t2(const EdgeIdx2& edge, const std::vector<EdgeReg>& edge_regs,
                                       const std::vector<NodeReg2>& node_regs, std::size_t distance,
                                       std::size_t col_max) {

    if (!edge.has_value()) {
        return std::nullopt;
    }

    std::size_t idx =
        static_cast<std::size_t>(edge->y) * distance + static_cast<std::size_t>(edge->x);
    const auto& reg = edge_regs[idx];

    if (reg.edge_chained_direction == CHAIN_DIR_FROM_UPPER) {
        const auto node_u = GraphTopology2::data_edge_upper_node(edge->y, edge->x, distance);
        if (!node_u.has_value()) {
            return std::nullopt;
        }
        const auto flat = node_u->flat_index(col_max);
        return node_regs[flat].edge_merged_from;
    } else if (reg.edge_chained_direction == CHAIN_DIR_FROM_LOWER) {
        const auto node_l = GraphTopology2::data_edge_lower_node(edge->y, edge->x, distance);
        if (!node_l.has_value()) {
            return std::nullopt;
        }
        const auto flat = node_l->flat_index(col_max);
        return node_regs[flat].edge_merged_from;
    }

    return std::nullopt;
}

static void walk_chain_t2(const EdgeIdx2& start_edge, const std::vector<EdgeReg>& edge_regs,
                          const std::vector<NodeReg2>& node_regs,
                          std::vector<std::vector<bool>>& corrections, std::size_t distance,
                          std::size_t col_max) {

    EdgeIdx2 current_edge = start_edge;

    while (true) {
        if (!current_edge.has_value()) {
            break;
        }

        corrections[current_edge->y][current_edge->x] =
            !corrections[current_edge->y][current_edge->x];

        current_edge =
            get_edge_chained_to_t2(current_edge, edge_regs, node_regs, distance, col_max);
    }
}

// =========================================================================
// Public API
// =========================================================================

bool is_edge_in_core(const EdgeIdx& edge, std::size_t core_start, std::size_t core_end) {
    const auto t = static_cast<std::size_t>(edge.t);
    if (edge.is_temporal) {
        return t >= core_start + 1 && t <= core_end;
    } else {
        return t >= core_start && t + 1 <= core_end;
    }
}

PeelOutput run_peeling_t1(const std::vector<PeelEntry>& peel_entries,
                          const std::vector<EdgeReg>& edge_regs,
                          const std::vector<NodeReg>& node_regs, const GraphTopology& graph,
                          std::size_t core_start, std::size_t core_end, bool has_left_seam,
                          bool has_right_seam) {

    const auto d = graph.params.distance;
    const auto row_max = graph.params.row_max;
    const auto col_max = graph.params.col_max;

    std::vector<std::vector<bool>> corrections(d, std::vector<bool>(d, false));
    std::vector<std::vector<bool>> left_seam(row_max, std::vector<bool>(col_max, false));
    std::vector<std::vector<bool>> right_seam(row_max, std::vector<bool>(col_max, false));

    for (const auto& entry : peel_entries) {
        const OptEdgeIdx meeting_edge = entry.edge_id;
        process_edge_t1(meeting_edge, corrections, left_seam, right_seam, graph, core_start,
                        core_end, has_left_seam, has_right_seam);

        if (entry.node_u_edge_merged_from.has_value()) {
            walk_chain_t1(entry.node_u_edge_merged_from, edge_regs, node_regs, corrections,
                          left_seam, right_seam, graph, core_start, core_end, has_left_seam,
                          has_right_seam);
        }

        if (entry.node_l_edge_merged_from.has_value()) {
            walk_chain_t1(entry.node_l_edge_merged_from, edge_regs, node_regs, corrections,
                          left_seam, right_seam, graph, core_start, core_end, has_left_seam,
                          has_right_seam);
        }
    }

    return {.corrections = std::move(corrections),
            .left_seam = std::move(left_seam),
            .right_seam = std::move(right_seam)};
}

PeelOutput2 run_peeling_t2(const std::vector<PeelEntry2>& peel_entries,
                           const std::vector<EdgeReg>& edge_regs,
                           const std::vector<NodeReg2>& node_regs, std::size_t distance,
                           std::size_t col_max) {

    std::vector<std::vector<bool>> corrections(distance, std::vector<bool>(distance, false));

    for (const auto& entry : peel_entries) {
        corrections[entry.edge_id.y][entry.edge_id.x] =
            !corrections[entry.edge_id.y][entry.edge_id.x];

        if (entry.node_u_edge_merged_from.has_value()) {
            walk_chain_t2(entry.node_u_edge_merged_from, edge_regs, node_regs, corrections,
                          distance, col_max);
        }

        if (entry.node_l_edge_merged_from.has_value()) {
            walk_chain_t2(entry.node_l_edge_merged_from, edge_regs, node_regs, corrections,
                          distance, col_max);
        }
    }

    return PeelOutput2{std::move(corrections)};
}

} // namespace qec
