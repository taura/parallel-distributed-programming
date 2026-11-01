#include "fsm.hpp"

#include "merge.hpp"
#include "peeling.hpp"
#include "scanner.hpp"

namespace qec {

// =========================================================================
// Edge Growth Case Table
// =========================================================================

GrowthResult growth_case_table(bool node_u_odd, bool edge_u_state, bool edge_l_state,
                               bool node_l_odd) {

    const auto case_val = static_cast<uint8_t>(
        (static_cast<uint8_t>(node_u_odd) << 3) | (static_cast<uint8_t>(edge_u_state) << 2) |
        (static_cast<uint8_t>(edge_l_state) << 1) | static_cast<uint8_t>(node_l_odd));

    switch (case_val) {
    case 0b0001:
        return {.edge_u_state = false, .edge_l_state = true, .merge_required = false}; // L grows
    case 0b1000:
        return {.edge_u_state = true, .edge_l_state = false, .merge_required = false}; // U grows
    case 0b0011:                                                                       // L->U meet
    case 0b1100:                                                                       // U->L meet
    case 0b1001: // Both grow, meet
    case 0b1011: // Both grown, meet
    case 0b1101: // Both grown, meet
        return {.edge_u_state = true, .edge_l_state = true, .merge_required = true};
    default:
        return {.edge_u_state = edge_u_state,
                .edge_l_state = edge_l_state,
                .merge_required = false}; // No change
    }
}

EdgeGrowthResult evaluate_edge_growth(std::optional<std::size_t> upper_flat,
                                      std::optional<std::size_t> lower_flat, const EdgeReg& reg,
                                      const std::vector<NodeReg>& nodes, bool /*is_temporal*/) {

    std::optional<RootParity> node_u_parity;
    std::optional<RootParity> node_l_parity;

    if (upper_flat.has_value()) {
        node_u_parity = nodes[upper_flat.value()].root_parity;
    }
    if (lower_flat.has_value()) {
        node_l_parity = nodes[lower_flat.value()].root_parity;
    }

    const bool node_u_is_none = !upper_flat.has_value();
    const bool node_l_is_none = !lower_flat.has_value();

    const bool node_u_is_odd = !node_u_is_none && node_u_parity.has_value() &&
                               !node_u_parity->frozen && node_u_parity->odd;

    const bool node_l_is_odd = !node_l_is_none && node_l_parity.has_value() &&
                               !node_l_parity->frozen && node_l_parity->odd;

    const bool u_frozen = node_u_parity.has_value() && node_u_parity->frozen;
    const bool l_frozen = node_l_parity.has_value() && node_l_parity->frozen;

    if (u_frozen || l_frozen) {
        return {.edge_u_state = reg.edge_u_tree_state,
                .edge_l_state = reg.edge_l_tree_state,
                .merge_required = false,
                .chain_direction = CHAIN_DIR_NONE};
    }

    const auto growth = growth_case_table(node_u_is_odd, reg.edge_u_tree_state,
                                          reg.edge_l_tree_state, node_l_is_odd);

    uint8_t chain_dir = CHAIN_DIR_NONE;
    if (reg.edge_chained_direction == CHAIN_DIR_NONE) {
        if (node_u_is_odd && !node_l_is_odd) {
            chain_dir = CHAIN_DIR_FROM_UPPER;
        } else if (!node_u_is_odd && node_l_is_odd) {
            chain_dir = CHAIN_DIR_FROM_LOWER;
        }
    }

    return {.edge_u_state = growth.edge_u_state,
            .edge_l_state = growth.edge_l_state,
            .merge_required = growth.merge_required,
            .chain_direction = chain_dir};
}

// =========================================================================
// HwDecoderFsm (Type-1)
// =========================================================================

HwDecoderFsm::HwDecoderFsm(const GraphTopology& graph_in, std::size_t core_start,
                           std::size_t core_end, bool has_left_seam, bool has_right_seam)
    : graph(graph_in), core_start(core_start), core_end(core_end),
      has_left_seam(has_left_seam), has_right_seam(has_right_seam) {

    const auto d = graph.params.distance;
    const auto wr = graph.params.window_rounds;
    const auto m = graph.params.m;
    const auto rm = graph.params.row_max;
    const auto cm = graph.params.col_max;

    const auto node_count = wr * m;
    const auto data_edge_count = wr * d * d;
    const auto temp_edge_count = (wr + 1) * rm * cm;
    const auto total_edges = data_edge_count + temp_edge_count;

    nodes.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        nodes.emplace_back(i);
    }

    edges.resize(total_edges);

    corrections.assign(d, std::vector<bool>(d, false));
    left_seam.assign(rm, std::vector<bool>(cm, false));
    right_seam.assign(rm, std::vector<bool>(cm, false));

    grow_new_states_.reserve(total_edges);
    check_data_zdr_row_cols_ = d;
    check_data_zdr_row_.assign(wr * d, false);
    check_temp_zdr_col_cols_ = cm;
    check_temp_zdr_col_.assign((wr + 1) * cm, false);
}

void HwDecoderFsm::reset(const std::vector<bool>& /*syndrome*/, std::size_t new_core_start,
                          std::size_t new_core_end, bool new_has_left_seam,
                          bool new_has_right_seam) {
    core_start = new_core_start;
    core_end = new_core_end;
    has_left_seam = new_has_left_seam;
    has_right_seam = new_has_right_seam;
}

DecoderOutput HwDecoderFsm::run(const std::vector<bool>& syndrome) {
    init(syndrome);

    std::vector<PeelEntry> all_peel_entries;

    while (true) {
        const bool grown = grow();
        auto merge_queries = check();

        if (!merge_queries.empty()) {
            auto peel_entries =
                process_merge_queries<MergeQuery, NodeReg, PeelEntry>(merge_queries, nodes);
            all_peel_entries.insert(all_peel_entries.end(), peel_entries.begin(),
                                    peel_entries.end());

            if (grown) {
                continue;
            }
            break;
        } else {
            if (grown) {
                continue;
            }
            break;
        }
    }

    if (!all_peel_entries.empty()) {
        auto output = run_peeling_t1(all_peel_entries, edges, nodes, graph, core_start, core_end,
                                     has_left_seam, has_right_seam);
        corrections = std::move(output.corrections);
        left_seam = std::move(output.left_seam);
        right_seam = std::move(output.right_seam);
    }

    return {.corrections = corrections, .left_seam = left_seam, .right_seam = right_seam};
}

void HwDecoderFsm::init(const std::vector<bool>& syndrome) {
    const auto d = graph.params.distance;
    const auto rm = graph.params.row_max;
    const auto cm = graph.params.col_max;

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        nodes[i].initial_parity = (i < syndrome.size()) ? syndrome[i] : false;
        nodes[i].root_id = i;
        nodes[i].root_parity = RootParity(false, nodes[i].initial_parity);
        nodes[i].subtree_size = 1;
        nodes[i].edge_merged_from = std::nullopt;
        nodes[i].cluster_next = i;
    }

    for (auto& edge : edges) {
        edge = EdgeReg();
    }

    corrections.assign(d, std::vector<bool>(d, false));
    left_seam.assign(rm, std::vector<bool>(cm, false));
    right_seam.assign(rm, std::vector<bool>(cm, false));
}

bool HwDecoderFsm::grow() {
    grow_new_states_.clear();
    bool grown_flag = false;

    auto is_node_odd_active = [&](std::optional<std::size_t> flat) -> bool {
        if (!flat.has_value()) return false;
        const auto& rp = nodes[flat.value()].root_parity;
        return !rp.frozen && rp.odd;
    };

    auto evaluate = [&](const auto& info, bool is_temporal) {
        const auto& reg = edges[edge_flat_index(info.edge, graph.params)];

        if (!reg.edge_u_tree_state && !reg.edge_l_tree_state &&
            !is_node_odd_active(info.upper_flat) && !is_node_odd_active(info.lower_flat)) {
            grow_new_states_.push_back({false, false, false, CHAIN_DIR_NONE});
            return;
        }

        const auto result =
            evaluate_edge_growth(info.upper_flat, info.lower_flat, reg, nodes, is_temporal);

        if (result.edge_u_state != reg.edge_u_tree_state ||
            result.edge_l_state != reg.edge_l_tree_state) {
            grown_flag = true;
        }

        grow_new_states_.push_back({result.edge_u_state, result.edge_l_state,
                                    result.merge_required, result.chain_direction});
    };

    for (const auto& info : graph.data_edges) {
        evaluate(info, false);
    }
    for (const auto& info : graph.temp_edges) {
        evaluate(info, true);
    }

    auto latch = [&](const auto& info, std::size_t state_idx) {
        const auto& ns = grow_new_states_[state_idx];
        EdgeReg& reg = edges[edge_flat_index(info.edge, graph.params)];
        reg.edge_u_tree_state = ns.new_u;
        reg.edge_l_tree_state = ns.new_l;
        reg.merge_required = ns.merge_req;
        if (reg.edge_chained_direction == CHAIN_DIR_NONE && ns.chain_dir != CHAIN_DIR_NONE) {
            reg.edge_chained_direction = ns.chain_dir;
        }
    };

    for (std::size_t i = 0; i < graph.data_edges.size(); ++i) {
        latch(graph.data_edges[i], i);
    }
    const auto temp_offset = graph.data_edges.size();
    for (std::size_t i = 0; i < graph.temp_edges.size(); ++i) {
        latch(graph.temp_edges[i], temp_offset + i);
    }

    return grown_flag;
}

std::vector<MergeQuery> HwDecoderFsm::check() {
    const auto d = graph.params.distance;
    const auto wr = graph.params.window_rounds;
    const auto rm = graph.params.row_max;
    const auto cm = graph.params.col_max;

    std::fill(check_data_zdr_row_.begin(), check_data_zdr_row_.end(), false);
    for (const auto& info : graph.data_edges) {
        if (edges[edge_flat_index(info.edge, graph.params)].merge_required) {
            check_data_zdr_row_[static_cast<std::size_t>(info.edge.t) * d + info.edge.y] = true;
        }
    }

    std::fill(check_temp_zdr_col_.begin(), check_temp_zdr_col_.end(), false);
    for (const auto& info : graph.temp_edges) {
        bool mr = edges[edge_flat_index(info.edge, graph.params)].merge_required;

        if (mr) {
            const auto i = static_cast<std::size_t>(info.edge.t);
            if (i == 0 && !has_left_seam) {
                mr = false;
            }
            if (i == wr && !has_right_seam) {
                mr = false;
            }
        }

        if (mr) {
            check_temp_zdr_col_[static_cast<std::size_t>(info.edge.t) * cm + info.edge.y] = true;
        }
    }

    auto edge_merge_required = [&](const EdgeIdx& edge) -> bool {
        const bool mr = edges[edge_flat_index(edge, graph.params)].merge_required;
        if (mr && edge.is_temporal) {
            const auto i = static_cast<std::size_t>(edge.t);
            if (i == 0 && !has_left_seam) {
                return false;
            }
            if (i == wr && !has_right_seam) {
                return false;
            }
        }
        return mr;
    };

    MergeScanner scanner;
    auto merge_edges = scanner.scan_all_flat(check_data_zdr_row_, check_data_zdr_row_cols_,
                                             check_temp_zdr_col_, check_temp_zdr_col_cols_,
                                             edge_merge_required, wr, d, rm, cm);

    std::vector<MergeQuery> queries;
    for (const auto& edge : merge_edges) {
        std::size_t node_u_flat, node_l_flat;

        if (edge.is_temporal) {
            const auto& info = graph.get_temp_edge(static_cast<std::size_t>(edge.t),
                                                   static_cast<std::size_t>(edge.x),  // row
                                                   static_cast<std::size_t>(edge.y)); // col
            node_u_flat = info.upper_flat.value_or(SIZE_MAX);
            node_l_flat = info.lower_flat.value_or(SIZE_MAX);
        } else {
            const auto& info = graph.get_data_edge(static_cast<std::size_t>(edge.t),
                                                   static_cast<std::size_t>(edge.y),
                                                   static_cast<std::size_t>(edge.x));
            node_u_flat = info.upper_flat.value_or(SIZE_MAX);
            node_l_flat = info.lower_flat.value_or(SIZE_MAX);
        }

        queries.push_back(
            {.edge_id = edge, .node_u_flat = node_u_flat, .node_l_flat = node_l_flat});
    }

    return queries;
}

// =========================================================================
// HwDecoderFsm2 (Type-2)
// =========================================================================

HwDecoderFsm2::HwDecoderFsm2(const GraphTopology2& graph_in) : graph(graph_in) {

    const auto d = graph.distance;
    const auto node_count = graph.m;
    const auto edge_count = d * d;

    nodes.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
        nodes.emplace_back(i);
    }

    edges.resize(edge_count);
    corrections.assign(d, std::vector<bool>(d, false));
}

DecoderOutput2 HwDecoderFsm2::run_fsm(const std::vector<bool>& syndrome) {
    init(syndrome);

    std::vector<PeelEntry2> all_peel_entries;

    while (true) {
        const bool grown = grow();
        auto merge_queries = check();

        if (!merge_queries.empty()) {
            auto peel_entries =
                process_merge_queries<MergeQuery2, NodeReg2, PeelEntry2>(merge_queries, nodes);
            all_peel_entries.insert(all_peel_entries.end(), peel_entries.begin(),
                                    peel_entries.end());

            if (grown) {
                continue;
            }
            break;
        } else {
            if (grown) {
                continue;
            }
            break;
        }
    }

    if (!all_peel_entries.empty()) {
        auto output = run_peeling_t2(all_peel_entries, edges, nodes, graph.distance, graph.col_max);
        corrections = std::move(output.corrections);
    }

    return {.corrections = corrections};
}

void HwDecoderFsm2::init(const std::vector<bool>& syndrome) {
    const auto d = graph.distance;

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        nodes[i].initial_parity = (i < syndrome.size()) ? syndrome[i] : false;
        nodes[i].root_id = i;
        nodes[i].root_parity = RootParity(false, nodes[i].initial_parity);
        nodes[i].subtree_size = 1;
        nodes[i].edge_merged_from = std::nullopt;
        nodes[i].cluster_next = i;
    }

    for (auto& edge : edges) {
        edge = EdgeReg();
    }

    corrections.assign(d, std::vector<bool>(d, false));
}

bool HwDecoderFsm2::grow() {
    const auto d = graph.distance;
    bool grown_flag = false;

    struct NewState {
        bool new_u;
        bool new_l;
        bool merge_req;
        uint8_t chain_dir;
    };

    std::vector<NewState> new_states;
    new_states.reserve(edges.size());

    for (const auto& info : graph.data_edges) {
        const auto flat =
            static_cast<std::size_t>(info.edge.y) * d + static_cast<std::size_t>(info.edge.x);
        const auto& reg = edges[flat];

        std::optional<RootParity> node_u_parity;
        std::optional<RootParity> node_l_parity;

        if (info.upper_flat.has_value()) {
            node_u_parity = nodes[info.upper_flat.value()].root_parity;
        }
        if (info.lower_flat.has_value()) {
            node_l_parity = nodes[info.lower_flat.value()].root_parity;
        }

        const bool node_u_is_none = !info.upper_flat.has_value();
        const bool node_l_is_none = !info.lower_flat.has_value();

        const bool node_u_is_odd = !node_u_is_none && node_u_parity.has_value() &&
                                   !node_u_parity->frozen && node_u_parity->odd;

        const bool node_l_is_odd = !node_l_is_none && node_l_parity.has_value() &&
                                   !node_l_parity->frozen && node_l_parity->odd;

        const bool u_frozen = node_u_parity.has_value() && node_u_parity->frozen;
        const bool l_frozen = node_l_parity.has_value() && node_l_parity->frozen;

        if (u_frozen || l_frozen) {
            new_states.push_back(
                {reg.edge_u_tree_state, reg.edge_l_tree_state, false, CHAIN_DIR_NONE});
            continue;
        }

        const auto growth = growth_case_table(node_u_is_odd, reg.edge_u_tree_state,
                                              reg.edge_l_tree_state, node_l_is_odd);

        uint8_t chain_dir = CHAIN_DIR_NONE;
        if (reg.edge_chained_direction == CHAIN_DIR_NONE) {
            if (node_u_is_odd && !node_l_is_odd) {
                chain_dir = CHAIN_DIR_FROM_UPPER;
            } else if (!node_u_is_odd && node_l_is_odd) {
                chain_dir = CHAIN_DIR_FROM_LOWER;
            }
        }

        if (growth.edge_u_state != reg.edge_u_tree_state ||
            growth.edge_l_state != reg.edge_l_tree_state) {
            grown_flag = true;
        }

        new_states.push_back(
            {growth.edge_u_state, growth.edge_l_state, growth.merge_required, chain_dir});
    }

    // Latch
    for (std::size_t i = 0; i < graph.data_edges.size(); ++i) {
        const auto& info = graph.data_edges[i];
        const auto flat =
            static_cast<std::size_t>(info.edge.y) * d + static_cast<std::size_t>(info.edge.x);
        const auto& ns = new_states[i];
        EdgeReg& reg = edges[flat];
        reg.edge_u_tree_state = ns.new_u;
        reg.edge_l_tree_state = ns.new_l;
        reg.merge_required = ns.merge_req;
        if (reg.edge_chained_direction == CHAIN_DIR_NONE && ns.chain_dir != CHAIN_DIR_NONE) {
            reg.edge_chained_direction = ns.chain_dir;
        }
    }

    return grown_flag;
}

std::vector<MergeQuery2> HwDecoderFsm2::check() const {
    const auto d = graph.distance;

    std::vector<bool> zdr_row(d, false);
    std::vector<std::vector<bool>> merge_required_2d(d, std::vector<bool>(d, false));

    for (const auto& info : graph.data_edges) {
        const auto flat =
            static_cast<std::size_t>(info.edge.y) * d + static_cast<std::size_t>(info.edge.x);
        if (edges[flat].merge_required) {
            zdr_row[info.edge.y] = true;
            merge_required_2d[info.edge.y][info.edge.x] = true;
        }
    }

    MergeScanner2 scanner;
    auto merge_edges = scanner.scan_all(zdr_row, merge_required_2d, d);

    std::vector<MergeQuery2> queries;
    for (const auto& edge : merge_edges) {
        const auto& info =
            graph.get_data_edge(static_cast<std::size_t>(edge.y), static_cast<std::size_t>(edge.x));
        queries.push_back({.edge_id = edge,
                           .node_u_flat = info.upper_flat.value_or(SIZE_MAX),
                           .node_l_flat = info.lower_flat.value_or(SIZE_MAX)});
    }

    return queries;
}

} // namespace qec
