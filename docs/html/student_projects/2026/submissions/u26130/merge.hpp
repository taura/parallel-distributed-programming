#pragma once

#include "sv_types.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace qec {

// =========================================================================
// Templated merge operations — works for both Type-1 and Type-2
// =========================================================================

// Finds the root of `start` and records the traversed nodes into `path` (which
// is cleared first). `path` is caller-owned so it can be reused across calls,
// avoiding a heap allocation per find in the hot merge loop.
template <typename NodeRegT>
std::size_t find_root(const std::vector<NodeRegT>& nodes, std::size_t start,
                      std::vector<std::size_t>& path) {
    path.clear();
    std::size_t current = start;

    while (true) {
        const auto root_id = nodes[current].root_id;
        if (root_id == current) {
            return current;
        }
        path.push_back(current);
        current = root_id;
    }
}

namespace detail {

// Walk a circular cluster list starting at `head`, applying `fn` to each node.
template <typename NodeRegT, typename Fn>
void walk_cluster(std::vector<NodeRegT>& nodes, std::size_t head, Fn&& fn) {
    std::size_t cur = head;
    do {
        fn(nodes[cur]);
        cur = nodes[cur].cluster_next;
    } while (cur != head);
}

// Splice two circular cluster lists (a rooted at head_a, b at head_b).
// After splice, both are in a single circular list.
template <typename NodeRegT>
void splice_cluster_lists(std::vector<NodeRegT>& nodes, std::size_t head_a, std::size_t head_b) {
    // Swap next pointers of the two heads to merge circular lists.
    std::swap(nodes[head_a].cluster_next, nodes[head_b].cluster_next);
}

} // namespace detail

template <typename MergeQueryT, typename NodeRegT, typename PeelEntryT>
std::vector<PeelEntryT> process_merge_queries(const std::vector<MergeQueryT>& merge_queries,
                                              std::vector<NodeRegT>& nodes) {
    std::vector<PeelEntryT> peel_entries;

    // Reused across queries to avoid a per-find heap allocation.
    std::vector<std::size_t> path_a;
    std::vector<std::size_t> path_b;

    for (const auto& query : merge_queries) {
        const auto node_u_flat = query.node_u_flat;
        const auto node_l_flat = query.node_l_flat;

        const bool u_is_boundary = (node_u_flat == SIZE_MAX);
        const bool l_is_boundary = (node_l_flat == SIZE_MAX);

        if (u_is_boundary && l_is_boundary) {
            continue;
        }

        const RootParity boundary_parity(false, true);

        std::size_t root_a;
        path_a.clear();
        RootParity parity_a;
        uint32_t size_a;

        if (u_is_boundary) {
            root_a = SIZE_MAX;
            parity_a = boundary_parity;
            size_a = 0;
        } else {
            root_a = find_root(nodes, node_u_flat, path_a);
            parity_a = nodes[root_a].root_parity;
            size_a = nodes[root_a].subtree_size;
        }

        std::size_t root_b;
        path_b.clear();
        RootParity parity_b;
        uint32_t size_b;

        if (l_is_boundary) {
            root_b = SIZE_MAX - 1;
            parity_b = boundary_parity;
            size_b = 0;
        } else {
            root_b = find_root(nodes, node_l_flat, path_b);
            parity_b = nodes[root_b].root_parity;
            size_b = nodes[root_b].subtree_size;
        }

        const bool should_union = (root_a != root_b) && !parity_a.frozen && !parity_b.frozen &&
                                  (parity_a.odd || parity_b.odd);

        const bool clusters_met = should_union && parity_a.odd && parity_b.odd;

        if (should_union) {
            std::size_t union_root_a, union_root_b;
            uint32_t union_size_a, union_size_b;
            bool a_is_bdry, b_is_bdry;

            if (!parity_a.odd && parity_b.odd) {
                union_root_a = root_b;
                union_root_b = root_a;
                union_size_a = size_b;
                union_size_b = size_a;
                a_is_bdry = l_is_boundary;
                b_is_bdry = u_is_boundary;
            } else {
                union_root_a = root_a;
                union_root_b = root_b;
                union_size_a = size_a;
                union_size_b = size_b;
                a_is_bdry = u_is_boundary;
                b_is_bdry = l_is_boundary;
            }

            std::size_t new_root, loser;
            bool winner_is_bdry, loser_is_bdry;

            if (union_size_a >= union_size_b) {
                new_root = union_root_a;
                loser = union_root_b;
                winner_is_bdry = a_is_bdry;
                loser_is_bdry = b_is_bdry;
            } else {
                new_root = union_root_b;
                loser = union_root_a;
                winner_is_bdry = b_is_bdry;
                loser_is_bdry = a_is_bdry;
            }

            const auto new_size = union_size_a + union_size_b;
            const auto new_parity = RootParity::merge(parity_a, parity_b);

            if (loser_is_bdry) {
                if (!winner_is_bdry) {
                    detail::walk_cluster(nodes, new_root, [&](NodeRegT& node) {
                        node.root_parity = new_parity;
                    });
                    nodes[new_root].subtree_size = new_size;
                }
            } else if (winner_is_bdry) {
                detail::walk_cluster(nodes, loser, [&](NodeRegT& node) {
                    node.root_parity = new_parity;
                });
            } else {
                detail::walk_cluster(nodes, loser, [&](NodeRegT& node) {
                    node.root_id = new_root;
                    node.root_parity = new_parity;
                });
                detail::walk_cluster(nodes, new_root, [&](NodeRegT& node) {
                    node.root_parity = new_parity;
                });
                nodes[new_root].subtree_size = new_size;

                detail::splice_cluster_lists(nodes, new_root, loser);
            }

            if (clusters_met) {
                using OptEdge = decltype(NodeRegT::edge_merged_from);
                const OptEdge u_emf =
                    u_is_boundary ? OptEdge{std::nullopt} : nodes[node_u_flat].edge_merged_from;
                const OptEdge l_emf =
                    l_is_boundary ? OptEdge{std::nullopt} : nodes[node_l_flat].edge_merged_from;

                peel_entries.push_back(PeelEntryT{query.edge_id, u_emf, l_emf});
            } else if (!parity_a.odd && parity_b.odd) {
                if (!u_is_boundary) {
                    nodes[node_u_flat].edge_merged_from = query.edge_id;
                }
            } else if (parity_a.odd && !parity_b.odd) {
                if (!l_is_boundary) {
                    nodes[node_l_flat].edge_merged_from = query.edge_id;
                }
            }
        }

        if (!u_is_boundary) {
            const auto final_root_a = nodes[node_u_flat].root_id;
            const auto final_parity_a = nodes[final_root_a].root_parity;
            for (std::size_t node_flat : path_a) {
                nodes[node_flat].root_id = final_root_a;
                nodes[node_flat].root_parity = final_parity_a;
            }
        }

        if (!l_is_boundary) {
            const auto final_root_b = nodes[node_l_flat].root_id;
            const auto final_parity_b = nodes[final_root_b].root_parity;
            for (std::size_t node_flat : path_b) {
                nodes[node_flat].root_id = final_root_b;
                nodes[node_flat].root_parity = final_parity_b;
            }
        }
    }

    return peel_entries;
}

} // namespace qec
