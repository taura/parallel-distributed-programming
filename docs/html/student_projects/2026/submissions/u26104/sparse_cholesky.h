#ifndef PGO_SPARSE_CHOLESKY_H
#define PGO_SPARSE_CHOLESKY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// Serial simplicial, left-looking Cholesky. Lower triangle in scalar CSC.
// Natural ordering: the permutation is the identity. No dense n-by-n storage.
class SparseCholesky {
public:
    int n = 0;
    std::vector<std::size_t> col_ptr;
    std::vector<int> row_idx;
    std::vector<double> values;
    std::size_t input_nnz = 0;
    int failed_column = -1;

    // Each pair is (row, column), row >= column. Diagonals are implicit.
    // Eliminating a vertex connects all its remaining neighbors (fill-in).
    void analyze(int size, const std::vector<std::pair<int, int>>& lower,
                 std::size_t max_nnz) {
        n = size;
        if (n <= 0) throw std::runtime_error("Empty linear system");
        std::vector<std::set<int>> graph(n);
        std::size_t count = static_cast<std::size_t>(n);
        auto check_size = [&]() {
            if (count > max_nnz)
                throw std::runtime_error("Factor structure exceeds --max-factor-nnz; "
                                         "use a smaller graph or increase the limit");
        };
        check_size();
        for (const auto& rc : lower) {
            const int r = rc.first, c = rc.second;
            if (c < 0 || r < c || r >= n)
                throw std::runtime_error("Invalid lower-triangle coordinate");
            if (r != c && graph[c].insert(r).second) ++count;
            check_size();
        }
        input_nnz = count;
        col_ptr.assign(n + 1, 0);
        row_idx.clear();
        incoming.assign(n, {});
        for (int j = 0; j < n; ++j) {
            col_ptr[j] = row_idx.size();
            row_idx.push_back(j);
            std::vector<int> neighbors(graph[j].begin(), graph[j].end());
            for (int i : neighbors) {
                incoming[i].push_back({j, row_idx.size()});
                row_idx.push_back(i);
            }
            for (std::size_t a = 0; a < neighbors.size(); ++a) {
                for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                    if (graph[neighbors[a]].insert(neighbors[b]).second) {
                        ++count;
                        check_size();
                    }
                }
            }
            graph[j].clear();
        }
        col_ptr[n] = row_idx.size();
        values.assign(row_idx.size(), 0.0);
        work.resize(n);
    }

    std::size_t position(int row, int col) const {
        const auto first = row_idx.begin() + col_ptr[col];
        const auto last = row_idx.begin() + col_ptr[col + 1];
        const auto it = std::lower_bound(first, last, row);
        if (it == last || *it != row)
            throw std::runtime_error("Entry missing from symbolic factor");
        return static_cast<std::size_t>(it - row_idx.begin());
    }

    // Input has the factor's pattern; locations of fill entries contain zero.
    bool factorize(const std::vector<double>& matrix) {
        if (matrix.size() != values.size())
            throw std::runtime_error("Wrong numeric matrix size");
        failed_column = -1;
        for (int j = 0; j < n; ++j) {
            for (std::size_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p)
                work[row_idx[p]] = matrix[p];
            // incoming[j] lists earlier columns k with a structural L(j,k).
            for (const auto& entry : incoming[j]) {
                const int k = entry.first;
                const std::size_t jk = entry.second;
                const double ljk = values[jk];
                for (std::size_t p = jk; p < col_ptr[k + 1]; ++p)
                    work[row_idx[p]] -= values[p] * ljk;
            }
            if (!(work[j] > 0.0) || !std::isfinite(work[j])) {
                failed_column = j;
                return false;
            }
            const double diag = std::sqrt(work[j]);
            values[col_ptr[j]] = diag;
            for (std::size_t p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p) {
                values[p] = work[row_idx[p]] / diag;
                if (!std::isfinite(values[p])) {
                    failed_column = j;
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<double> solve(const std::vector<double>& rhs) const {
        if (rhs.size() != static_cast<std::size_t>(n))
            throw std::runtime_error("Wrong right-hand side size");
        std::vector<double> x = rhs;
        // Forward solve, column scatter.
        for (int j = 0; j < n; ++j) {
            x[j] /= values[col_ptr[j]];
            for (std::size_t p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p)
                x[row_idx[p]] -= values[p] * x[j];
        }
        // Backward solve with L transpose, without storing the transpose.
        for (int j = n - 1; j >= 0; --j) {
            for (std::size_t p = col_ptr[j] + 1; p < col_ptr[j + 1]; ++p)
                x[j] -= values[p] * x[row_idx[p]];
            x[j] /= values[col_ptr[j]];
        }
        return x;
    }

private:
    std::vector<std::vector<std::pair<int, std::size_t>>> incoming;
    std::vector<double> work;
};

#endif
