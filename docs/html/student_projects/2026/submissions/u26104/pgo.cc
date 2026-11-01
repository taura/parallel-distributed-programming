#include "sparse_cholesky.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>

using Clock = std::chrono::steady_clock;
constexpr double pi = 3.14159265358979323846;
constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double wrap(double a) { return std::remainder(a, 2.0 * pi); }
double norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

struct Pose { double x, y, theta; };
struct Edge {
    int i, j;
    Vec3 z, weight;
    bool loop;
    std::array<std::size_t, 4> block = {absent, absent, absent, absent};
};
struct Config {
    int poses_per_lap = 300, laps = 3, loop_neighbors = 2;
    int iterations = 20, max_trials = 10, min_gap = 0;
    std::uint32_t seed = 1;
    double scale = 20.0, loop_radius = 2.0, noise_scale = 1.0;
    double initial_lambda = 1e-4;
    std::size_t max_factor_nnz = 5000000;
    std::string image, csv;
    bool self_test = false;
};
struct Problem {
    std::vector<Pose> truth, initial;
    std::vector<Edge> edges;
};

// Relative pose of j expressed in i's local coordinate frame.
Vec3 relative(const Pose& i, const Pose& j) {
    const double c = std::cos(i.theta), s = std::sin(i.theta);
    const double dx = j.x - i.x, dy = j.y - i.y;
    return {c * dx + s * dy, -s * dx + c * dy, wrap(j.theta - i.theta)};
}
Pose compose(const Pose& p, const Vec3& z) {
    const double c = std::cos(p.theta), s = std::sin(p.theta);
    return {p.x + c * z[0] - s * z[1], p.y + s * z[0] + c * z[1],
            wrap(p.theta + z[2])};
}

// r = predicted relative pose - measured relative pose (wrapped angle).
Vec3 linearize(const Pose& i, const Pose& j, const Vec3& z, Mat3& a, Mat3& b) {
    const Vec3 q = relative(i, j);
    const double c = std::cos(i.theta), s = std::sin(i.theta);
    a = {-c, -s, q[1], s, -c, -q[0], 0, 0, -1};
    b = { c,  s,    0, -s, c,      0, 0, 0,  1};
    return {q[0] - z[0], q[1] - z[1], wrap(q[2] - z[2])};
}

Problem generate(const Config& cfg) {
    Problem p;
    const int n = cfg.poses_per_lap * cfg.laps + 1;
    p.truth.reserve(n);
    // Figure eight, nonzero tangent everywhere. Rotate to anchor heading zero.
    const double heading0 = std::atan2(1.3, 1.0);
    const double c0 = std::cos(heading0), s0 = std::sin(heading0);
    for (int k = 0; k < n; ++k) {
        const double t = 2.0 * pi * k / cfg.poses_per_lap;
        const double x = cfg.scale * std::sin(t);
        const double y = 0.65 * cfg.scale * std::sin(2.0 * t);
        p.truth.push_back({c0 * x + s0 * y, -s0 * x + c0 * y,
                          wrap(std::atan2(1.3 * std::cos(2.0 * t), std::cos(t))
                               - heading0)});
    }
    p.truth[0] = {0, 0, 0};
    std::mt19937 rng(cfg.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    auto add_edge = [&](int i, int j, bool loop) {
        Vec3 z = relative(p.truth[i], p.truth[j]);
        // Nominal uncertainties also define weights when noise_scale == 0.
        const Vec3 sigma = loop ? Vec3{0.03, 0.03, 0.01} : Vec3{0.02, 0.02, 0.005};
        Vec3 weight;
        for (int k = 0; k < 3; ++k) {
            z[k] += cfg.noise_scale * sigma[k] * normal(rng);
            weight[k] = 1.0 / (sigma[k] * sigma[k]);
        }
        z[2] = wrap(z[2]);
        p.edges.push_back({i, j, z, weight, loop});
    };
    p.initial.push_back(p.truth[0]);
    for (int j = 1; j < n; ++j) {
        add_edge(j - 1, j, false);
        p.initial.push_back(compose(p.initial.back(), p.edges.back().z));
    }

    // Spatial bins avoid an unconditional all-pairs scan. In crowded bins the
    // candidate count can still grow; input generation is timed separately.
    using Cell = std::pair<int, int>;
    std::map<Cell, std::vector<int>> bins;
    const int gap = cfg.min_gap > 0 ? cfg.min_gap : std::max(2, cfg.poses_per_lap / 4);
    if (cfg.loop_neighbors > 0) {
        for (int j = 0; j < n; ++j) {
            const auto& pose = p.truth[j];
            const int bx = static_cast<int>(std::floor(pose.x / cfg.loop_radius));
            const int by = static_cast<int>(std::floor(pose.y / cfg.loop_radius));
            std::vector<std::pair<double, int>> candidates;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    const auto it = bins.find({bx + dx, by + dy});
                    if (it == bins.end()) continue;
                    for (int i : it->second) {
                        if (j - i < gap || j - i <= 1) continue;
                        const double x = pose.x - p.truth[i].x;
                        const double y = pose.y - p.truth[i].y;
                        const double d2 = x * x + y * y;
                        if (d2 <= cfg.loop_radius * cfg.loop_radius)
                            candidates.push_back({d2, i});
                    }
                }
            }
            std::sort(candidates.begin(), candidates.end());
            const std::size_t take = std::min(candidates.size(),
                                            static_cast<std::size_t>(cfg.loop_neighbors));
            for (std::size_t k = 0; k < take; ++k) add_edge(candidates[k].second, j, true);
            bins[{bx, by}].push_back(j);
        }
    }
    return p;
}

// Full symmetric block CSR for assembly and residual checks. Pose zero is
// eliminated, so block row r corresponds to pose r+1. Dense 3x3 blocks, row-major.
struct BlockCSR {
    int blocks;
    std::vector<std::size_t> row_ptr;
    std::vector<int> col_idx;
    std::vector<Mat3> values;
    std::vector<std::size_t> diag;

    BlockCSR(int poses, std::vector<Edge>& edges) : blocks(poses - 1) {
        std::vector<std::set<int>> rows(blocks);
        for (int i = 0; i < blocks; ++i) rows[i].insert(i);
        for (const Edge& e : edges) {
            if (e.i != 0 && e.j != 0) {
                rows[e.i - 1].insert(e.j - 1);
                rows[e.j - 1].insert(e.i - 1);
            }
        }
        row_ptr.push_back(0);
        for (const auto& row : rows) {
            col_idx.insert(col_idx.end(), row.begin(), row.end());
            row_ptr.push_back(col_idx.size());
        }
        values.resize(col_idx.size());
        diag.resize(blocks);
        for (int i = 0; i < blocks; ++i) diag[i] = position(i, i);
        for (Edge& e : edges) {
            if (e.i != 0) e.block[0] = diag[e.i - 1];
            if (e.i != 0 && e.j != 0) {
                e.block[1] = position(e.i - 1, e.j - 1);
                e.block[2] = position(e.j - 1, e.i - 1);
            }
            if (e.j != 0) e.block[3] = diag[e.j - 1];
        }
    }
    std::size_t position(int r, int c) const {
        const auto begin = col_idx.begin() + row_ptr[r];
        const auto end = col_idx.begin() + row_ptr[r + 1];
        const auto it = std::lower_bound(begin, end, c);
        if (it == end || *it != c) throw std::runtime_error("Missing block");
        return static_cast<std::size_t>(it - col_idx.begin());
    }
    std::vector<double> multiply(const std::vector<double>& x) const {
        std::vector<double> y(3 * blocks, 0.0);
        for (int i = 0; i < blocks; ++i)
            for (std::size_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        y[3 * i + r] += values[p][3 * r + c] * x[3 * col_idx[p] + c];
        return y;
    }
};

double cost(const std::vector<Pose>& poses, const std::vector<Edge>& edges) {
    double sum = 0.0;
    for (const Edge& e : edges) {
        const Vec3 q = relative(poses[e.i], poses[e.j]);
        const Vec3 r = {q[0] - e.z[0], q[1] - e.z[1], wrap(q[2] - e.z[2])};
        for (int k = 0; k < 3; ++k) sum += 0.5 * e.weight[k] * r[k] * r[k];
    }
    return sum;
}

double assemble(const std::vector<Pose>& poses, const std::vector<Edge>& edges,
                BlockCSR& h, std::vector<double>& rhs) {
    for (auto& block : h.values) block.fill(0.0);
    rhs.assign(3 * h.blocks, 0.0);
    double sum = 0.0;
    for (const Edge& e : edges) {
        Mat3 a, b;
        const Vec3 r = linearize(poses[e.i], poses[e.j], e.z, a, b);
        for (int k = 0; k < 3; ++k) sum += 0.5 * e.weight[k] * r[k] * r[k];
        const Mat3* jac[2] = {&a, &b};
        const int ids[2] = {e.i, e.j};
        for (int u = 0; u < 2; ++u) {
            if (ids[u] == 0) continue;
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k)
                    rhs[3 * (ids[u] - 1) + c] -= (*jac[u])[3 * k + c] * e.weight[k] * r[k];
            for (int v = 0; v < 2; ++v) {
                if (ids[v] == 0) continue;
                Mat3& block = h.values[e.block[2 * u + v]];
                for (int rr = 0; rr < 3; ++rr)
                    for (int cc = 0; cc < 3; ++cc)
                        for (int k = 0; k < 3; ++k)
                            block[3 * rr + cc] += (*jac[u])[3 * k + rr] * e.weight[k]
                                                  * (*jac[v])[3 * k + cc];
            }
        }
    }
    return sum;
}

// Mapping from original blocks to the lower triangle of L's numeric workspace.
struct Mapping { std::size_t block; int element; std::size_t factor; };
std::vector<Mapping> prepare_factor(const BlockCSR& h, SparseCholesky& solver,
                                    std::size_t max_nnz) {
    std::vector<std::pair<int, int>> lower;
    for (int br = 0; br < h.blocks; ++br)
        for (std::size_t p = h.row_ptr[br]; p < h.row_ptr[br + 1]; ++p)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    const int row = 3 * br + r, col = 3 * h.col_idx[p] + c;
                    if (row >= col) lower.push_back({row, col});
                }
    solver.analyze(3 * h.blocks, lower, max_nnz);
    std::vector<Mapping> mapping;
    mapping.reserve(lower.size());
    for (int br = 0; br < h.blocks; ++br)
        for (std::size_t p = h.row_ptr[br]; p < h.row_ptr[br + 1]; ++p)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    const int row = 3 * br + r, col = 3 * h.col_idx[p] + c;
                    if (row >= col) mapping.push_back({p, 3 * r + c, solver.position(row, col)});
                }
    return mapping;
}

struct Timings {
    double structure = 0, symbolic = 0, assembly = 0, numeric_copy = 0;
    double factor = 0, solve = 0, linear_check = 0, candidate = 0, optimize = 0;
};
struct Result {
    std::vector<Pose> poses;
    double initial_cost = 0, final_cost = 0, max_linear_residual = 0;
    std::size_t a_nnz = 0, l_nnz = 0, block_nnz = 0;
    int accepted = 0, rejected = 0, factorizations = 0, factor_failures = 0;
    std::string status = "iteration_limit";
    Timings time;
};

Result optimize(Problem& problem, const Config& cfg, bool verbose = true) {
    Result out;
    out.poses = problem.initial;
    auto start = Clock::now();
    BlockCSR h(static_cast<int>(out.poses.size()), problem.edges);
    out.time.structure = elapsed(start);
    start = Clock::now();
    SparseCholesky solver;
    const auto mapping = prepare_factor(h, solver, cfg.max_factor_nnz);
    out.time.symbolic = elapsed(start);
    out.a_nnz = solver.input_nnz;
    out.l_nnz = solver.values.size();
    out.block_nnz = h.values.size();
    if (verbose)
        std::cout << "structure: unknowns=" << solver.n << " blocks=" << out.block_nnz
                  << " A_lower_nnz=" << out.a_nnz << " L_nnz=" << out.l_nnz
                  << " fill_ratio=" << double(out.l_nnz) / out.a_nnz << '\n';
    const auto optimize_start = Clock::now();
    out.initial_cost = cost(out.poses, problem.edges);
    out.final_cost = out.initial_cost;
    std::vector<double> rhs, diagonal(solver.n), matrix(solver.values.size());
    double lambda = cfg.initial_lambda;
    for (int iteration = 0; iteration < cfg.iterations; ++iteration) {
        start = Clock::now();
        const double current = assemble(out.poses, problem.edges, h, rhs);
        double scaled_gradient = 0.0;
        for (int i = 0; i < solver.n; ++i) {
            diagonal[i] = std::max(h.values[h.diag[i / 3]][4 * (i % 3)], 1e-12);
            scaled_gradient = std::max(scaled_gradient, std::abs(rhs[i]) / std::sqrt(diagonal[i]));
        }
        out.time.assembly += elapsed(start);
        if (!std::isfinite(current) || !std::isfinite(norm(rhs)))
            throw std::runtime_error("Non-finite objective or right-hand side");
        if (scaled_gradient < 1e-8) {
            out.status = "gradient_converged";
            break;
        }
        bool accepted = false, stop = false;
        for (int trial = 0; trial < cfg.max_trials; ++trial) {
            start = Clock::now();
            std::fill(matrix.begin(), matrix.end(), 0.0);
            for (const auto& m : mapping) matrix[m.factor] = h.values[m.block][m.element];
            for (int i = 0; i < solver.n; ++i)
                matrix[solver.col_ptr[i]] += lambda * diagonal[i];
            out.time.numeric_copy += elapsed(start);
            start = Clock::now();
            const bool ok = solver.factorize(matrix);
            out.time.factor += elapsed(start);
            ++out.factorizations;
            if (!ok) {
                ++out.factor_failures;
                ++out.rejected;
                if (verbose) std::cout << "  retry: factorization failed at column "
                                       << solver.failed_column << " lambda=" << lambda << '\n';
                lambda *= 10.0;
                continue;
            }
            start = Clock::now();
            const std::vector<double> step = solver.solve(rhs);
            out.time.solve += elapsed(start);
            start = Clock::now();
            std::vector<double> residual = h.multiply(step);
            for (int i = 0; i < solver.n; ++i)
                residual[i] += lambda * diagonal[i] * step[i] - rhs[i];
            const double rel = norm(residual) / std::max(norm(rhs), 1e-30);
            out.time.linear_check += elapsed(start);
            if (!std::isfinite(rel) || rel > 1e-6)
                throw std::runtime_error("Linear solve residual exceeds 1e-6");
            out.max_linear_residual = std::max(out.max_linear_residual, rel);
            start = Clock::now();
            std::vector<Pose> candidate = out.poses;
            double max_translation = 0.0, max_angle = 0.0;
            for (std::size_t i = 1; i < candidate.size(); ++i) {
                const std::size_t j = 3 * (i - 1);
                candidate[i].x += step[j];
                candidate[i].y += step[j + 1];
                candidate[i].theta = wrap(candidate[i].theta + step[j + 2]);
                max_translation = std::max(max_translation, std::hypot(step[j], step[j + 1]));
                max_angle = std::max(max_angle, std::abs(step[j + 2]));
            }
            const double next = cost(candidate, problem.edges);
            out.time.candidate += elapsed(start);
            if (std::isfinite(next) && next < current) {
                out.poses.swap(candidate);
                out.final_cost = next;
                ++out.accepted;
                accepted = true;
                if (verbose)
                    std::cout << "iter=" << iteration + 1 << " trial=" << trial + 1
                              << " cost=" << next << " lambda=" << lambda
                              << " linear_rel=" << rel << " step_m=" << max_translation
                              << " step_rad=" << max_angle << '\n';
                // Do not label tiny, heavily damped steps as convergence.
                if (lambda <= 1e-3 &&
                    (max_translation < 1e-7 && max_angle < 1e-8)) {
                    out.status = "step_converged";
                    stop = true;
                } else if (lambda <= 1e-3 &&
                           (current - next) / std::max(1.0, current) < 1e-10) {
                    out.status = "cost_converged";
                    stop = true;
                }
                lambda = std::max(1e-12, lambda / 3.0);
                break;
            }
            ++out.rejected;
            if (verbose) std::cout << "  retry: candidate_cost=" << next
                                   << " current_cost=" << current << " lambda=" << lambda << '\n';
            lambda *= 10.0;
        }
        if (!accepted) {
            out.status = "no_acceptable_step";
            break;
        }
        if (stop) break;
    }
    out.time.optimize = elapsed(optimize_start);
    return out;
}

std::pair<double, double> rmse(const std::vector<Pose>& x, const std::vector<Pose>& truth) {
    double pos = 0, angle = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i].x - truth[i].x, dy = x[i].y - truth[i].y;
        const double dt = wrap(x[i].theta - truth[i].theta);
        pos += dx * dx + dy * dy;
        angle += dt * dt;
    }
    return {std::sqrt(pos / x.size()), std::sqrt(angle / x.size())};
}

void write_csv(const std::string& filename, const Problem& p, const Result& r) {
    std::ofstream f(filename);
    if (!f) throw std::runtime_error("Cannot open CSV: " + filename);
    f << "index,true_x,true_y,true_theta,initial_x,initial_y,initial_theta,final_x,final_y,final_theta\n";
    f << std::setprecision(17);
    for (std::size_t i = 0; i < p.truth.size(); ++i) {
        f << i;
        for (const auto* poses : {&p.truth, &p.initial, &r.poses})
            f << ',' << (*poses)[i].x << ',' << (*poses)[i].y << ',' << (*poses)[i].theta;
        f << '\n';
    }
    if (!f) throw std::runtime_error("CSV write failed");
}

void write_image(const std::string& filename, const Problem& p, const Result& result) {
    constexpr int width = 1000, height = 800, margin = 35;
    std::vector<unsigned char> pixels(width * height * 3, 255);
    double xmin = p.truth[0].x, xmax = xmin, ymin = p.truth[0].y, ymax = ymin;
    for (const auto* poses : {&p.truth, &p.initial, &result.poses})
        for (const auto& pose : *poses) {
            xmin = std::min(xmin, pose.x); xmax = std::max(xmax, pose.x);
            ymin = std::min(ymin, pose.y); ymax = std::max(ymax, pose.y);
        }
    const double scale = std::min((width - 2.0 * margin) / std::max(xmax - xmin, 1e-9),
                                  (height - 2.0 * margin) / std::max(ymax - ymin, 1e-9));
    auto point = [&](const Pose& pose) {
        return std::pair<int, int>{static_cast<int>(width / 2.0 + (pose.x - (xmin + xmax) / 2) * scale),
                                  static_cast<int>(height / 2.0 - (pose.y - (ymin + ymax) / 2) * scale)};
    };
    using Color = std::array<unsigned char, 3>;
    auto pixel = [&](int x, int y, const Color& color) {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        for (int k = 0; k < 3; ++k) pixels[3 * (y * width + x) + k] = color[k];
    };
    auto line = [&](const Pose& a, const Pose& b, const Color& color, int radius) {
        const auto pa = point(a), pb = point(b);
        const int dx = pb.first - pa.first, dy = pb.second - pa.second;
        const int steps = std::max(1, std::max(std::abs(dx), std::abs(dy)));
        for (int k = 0; k <= steps; ++k) {
            const int x = pa.first + static_cast<int>(std::lround(double(dx) * k / steps));
            const int y = pa.second + static_cast<int>(std::lround(double(dy) * k / steps));
            for (int yy = -radius; yy <= radius; ++yy)
                for (int xx = -radius; xx <= radius; ++xx) pixel(x + xx, y + yy, color);
        }
    };
    auto path = [&](const std::vector<Pose>& poses, const Color& color, int radius) {
        for (std::size_t i = 1; i < poses.size(); ++i) line(poses[i - 1], poses[i], color, radius);
    };
    path(p.truth, {50, 160, 70}, 2);
    path(p.initial, {220, 80, 60}, 0);
    path(result.poses, {40, 70, 220}, 0);
    const auto anchor = point(p.truth[0]);
    for (int d = -6; d <= 6; ++d) {
        pixel(anchor.first + d, anchor.second, {0, 0, 0});
        pixel(anchor.first, anchor.second + d, {0, 0, 0});
    }
    std::ofstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PPM: " + filename);
    f << "P6\n# green=truth red=initial blue=optimized black=anchor\n"
      << width << ' ' << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (!f) throw std::runtime_error("PPM write failed");
}

// Dense reference used only by --self-test on tiny systems.
std::vector<double> dense_solve(std::vector<double> a, std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) a[j * n + j] -= a[j * n + k] * a[j * n + k];
        if (!(a[j * n + j] > 0)) throw std::runtime_error("Dense reference not SPD");
        a[j * n + j] = std::sqrt(a[j * n + j]);
        for (int i = j + 1; i < n; ++i) {
            for (int k = 0; k < j; ++k) a[i * n + j] -= a[i * n + k] * a[j * n + k];
            a[i * n + j] /= a[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) b[i] -= a[i * n + j] * b[j];
        b[i] /= a[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) b[i] -= a[j * n + i] * b[j];
        b[i] /= a[i * n + i];
    }
    return b;
}
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("Self-test failed: " + message);
}

void self_test() {
    // Finite differences cover orientations near angle wrap, but avoid the
    // residual's discontinuity at exactly +/-pi.
    double jac_error = 0.0;
    for (double theta : {-3.13, -0.7, 0.2, 3.13}) {
        const Pose p{1.2, -0.8, theta}, q{-2.1, 3.2, wrap(theta + 0.3)};
        const Vec3 z = relative(p, q);
        Mat3 a, b;
        linearize(p, q, z, a, b);
        for (int col = 0; col < 6; ++col) {
            Pose pp = p, pm = p, qp = q, qm = q;
            auto perturb = [](Pose& v, int k, double d) {
                if (k == 0) v.x += d;
                else if (k == 1) v.y += d;
                else v.theta += d;
            };
            const double eps = 1e-6;
            perturb(col < 3 ? pp : qp, col % 3, eps);
            perturb(col < 3 ? pm : qm, col % 3, -eps);
            Mat3 dummy_a, dummy_b;
            const Vec3 rp = linearize(pp, qp, z, dummy_a, dummy_b);
            const Vec3 rm = linearize(pm, qm, z, dummy_a, dummy_b);
            for (int row = 0; row < 3; ++row) {
                const double derivative = (rp[row] - rm[row]) / (2 * eps);
                const double analytic = (col < 3 ? a : b)[3 * row + col % 3];
                jac_error = std::max(jac_error, std::abs(derivative - analytic));
            }
        }
    }
    require(jac_error < 1e-7, "analytic Jacobian");
    std::cout << "PASS Jacobian finite differences: max_abs=" << jac_error << '\n';

    // SPD graph Laplacian + identity, with cycles generating real fill-in.
    constexpr int n = 18;
    std::vector<double> dense(n * n, 0.0);
    for (int i = 0; i < n; ++i) dense[i * n + i] = 1.0;
    for (int i = 0; i < n; ++i) {
        for (int offset : {1, 5}) {
            const int j = (i + offset) % n;
            const double w = 0.3 + 0.1 * (i % 4);
            dense[i * n + i] += w; dense[j * n + j] += w;
            dense[i * n + j] -= w; dense[j * n + i] -= w;
        }
    }
    std::vector<std::pair<int, int>> lower;
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            if (dense[i * n + j] != 0) lower.push_back({i, j});
    SparseCholesky solver;
    solver.analyze(n, lower, 10000);
    require(solver.values.size() > solver.input_nnz, "test must exercise fill-in");
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::vector<double> matrix(solver.values.size(), 0.0), ref = dense;
        for (int i = 0; i < n; ++i) ref[i * n + i] += repeat * 0.7;
        for (auto rc : lower) matrix[solver.position(rc.first, rc.second)] = ref[rc.first * n + rc.second];
        std::vector<double> exact(n), rhs(n, 0.0);
        for (int i = 0; i < n; ++i) exact[i] = std::sin(0.3 * i) + 0.2;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) rhs[i] += ref[i * n + j] * exact[j];
        require(solver.factorize(matrix), "sparse factorization");
        const auto x = solver.solve(rhs), xd = dense_solve(ref, rhs);
        double error = 0.0;
        for (int i = 0; i < n; ++i)
            error = std::max({error, std::abs(x[i] - exact[i]), std::abs(x[i] - xd[i])});
        require(error < 1e-11, "sparse/dense/known-solution comparison");
        // Reconstruct LL^T independently on this tiny matrix.
        std::vector<double> l(n * n, 0.0);
        for (int j = 0; j < n; ++j)
            for (std::size_t p = solver.col_ptr[j]; p < solver.col_ptr[j + 1]; ++p)
                l[solver.row_idx[p] * n + j] = solver.values[p];
        double factor_error = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double v = 0;
                for (int k = 0; k < n; ++k) v += l[i * n + k] * l[j * n + k];
                factor_error = std::max(factor_error, std::abs(v - ref[i * n + j]));
            }
        require(factor_error < 1e-11, "factor reconstruction");
        std::cout << "PASS sparse solve/reconstruction repeat=" << repeat
                  << " solution_error=" << error << " factor_error=" << factor_error << '\n';
    }
    std::vector<double> bad(solver.values.size(), 0.0);
    require(!solver.factorize(bad), "non-SPD detection");

    Config cfg;
    cfg.poses_per_lap = 32; cfg.laps = 2; cfg.scale = 3;
    cfg.loop_radius = 0.8; cfg.noise_scale = 0; cfg.iterations = 30;
    Problem p = generate(cfg);
    for (std::size_t i = 1; i < p.initial.size(); ++i) {
        p.initial[i].x += 0.05 * std::sin(double(i));
        p.initial[i].y += 0.04 * std::cos(0.7 * i);
        p.initial[i].theta = wrap(p.initial[i].theta + 0.02 * std::sin(0.9 * i));
    }
    // Check assembled PGO matrix/gradient against finite differences of cost.
    BlockCSR h(static_cast<int>(p.initial.size()), p.edges);
    std::vector<double> rhs;
    assemble(p.initial, p.edges, h, rhs);
    for (int k : {0, 1, 2, 27, 50, 101}) {
        auto plus = p.initial, minus = p.initial;
        const int i = k / 3 + 1, axis = k % 3;
        const double eps = 1e-6;
        if (axis == 0) { plus[i].x += eps; minus[i].x -= eps; }
        if (axis == 1) { plus[i].y += eps; minus[i].y -= eps; }
        if (axis == 2) { plus[i].theta += eps; minus[i].theta -= eps; }
        const double g = (cost(plus, p.edges) - cost(minus, p.edges)) / (2 * eps);
        require(std::abs(g + rhs[k]) < 1e-4, "assembled gradient");
    }
    SparseCholesky pg_solver;
    const auto mapping = prepare_factor(h, pg_solver, cfg.max_factor_nnz);
    const int m = pg_solver.n;
    std::vector<double> pg_dense(m * m, 0.0), numeric(pg_solver.values.size(), 0.0);
    for (int br = 0; br < h.blocks; ++br)
        for (std::size_t b = h.row_ptr[br]; b < h.row_ptr[br + 1]; ++b)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    pg_dense[(3 * br + r) * m + 3 * h.col_idx[b] + c] = h.values[b][3 * r + c];
    for (const auto& v : mapping) numeric[v.factor] = h.values[v.block][v.element];
    for (int i = 0; i < m; ++i) {
        pg_dense[i * m + i] += 1.0;
        numeric[pg_solver.col_ptr[i]] += 1.0;
        for (int j = 0; j < m; ++j)
            require(std::abs(pg_dense[i * m + j] - pg_dense[j * m + i]) < 1e-8,
                    "assembled matrix symmetry");
    }
    require(pg_solver.factorize(numeric), "PGO matrix factorization");
    const auto xs = pg_solver.solve(rhs), xd = dense_solve(pg_dense, rhs);
    double pg_error = 0;
    for (int i = 0; i < m; ++i) pg_error = std::max(pg_error, std::abs(xs[i] - xd[i]));
    require(pg_error < 1e-9, "PGO sparse/dense mapping");
    std::cout << "PASS PGO assembly and dense comparison: max_abs=" << pg_error << '\n';
    const Result r = optimize(p, cfg, false);
    const auto error = rmse(r.poses, p.truth);
    require(r.final_cost < 1e-10 && error.first < 1e-6 && error.second < 1e-6,
            "noise-free PGO recovery");
    require(r.poses[0].x == 0 && r.poses[0].y == 0 && r.poses[0].theta == 0, "fixed anchor");
    std::cout << "PASS noise-free PGO: cost=" << r.final_cost << " position_rmse=" << error.first << '\n';
    cfg.noise_scale = 1;
    p = generate(cfg);
    const auto initial_error = rmse(p.initial, p.truth);
    const Result noisy = optimize(p, cfg, false);
    const auto final_error = rmse(noisy.poses, p.truth);
    require(noisy.final_cost < noisy.initial_cost && final_error.first < initial_error.first,
            "seeded noisy trajectory improvement");
    require(noisy.status != "no_acceptable_step", "noisy convergence");
    std::cout << "PASS noisy PGO: cost=" << noisy.initial_cost << " -> " << noisy.final_cost
              << " position_rmse=" << initial_error.first << " -> " << final_error.first << '\n';
    std::cout << "All self-tests passed.\n";
}

void usage() {
    std::cout << "Serial 2D PGO with sparse Cholesky (natural ordering)\n"
              << "Usage: ./pgo [options]\n"
              << "  --poses-per-lap N    samples per lap (default 300, >= 8)\n"
              << "  --laps N             lap count (3); poses = samples * laps + 1\n"
              << "  --loop-neighbors N   max loop edges added per new pose (2; 0 disables)\n"
              << "  --loop-radius X      loop candidate radius in meters (2)\n"
              << "  --min-gap N          minimum index separation (0 = samples/4, at least 2)\n"
              << "  --scale X            trajectory x amplitude in meters (20)\n"
              << "  --noise-scale X      multiplier of generated noise (1; 0 = exact observations)\n"
              << "  --iterations N       maximum nonlinear iterations (20)\n"
              << "  --max-trials N       maximum damping trials per iteration (10)\n"
              << "  --lambda X           initial dimensionless damping (1e-4)\n"
              << "  --seed N             random seed (1)\n"
              << "  --max-factor-nnz N   factor structure size limit (5000000)\n"
              << "  --image PATH         write P6 PPM (green truth/red initial/blue final)\n"
              << "  --csv PATH           write trajectory coordinates\n"
              << "  --self-test          run small verification cases only\n"
              << "  --help               show this message\n";
}
Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--self-test") { c.self_test = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value: " + key);
        const std::string value = argv[++i];
        if (key == "--image") { c.image = value; continue; }
        if (key == "--csv") { c.csv = value; continue; }
        std::size_t end = 0;
        const double v = std::stod(value, &end);
        if (end != value.size() || !std::isfinite(v)) throw std::runtime_error("Invalid value: " + key);
        auto integer = [&]() {
            if (v < 0 || v > 1000000000 || std::floor(v) != v)
                throw std::runtime_error("Expected nonnegative integer: " + key);
            return static_cast<int>(v);
        };
        if (key == "--poses-per-lap") c.poses_per_lap = integer();
        else if (key == "--laps") c.laps = integer();
        else if (key == "--loop-neighbors") c.loop_neighbors = integer();
        else if (key == "--iterations") c.iterations = integer();
        else if (key == "--max-trials") c.max_trials = integer();
        else if (key == "--min-gap") c.min_gap = integer();
        else if (key == "--seed") c.seed = static_cast<std::uint32_t>(integer());
        else if (key == "--max-factor-nnz") c.max_factor_nnz = static_cast<std::size_t>(integer());
        else if (key == "--scale") c.scale = v;
        else if (key == "--loop-radius") c.loop_radius = v;
        else if (key == "--noise-scale") c.noise_scale = v;
        else if (key == "--lambda") c.initial_lambda = v;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (c.poses_per_lap < 8 || c.laps < 1 ||
        static_cast<long long>(c.poses_per_lap) * c.laps > 100000 ||
        c.iterations < 1 || c.max_trials < 1 || c.scale < 1e-6 || c.scale > 1e6 ||
        c.loop_radius < 1e-6 || c.scale / c.loop_radius > 1e8 ||
        c.noise_scale < 0 || c.noise_scale > 100 ||
        c.initial_lambda < 1e-12 || c.initial_lambda > 1e12 || c.max_factor_nnz < 1)
        throw std::runtime_error("Parameters outside supported range; see --help "
                                 "(at most 100000 intervals)");
    return c;
}

int main(int argc, char** argv) {
    try {
        std::cout << std::scientific << std::setprecision(6);
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--help") { usage(); return 0; }
        const Config cfg = parse(argc, argv);
        if (cfg.self_test) { self_test(); return 0; }
        const auto start = Clock::now();
        Problem p = generate(cfg);
        const double generation = elapsed(start);
        std::size_t loops = 0;
        for (const Edge& e : p.edges) if (e.loop) ++loops;
        std::cout << "problem: poses=" << p.truth.size() << " edges=" << p.edges.size()
                  << " odometry=" << p.edges.size() - loops << " loops=" << loops
                  << " seed=" << cfg.seed << " noise_scale=" << cfg.noise_scale << '\n';
        std::cout << "config: samples_per_lap=" << cfg.poses_per_lap << " laps=" << cfg.laps
                  << " scale_m=" << cfg.scale << " loop_radius_m=" << cfg.loop_radius
                  << " loop_neighbors=" << cfg.loop_neighbors
                  << " min_gap=" << (cfg.min_gap > 0 ? cfg.min_gap : std::max(2, cfg.poses_per_lap / 4))
                  << " max_iterations=" << cfg.iterations << " initial_lambda=" << cfg.initial_lambda << '\n';
        const Result r = optimize(p, cfg);
        const auto before = rmse(p.initial, p.truth), after = rmse(r.poses, p.truth);
        std::cout << "result: status=" << r.status << " accepted=" << r.accepted
                  << " rejected=" << r.rejected << " factorizations=" << r.factorizations
                  << " factor_failures=" << r.factor_failures << '\n';
        std::cout << "cost: " << r.initial_cost << " -> " << r.final_cost << '\n'
                  << "position_rmse_m: " << before.first << " -> " << after.first << '\n'
                  << "angle_rmse_rad: " << before.second << " -> " << after.second << '\n'
                  << "max_linear_relative_residual: " << r.max_linear_residual << '\n';
        const Timings& t = r.time;
        std::cout << "time_seconds:\n"
                  << "  input_generation=" << generation << '\n'
                  << "  block_structure=" << t.structure << '\n'
                  << "  symbolic_and_mapping=" << t.symbolic << '\n'
                  << "  assembly=" << t.assembly << '\n'
                  << "  numeric_copy_and_damping=" << t.numeric_copy << '\n'
                  << "  factorization=" << t.factor << '\n'
                  << "  triangular_solve=" << t.solve << '\n'
                  << "  linear_residual_check=" << t.linear_check << '\n'
                  << "  candidate_update_and_cost=" << t.candidate << '\n'
                  << "  optimization_total=" << t.optimize << '\n'
                  << "  backend_with_setup=" << t.structure + t.symbolic + t.optimize << '\n';
        if (!cfg.image.empty()) {
            write_image(cfg.image, p, r);
            std::cout << "image: " << cfg.image << " (green=truth red=initial blue=optimized)\n";
        }
        if (!cfg.csv.empty()) { write_csv(cfg.csv, p, r); std::cout << "csv: " << cfg.csv << '\n'; }
        return r.status == "no_acceptable_step" ? 2 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
