// PD02: Serial Monte Carlo Trajectory Optimization (Cross-Entropy Method)
// Standard C++17 only. No external libraries or hardware dependencies.
// Output: trajectory.ppm (all iterations, color-coded), best_trajectory.csv, summary.txt
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr double PI = 3.14159265358979323846;
constexpr double WORLD_W = 12.0;
constexpr double WORLD_H = 9.0;
constexpr double ROBOT_R = 0.22;

struct State { double x, y, theta; };
struct Obstacle { double x, y, radius; };
struct Result {
    double cost = std::numeric_limits<double>::infinity();
    double final_distance = std::numeric_limits<double>::infinity();
    double min_clearance = std::numeric_limits<double>::infinity();
    int collision_steps = 0;
    std::vector<State> states;
};
struct Candidate { double cost; std::vector<double> controls; };
struct Settings {
    int samples = 1400;
    int steps = 90;
    int iterations = 7;
    int knots = 12;
    std::uint64_t seed = 42;
    double dt = 0.14;
    std::string output = "trajectory.ppm";
};

static const State START{1.0, 1.1, 0.45};
static const State GOAL{10.4, 7.3, 0.0};
static const std::vector<Obstacle> OBSTACLES{
    {3.3, 2.8, 0.83},
    {5.1, 4.45, 0.90},
    {6.5, 2.00, 0.78},
    {7.45, 6.85, 0.89},
    {9.25, 4.30, 0.69}
};

double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
double wrap(double a) {
    while (a > PI) a -= 2.0 * PI;
    while (a < -PI) a += 2.0 * PI;
    return a;
}
double dist(double x, double y, double a, double b) { return std::hypot(x - a, y - b); }

// Piecewise-linearly interpolate a compact sequence of steering corrections.
// Each sampled trajectory is independent: this loop can be parallelized later.
double correction_at(const std::vector<double>& controls, int step, int total_steps) {
    if (controls.size() == 1) return controls[0];
    double pos = double(step) * double(controls.size() - 1) / double(std::max(1, total_steps - 1));
    int k = std::min(int(pos), int(controls.size()) - 2);
    double fraction = pos - k;
    return controls[k] * (1.0 - fraction) + controls[k + 1] * fraction;
}

Result simulate(const Settings& s, const std::vector<double>& controls, bool keep_states) {
    Result out;
    State robot = START;
    if (keep_states) {
        out.states.reserve(static_cast<size_t>(s.steps) + 1);
        out.states.push_back(robot);
    }
    out.cost = 0.0;
    for (int t = 0; t < s.steps; ++t) {
        const double goal_dist = dist(robot.x, robot.y, GOAL.x, GOAL.y);
        // Feedback toward the goal; sampled low-frequency angular corrections
        // allow candidate trajectories to take alternative routes around obstacles.
        const double goal_angle = std::atan2(GOAL.y - robot.y, GOAL.x - robot.x);
        const double heading_error = wrap(goal_angle - robot.theta);
        const double angular_noise = correction_at(controls, t, s.steps);
        const double omega = clamp(2.3 * heading_error + angular_noise, -1.65, 1.65);
        const double speed = std::min(1.28, 1.65 * goal_dist);
        // Midpoint integration of the unicycle model.
        const double new_theta = wrap(robot.theta + omega * s.dt);
        robot.x += speed * s.dt * std::cos(robot.theta + omega * s.dt * 0.5);
        robot.y += speed * s.dt * std::sin(robot.theta + omega * s.dt * 0.5);
        robot.theta = new_theta;
        if (keep_states) out.states.push_back(robot);

        const double remaining = dist(robot.x, robot.y, GOAL.x, GOAL.y);
        out.cost += 0.40 * remaining * s.dt; // Prefer timely progress.
        out.cost += (0.18 * omega * omega + 0.06 * angular_noise * angular_noise) * s.dt;
        // Clearance is distance between robot boundary and obstacle boundary.
        for (const auto& ob : OBSTACLES) {
            const double clearance = dist(robot.x, robot.y, ob.x, ob.y) - ob.radius - ROBOT_R;
            out.min_clearance = std::min(out.min_clearance, clearance);
            if (clearance < 0.0) {
                ++out.collision_steps;
                out.cost += 2500.0 + 1500.0 * clearance * clearance;
            } else if (clearance < 0.75) {
                out.cost += 22.0 * (0.75 - clearance) * (0.75 - clearance) * s.dt;
            }
        }
        const double wall_clear = std::min({robot.x, WORLD_W - robot.x, robot.y, WORLD_H - robot.y}) - ROBOT_R;
        if (wall_clear < 0.0) out.cost += 4000.0 + 1000.0 * wall_clear * wall_clear;
        else if (wall_clear < 0.35) out.cost += 30.0 * (0.35 - wall_clear) * (0.35 - wall_clear);
    }
    out.final_distance = dist(robot.x, robot.y, GOAL.x, GOAL.y);
    out.cost += 180.0 * out.final_distance * out.final_distance;
    return out;
}

Settings parse_args(int argc, char** argv) {
    Settings s;
    auto positive = [](const char* text, const std::string& name) -> int {
        long long v = std::stoll(text);
        if (v < 1 || v > 2000000) throw std::invalid_argument(name + " must be in [1, 2000000]");
        return static_cast<int>(v);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./trajectory [--samples N] [--steps N] [--iterations N] "
                         "[--seed N] [--output FILE.ppm]\n"
                         "Defaults: --samples 1400 --steps 90 --iterations 7 --seed 42 "
                         "--output trajectory.ppm\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + a);
        const char* value = argv[++i];
        if (a == "--samples") s.samples = positive(value, a);
        else if (a == "--steps") s.steps = positive(value, a);
        else if (a == "--iterations") s.iterations = positive(value, a);
        else if (a == "--seed") s.seed = std::stoull(value);
        else if (a == "--output") s.output = value;
        else throw std::invalid_argument("Unknown option: " + a);
    }
    if (s.samples < 10) throw std::invalid_argument("--samples must be at least 10 for elite selection");
    if (s.iterations > 100) throw std::invalid_argument("--iterations must be <= 100");
    if (s.steps > 5000) throw std::invalid_argument("--steps must be <= 5000");
    if (s.output.empty()) throw std::invalid_argument("--output must not be empty");
    return s;
}

struct RGB { unsigned char r, g, b; };

// Every CEM iteration uses a distinct blue shade:
// earlier iterations are lighter, later iterations are darker.
RGB iteration_color(int iteration, int total_iterations) {
    const double t = (total_iterations <= 1)
        ? 0.5 : double(iteration) / (total_iterations - 1);
    auto mix = [t](int a, int b) {
        return static_cast<unsigned char>(std::lround(a * (1.0 - t) + b * t));
    };
    return {mix(174, 28), mix(214, 61), mix(247, 157)};
}

struct Canvas {
    int w, h;
    std::vector<RGB> p;
    Canvas(int width, int height) : w(width), h(height), p(static_cast<size_t>(w) * h, {250, 251, 254}) {}
    void pixel(int x, int y, RGB color, double alpha = 1.0) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        alpha = clamp(alpha, 0.0, 1.0);
        RGB& q = p[static_cast<size_t>(y) * w + x];
        q.r = static_cast<unsigned char>(q.r * (1.0 - alpha) + color.r * alpha);
        q.g = static_cast<unsigned char>(q.g * (1.0 - alpha) + color.g * alpha);
        q.b = static_cast<unsigned char>(q.b * (1.0 - alpha) + color.b * alpha);
    }
    void circle(double cx, double cy, double radius, RGB color, double alpha = 1.0) {
        const int xmin = std::max(0, int(std::floor(cx - radius)));
        const int xmax = std::min(w - 1, int(std::ceil(cx + radius)));
        const int ymin = std::max(0, int(std::floor(cy - radius)));
        const int ymax = std::min(h - 1, int(std::ceil(cy + radius)));
        for (int y = ymin; y <= ymax; ++y)
            for (int x = xmin; x <= xmax; ++x)
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius)
                    pixel(x, y, color, alpha);
    }
    void line(double x0, double y0, double x1, double y1, RGB color, double width, double alpha = 1.0) {
        const double length = std::hypot(x1 - x0, y1 - y0);
        const int n = std::max(1, int(length * 1.25));
        for (int i = 0; i <= n; ++i) {
            const double t = double(i) / n;
            circle(x0 * (1.0 - t) + x1 * t, y0 * (1.0 - t) + y1 * t, width * 0.5, color, alpha);
        }
    }
    // Filled rectangle and tiny bitmap digits for the iteration legend.
    void rectangle(int x0, int y0, int x1, int y1, RGB color) {
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                pixel(x, y, color);
    }
    void digit(int x0, int y0, int d, RGB color, int scale = 2) {
        static const char* glyph[10][7] = {
            {"111", "101", "101", "101", "101", "101", "111"},
            {"010", "110", "010", "010", "010", "010", "111"},
            {"111", "001", "001", "111", "100", "100", "111"},
            {"111", "001", "001", "111", "001", "001", "111"},
            {"101", "101", "101", "111", "001", "001", "001"},
            {"111", "100", "100", "111", "001", "001", "111"},
            {"111", "100", "100", "111", "101", "101", "111"},
            {"111", "001", "001", "001", "001", "001", "001"},
            {"111", "101", "101", "111", "101", "101", "111"},
            {"111", "101", "101", "111", "001", "001", "111"}
        };
        if (d < 0 || d > 9) return;
        for (int y = 0; y < 7; ++y)
            for (int x = 0; x < 3; ++x)
                if (glyph[d][y][x] == '1')
                    rectangle(x0 + x * scale, y0 + y * scale,
                              x0 + (x+1)*scale - 1,
                              y0 + (y+1)*scale - 1, color);
    }
    void number(int x0, int y0, int n, RGB color) {
        std::string digits = std::to_string(n);
        for (char ch : digits) {
            digit(x0, y0, ch - '0', color);
            x0 += 8;
        }
    }
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot write image: " + path);
        f << "P6\n" << w << " " << h << "\n255\n";
        for (const auto& c : p) {
            f.put(static_cast<char>(c.r));
            f.put(static_cast<char>(c.g));
            f.put(static_cast<char>(c.b));
        }
        if (!f) throw std::runtime_error("Error writing image: " + path);
    }
};

void draw(const std::string& path, const Result& best,
          const std::vector<std::vector<Result>>& samples_by_iteration) {
    Canvas c(1050, 795);
    const double s = 73.0;
    const double left = 84.0, bottom = 724.0;
    auto xpix = [&](double x) { return left + s * x; };
    auto ypix = [&](double y) { return bottom - s * y; };
    for (int i = 0; i <= int(WORLD_W); ++i)
        c.line(xpix(i), ypix(0), xpix(i), ypix(WORLD_H), {223, 229, 237}, 1.0);
    for (int i = 0; i <= int(WORLD_H); ++i)
        c.line(xpix(0), ypix(i), xpix(WORLD_W), ypix(i), {223, 229, 237}, 1.0);
    const RGB axis{100, 113, 136};
    for (int j = 0; j < 4; ++j) {
        const double px0 = (j == 0 || j == 3) ? xpix(0) : xpix(WORLD_W);
        const double py0 = (j == 0 || j == 1) ? ypix(0) : ypix(WORLD_H);
        const double px1 = (j == 0 || j == 3) ? xpix(WORLD_W) : xpix(0);
        const double py1 = (j == 0 || j == 1) ? ypix(WORLD_H) : ypix(0);
        if (j == 0 || j == 2) c.line(px0, py0, px1, py0, axis, 2);
        else c.line(px0, py0, px0, py1, axis, 2);
    }
    // Draw EVERY sampled trajectory in EVERY iteration, ordered from
    // lightest (early iteration) to darkest (late iteration).
    // Low per-line opacity reveals trajectory density and avoids saturation.
    for (size_t iter = 0; iter < samples_by_iteration.size(); ++iter) {
        const RGB shade = iteration_color(
            static_cast<int>(iter),
            static_cast<int>(samples_by_iteration.size()));
        for (const auto& path_result : samples_by_iteration[iter]) {
            for (size_t i = 1; i < path_result.states.size(); ++i) {
                const auto& a = path_result.states[i - 1];
                const auto& b = path_result.states[i];
                c.line(xpix(a.x), ypix(a.y), xpix(b.x), ypix(b.y),
                       shade, 1.2, 0.025);
            }
        }
    }
    for (const auto& ob : OBSTACLES) {
        // Light orange is robot-center forbidden region; dark red is physical obstacle.
        c.circle(xpix(ob.x), ypix(ob.y), s * (ob.radius + ROBOT_R), {254, 185, 111}, 0.48);
        c.circle(xpix(ob.x), ypix(ob.y), s * ob.radius, {203, 71, 77}, 0.86);
    }
    // Selected minimum-cost trajectory.
    for (size_t i = 1; i < best.states.size(); ++i) {
        const auto& a = best.states[i-1];
        const auto& b = best.states[i];
        c.line(xpix(a.x), ypix(a.y), xpix(b.x), ypix(b.y), {25, 112, 81}, 6.0, 0.96);
    }
    // Start: blue circle. Goal: gold crosshair. Finish: green disk.
    c.circle(xpix(START.x), ypix(START.y), 10.5, {37, 87, 213});
    c.circle(xpix(START.x), ypix(START.y), 4.0, {250, 251, 254});
    c.circle(xpix(GOAL.x), ypix(GOAL.y), 14.0, {242, 179, 45});
    c.circle(xpix(GOAL.x), ypix(GOAL.y), 7.0, {250, 251, 254});
    c.line(xpix(GOAL.x)-19, ypix(GOAL.y), xpix(GOAL.x)+19, ypix(GOAL.y), {158, 107, 13}, 2);
    c.line(xpix(GOAL.x), ypix(GOAL.y)-19, xpix(GOAL.x), ypix(GOAL.y)+19, {158, 107, 13}, 2);
    const auto& f = best.states.back();
    c.circle(xpix(f.x), ypix(f.y), 6.0, {20, 122, 71});

    // Numeric legend at the bottom. A swatch/number pair identifies the
    // CEM iteration. More than eight iterations: show representative indices.
    const int n_iterations = static_cast<int>(samples_by_iteration.size());
    const int n_legend = std::min(n_iterations, 8);
    for (int j = 0; j < n_legend; ++j) {
        const int idx = (n_legend == 1) ? 0 :
            j * (n_iterations - 1) / (n_legend - 1);
        const int x = 120 + j * 112;
        c.rectangle(x, 751, x + 66, 763,
                    iteration_color(idx, n_iterations));
        c.number(x + 26, 772, idx + 1, {58, 67, 80});
    }
    c.save(path);
}

int main(int argc, char** argv) {
    try {
        const Settings s = parse_args(argc, argv);
        auto tic = std::chrono::steady_clock::now();
        std::mt19937_64 rng(s.seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        std::vector<double> mean(s.knots, 0.0), sigma(s.knots, 0.80);
        std::vector<double> best_controls(s.knots, 0.0);
        Result best = simulate(s, best_controls, true);
        const double initial_cost = best.cost;
        // Retain the simulated trajectory of every sampled candidate,
        // grouped by CEM iteration, for the final visualization.
        std::vector<std::vector<Result>> samples_by_iteration(s.iterations);
        const int elite_count = std::max(2, s.samples / 10);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Serial Monte Carlo Trajectory Optimization (CEM)\n"
                  << "samples=" << s.samples << " steps=" << s.steps
                  << " iterations=" << s.iterations << " seed=" << s.seed << "\n"
                  << "initial nominal cost=" << initial_cost << "\n";
        for (int iter = 0; iter < s.iterations; ++iter) {
            std::vector<Candidate> population;
            population.reserve(s.samples);
            samples_by_iteration[iter].reserve(s.samples);
            for (int j = 0; j < s.samples; ++j) {
                std::vector<double> controls(s.knots);
                for (int k = 0; k < s.knots; ++k)
                    controls[k] = clamp(mean[k] + sigma[k] * gaussian(rng), -2.2, 2.2);
                // Save the complete path once. This is the same simulation
                // used for scoring, so it does not change the optimizer.
                Result score = simulate(s, controls, true);
                if (score.cost < best.cost) {
                    best_controls = controls;
                    best = score;
                }
                population.push_back({score.cost, std::move(controls)});
                samples_by_iteration[iter].push_back(std::move(score));
            }
            std::partial_sort(population.begin(), population.begin() + elite_count, population.end(),
                [](const Candidate& a, const Candidate& b){ return a.cost < b.cost; });
            for (int k = 0; k < s.knots; ++k) {
                double m = 0.0;
                for (int e = 0; e < elite_count; ++e) m += population[e].controls[k];
                m /= elite_count;
                double v = 0.0;
                for (int e = 0; e < elite_count; ++e) {
                    double d = population[e].controls[k] - m;
                    v += d * d;
                }
                v /= elite_count;
                // Smoothing helps avoid premature collapse and preserves exploration.
                mean[k] = 0.20 * mean[k] + 0.80 * m;
                sigma[k] = std::max(0.12, 0.20 * sigma[k] + 0.80 * std::sqrt(v + 0.01));
            }
            std::cout << "iteration " << (iter + 1) << "/" << s.iterations
                      << ": best_cost=" << best.cost
                      << ", goal_distance=" << best.final_distance
                      << ", collision_steps=" << best.collision_steps << "\n";
        }
        // Maintain reproducibility: best solution is evaluated exactly once more.
        best = simulate(s, best_controls, true);
        draw(s.output, best, samples_by_iteration);
        std::size_t visualized_trajectories = 0;
        for (const auto& group : samples_by_iteration)
            visualized_trajectories += group.size();
        std::ofstream csv("best_trajectory.csv");
        if (!csv) throw std::runtime_error("Cannot write best_trajectory.csv");
        csv << "step,time,x,y,theta\n" << std::setprecision(10);
        for (size_t i = 0; i < best.states.size(); ++i) {
            const auto& p = best.states[i];
            csv << i << ',' << i * s.dt << ',' << p.x << ',' << p.y << ',' << p.theta << '\n';
        }
        auto toc = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(toc - tic).count();
        std::ofstream report("summary.txt");
        if (!report) throw std::runtime_error("Cannot write summary.txt");
        report << std::fixed << std::setprecision(6)
               << "samples=" << s.samples << "\nsteps=" << s.steps << "\niterations=" << s.iterations
               << "\nseed=" << s.seed << "\ninitial_cost=" << initial_cost
               << "\nbest_cost=" << best.cost << "\nfinal_goal_distance=" << best.final_distance
               << "\nminimum_obstacle_clearance=" << best.min_clearance
               << "\ncollision_steps=" << best.collision_steps
               << "\ncomputation_seconds=" << elapsed
               << "\nvisualized_iterations=" << samples_by_iteration.size()
               << "\nvisualized_trajectories=" << visualized_trajectories << '\n';
        std::cout << "FINISHED\n"
                  << "initial_cost=" << initial_cost << "\n"
                  << "best_cost=" << best.cost << "\n"
                  << "final_goal_distance=" << best.final_distance << " m\n"
                  << "minimum_obstacle_clearance=" << best.min_clearance << " m\n"
                  << "collision_steps=" << best.collision_steps << "\n"
                  << "elapsed=" << elapsed << " s\n"
                  << "visualized_trajectories=" << visualized_trajectories << "\n"
                  << "Output files: " << s.output << ", best_trajectory.csv, summary.txt\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
