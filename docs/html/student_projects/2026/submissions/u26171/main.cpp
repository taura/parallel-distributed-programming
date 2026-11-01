#include "rpy.h"
#include "ppm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
struct Config {
    std::size_t n = 1000, cases = 1;
    std::uint32_t seed = 42;
    std::string output;
    std::string image;
    bool demo = false;
    bool check = false;
};

std::size_t positive_integer(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected a positive integer: " + text);
    const auto value = std::stoull(text);
    if (value == 0 || value > 1000000)
        throw std::invalid_argument("Integer must be between 1 and 1000000.");
    return static_cast<std::size_t>(value);
}

Config parse(int argc, char** argv) {
    Config c;
    bool n_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--check") { c.check = true; continue; }
        if (option == "--demo") { c.demo = true; continue; }
        if (i+1 == argc) throw std::invalid_argument("Missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--n") { c.n = positive_integer(value); n_given = true; }
        else if (option == "--cases") c.cases = positive_integer(value);
        else if (option == "--seed") c.seed = static_cast<std::uint32_t>(positive_integer(value));
        else if (option == "--output") c.output = value;
        else if (option == "--image") c.image = value;
        else throw std::invalid_argument("Unknown option: " + option);
    }
    if (c.demo && !n_given) c.n = 49;
    if (c.demo && c.cases != 1)
        throw std::invalid_argument("--demo uses one load case; omit --cases or use 1.");
    if (!c.image.empty() && c.image == c.output)
        throw std::invalid_argument("CSV and PPM output paths must differ.");
    return c;
}

std::vector<rpy::Vec3> make_demo_positions(std::size_t n) {
    std::size_t side = 1;
    while (side*side < n) ++side;
    std::vector<rpy::Vec3> x(n);
    for (std::size_t i=0; i<n; ++i)
        x[i] = {4.0*(i%side), 4.0*(i/side), 0.0};
    return x;
}

std::size_t central_particle(const std::vector<rpy::Vec3>& x) {
    double cx=0, cy=0;
    for (auto p : x) { cx+=p.x; cy+=p.y; }
    cx/=x.size(); cy/=x.size();
    std::size_t best=0;
    double distance=std::hypot(x[0].x-cx,x[0].y-cy);
    for (std::size_t i=1; i<x.size(); ++i) {
        const double d=std::hypot(x[i].x-cx,x[i].y-cy);
        if (d<distance) { best=i; distance=d; }
    }
    return best;
}

std::vector<rpy::Vec3> make_positions(std::size_t n, std::mt19937& rng) {
    std::size_t side = 1;
    while (side*side*side < n) ++side;
    std::uniform_real_distribution<double> jitter(-0.1, 0.1);
    std::vector<rpy::Vec3> x(n);
    // Radius = 1. Spacing = 4. Jitter preserves non-overlap.
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = {4.0*(i % side) + jitter(rng),
                4.0*((i/side) % side) + jitter(rng),
                4.0*(i/(side*side)) + jitter(rng)};
    }
    return x;
}

void write_csv(const std::string& path, const std::vector<rpy::Vec3>& x,
               const std::vector<rpy::Vec3>& f, const std::vector<rpy::Vec3>& u) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open output: " + path);
    out << "id,x,y,z,fx,fy,fz,ux,uy,uz\n" << std::setprecision(17);
    for (std::size_t i = 0; i < x.size(); ++i)
        out << i << ',' << x[i].x << ',' << x[i].y << ',' << x[i].z
            << ',' << f[i].x << ',' << f[i].y << ',' << f[i].z
            << ',' << u[i].x << ',' << u[i].y << ',' << u[i].z << '\n';
    out.close();
    if (!out) throw std::runtime_error("Failed writing output: " + path);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: RPYSolver [--check] [--n N] [--cases K] "
                         "[--seed S] [--output result.csv] [--image velocity.ppm] [--demo]\n"
                         "--check runs analytical checks and exits.\n"
                         "--demo uses a planar grid (default N=49), with one central particle forced along +X.\n"
                         "--image writes a P6 PPM of the last case's XY velocity projection.\n"
                         "Dimensionless inputs: radius=1, viscosity=1; no time integration.\n";
            return 0;
        }
        const Config c = parse(argc, argv);
        if (c.check) {
            rpy::run_self_checks();
            std::cout << "All analytical checks passed.\n";
            return 0;
        }
        const auto start = Clock::now();
        const rpy::Parameters p{};
        std::mt19937 rng(c.seed);
        auto x = c.demo ? make_demo_positions(c.n) : make_positions(c.n, rng);
        const std::size_t forced_particle = c.demo ? central_particle(x) : 0;
        std::vector<rpy::Vec3> f(c.n), u(c.n);
        const auto validation_start = Clock::now();
        rpy::validate_geometry(x, p); // Outside the kernel timer; O(N^2).
        const double validation_seconds =
            std::chrono::duration<double>(Clock::now()-validation_start).count();
        double kernel_seconds = 0.0, checksum = 0.0;
        std::uniform_real_distribution<double> force(-1.0, 1.0);
        for (std::size_t k = 0; k < c.cases; ++k) {
            // Case 0: uniform force. Later cases: different random loads on
            // the same geometry. Cases are not physical time steps.
            if (c.demo) {
                std::fill(f.begin(), f.end(), rpy::Vec3{});
                f[forced_particle] = {1,0,0};
            } else {
                for (auto& v : f)
                    v = k == 0 ? rpy::Vec3{1,0,0} : rpy::Vec3{force(rng),force(rng),force(rng)};
            }
            const auto t0 = Clock::now();
            rpy::apply_mobility_serial(x, f, p, u);
            kernel_seconds += std::chrono::duration<double>(Clock::now()-t0).count();
            for (const auto v : u) {
                if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                    throw std::runtime_error("Non-finite velocity.");
                checksum += v.x + 2*v.y + 3*v.z;
            }
        }
        rpy::Vec3 mean{};
        double power = 0.0;
        for (std::size_t i = 0; i < c.n; ++i) {
            mean.x += u[i].x; mean.y += u[i].y; mean.z += u[i].z;
            power += f[i].x*u[i].x + f[i].y*u[i].y + f[i].z*u[i].z;
        }
        if (!c.output.empty()) write_csv(c.output, x, f, u);
        const auto image_start = Clock::now();
        if (!c.image.empty()) rpy::write_velocity_ppm(c.image, x, f, u, c.demo);
        const double image_seconds = c.image.empty() ? 0.0 :
            std::chrono::duration<double>(Clock::now()-image_start).count();
        const double total_seconds = std::chrono::duration<double>(Clock::now()-start).count();
        std::cout << std::setprecision(10)
                  << "n=" << c.n << " cases=" << c.cases << " seed=" << c.seed << '\n'
                  << "validation_seconds=" << validation_seconds << '\n'
                  << "kernel_seconds=" << kernel_seconds << '\n'
                  << "image_seconds=" << image_seconds << '\n'
                  << "total_seconds=" << total_seconds << '\n'
                  << "directed_pair_evaluations=" << double(c.cases)*c.n*(c.n-1) << '\n'
                  << "checksum_all_cases=" << checksum << '\n'
                  << "last_case_mean_velocity=" << mean.x/c.n << ',' << mean.y/c.n << ',' << mean.z/c.n << '\n'
                  << "last_case_force_dot_velocity=" << power << '\n';
        if (c.demo) std::cout << "demo_forced_particle=" << forced_particle << '\n';
        if (!c.image.empty()) std::cout << "image=" << c.image << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
