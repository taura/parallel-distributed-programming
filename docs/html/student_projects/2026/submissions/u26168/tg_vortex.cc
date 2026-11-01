// Serial velocity-pressure Navier--Stokes solver for a periodic Taylor--Green vortex.
// Build: clang++ -O3 -Wall -Wextra -std=c++17 tg_vortex.cc -o tg_vortex
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Grid = std::vector<double>;

struct Parameters {
    int n = 64;
    int steps = 100;
    double dt = 0.01;
    double nu = 0.1;
    int max_cg = 1000;
    double cg_tol = 1e-10;
};

// All fields have N*N elements, laid out as [j*N+i], with periodic indexing.
// p(i,j) at ((i+1/2)h,(j+1/2)h); u(i,j) at (ih,(j+1/2)h);
// v(i,j) at ((i+1/2)h,jh). Each face appears exactly once.
struct Mesh {
    int n;
    double h;
    double inv_h;
    double inv_h2;

    int wrap(int a) const { return (a + n) % n; }
    std::size_t index(int i, int j) const {
        return static_cast<std::size_t>(wrap(j)) * n + wrap(i);
    }
    double at(const Grid& f, int i, int j) const { return f[index(i, j)]; }
};

int parse_int(const char* text) {
    std::size_t end = 0;
    const std::string value(text);
    int result = std::stoi(value, &end);
    if (end != value.size()) throw std::invalid_argument("invalid integer: " + value);
    return result;
}

double parse_double(const char* text) {
    std::size_t end = 0;
    const std::string value(text);
    double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result))
        throw std::invalid_argument("invalid real number: " + value);
    return result;
}

void initialize(const Mesh& m, Grid& u, Grid& v, Grid& pressure) {
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i) {
            const double x = i * m.h, y = j * m.h;
            u[m.index(i, j)] = std::sin(x) * std::cos(y + 0.5 * m.h);
            v[m.index(i, j)] = -std::cos(x + 0.5 * m.h) * std::sin(y);
            pressure[m.index(i, j)] = 0.25 *
                (std::cos(2.0 * (x + 0.5 * m.h)) +
                 std::cos(2.0 * (y + 0.5 * m.h)));
        }
    }
}

// Conservative, centered fluxes approximate div(u*u) and div(u*v) at u faces,
// and div(u*v) and div(v*v) at v faces. Explicit Euler for advection/diffusion.
void predict_velocity(const Mesh& m, const Grid& u, const Grid& v,
                      Grid& u_star, Grid& v_star, double dt, double nu) {
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i) {
            const auto a = [&m](const Grid& f, int x, int y) {
                return m.at(f, x, y);
            };
            const std::size_t k = m.index(i, j);
            const double ue = 0.5 * (u[k] + a(u, i + 1, j));
            const double uw = 0.5 * (a(u, i - 1, j) + u[k]);
            const double un = 0.5 * (u[k] + a(u, i, j + 1));
            const double us = 0.5 * (a(u, i, j - 1) + u[k]);
            const double vn = 0.5 * (a(v, i - 1, j + 1) + a(v, i, j + 1));
            const double vs = 0.5 * (a(v, i - 1, j) + a(v, i, j));
            const double adv_u = (ue * ue - uw * uw + un * vn - us * vs) * m.inv_h;
            const double lap_u = (a(u, i + 1, j) + a(u, i - 1, j) +
                                  a(u, i, j + 1) + a(u, i, j - 1) - 4.0 * u[k]) * m.inv_h2;
            u_star[k] = u[k] + dt * (-adv_u + nu * lap_u);

            const double ve = 0.5 * (v[k] + a(v, i + 1, j));
            const double vw = 0.5 * (a(v, i - 1, j) + v[k]);
            const double ur = 0.5 * (a(u, i + 1, j - 1) + a(u, i + 1, j));
            const double ul = 0.5 * (a(u, i, j - 1) + a(u, i, j));
            const double vt = 0.5 * (v[k] + a(v, i, j + 1));
            const double vb = 0.5 * (a(v, i, j - 1) + v[k]);
            const double adv_v = (ur * ve - ul * vw + vt * vt - vb * vb) * m.inv_h;
            const double lap_v = (a(v, i + 1, j) + a(v, i - 1, j) +
                                  a(v, i, j + 1) + a(v, i, j - 1) - 4.0 * v[k]) * m.inv_h2;
            v_star[k] = v[k] + dt * (-adv_v + nu * lap_v);
        }
    }
}

// Cell-centered MAC divergence. Also used on the predicted velocity.
void divergence(const Mesh& m, const Grid& u, const Grid& v, Grid& div) {
    for (int j = 0; j < m.n; ++j)
        for (int i = 0; i < m.n; ++i)
            div[m.index(i, j)] = (m.at(u, i + 1, j) - m.at(u, i, j) +
                                  m.at(v, i, j + 1) - m.at(v, i, j)) * m.inv_h;
}

// A = -Delta_h. It is positive definite on the zero-mean subspace.
void negative_laplacian(const Mesh& m, const Grid& input, Grid& output) {
    for (int j = 0; j < m.n; ++j)
        for (int i = 0; i < m.n; ++i) {
            const std::size_t k = m.index(i, j);
            output[k] = (4.0 * input[k] - m.at(input, i - 1, j) -
                         m.at(input, i + 1, j) - m.at(input, i, j - 1) -
                         m.at(input, i, j + 1)) * m.inv_h2;
        }
}

// Solve A*p = -div(u*)/dt, warm-starting from the preceding pressure.
int solve_pressure(const Mesh& m, const Grid& rhs, Grid& pressure,
                   Grid& r, Grid& direction, Grid& ap,
                   int max_cg, double tolerance) {
    negative_laplacian(m, pressure, ap);
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t k = 0; k < rhs.size(); ++k) {
        r[k] = rhs[k] - ap[k];
        direction[k] = r[k];
        rr += r[k] * r[k];
        bb += rhs[k] * rhs[k];
    }
    if (bb == 0.0) {
        std::fill(pressure.begin(), pressure.end(), 0.0);
        return 0;
    }
    const double target = tolerance * tolerance * bb;
    if (rr <= target) return 0;

    for (int iter = 1; iter <= max_cg; ++iter) {
        negative_laplacian(m, direction, ap);
        double pap = 0.0;
        for (std::size_t k = 0; k < rhs.size(); ++k)
            pap += direction[k] * ap[k];
        if (!(pap > 0.0) || !std::isfinite(pap))
            throw std::runtime_error("CG breakdown in pressure solve");
        const double alpha = rr / pap;
        double next_rr = 0.0;
        for (std::size_t k = 0; k < rhs.size(); ++k) {
            pressure[k] += alpha * direction[k];
            r[k] -= alpha * ap[k];
            next_rr += r[k] * r[k];
        }
        if (next_rr <= target) return iter;
        const double beta = next_rr / rr;
        for (std::size_t k = 0; k < rhs.size(); ++k)
            direction[k] = r[k] + beta * direction[k];
        rr = next_rr;
    }
    throw std::runtime_error("pressure CG did not converge; increase max_cg");
}

// u^(n+1) = u* - dt grad_h(p); D_h grad_h is exactly Delta_h.
void correct_velocity(const Mesh& m, const Grid& u_star, const Grid& v_star,
                      const Grid& pressure, Grid& u, Grid& v, double dt) {
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i) {
            const std::size_t k = m.index(i, j);
            u[k] = u_star[k] - dt * (pressure[k] - m.at(pressure, i - 1, j)) * m.inv_h;
            v[k] = v_star[k] - dt * (pressure[k] - m.at(pressure, i, j - 1)) * m.inv_h;
        }
    }
}

struct Errors {
    double velocity_rms = 0.0;
    double pressure_rms = 0.0;
    double max_divergence = 0.0;
};

Errors measure_errors(const Mesh& m, const Grid& u, const Grid& v,
                      const Grid& pressure, Grid& div, double nu, double t) {
    divergence(m, u, v, div);
    double velocity_error2 = 0.0, pressure_error2 = 0.0, max_div = 0.0;
    const double decay = std::exp(-2.0 * nu * t);
    double mean_p = 0.0;
    for (double p : pressure) mean_p += p;
    mean_p /= static_cast<double>(pressure.size()); // pressure gauge is arbitrary
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i) {
            const std::size_t k = m.index(i, j);
            const double x = i * m.h, y = j * m.h;
            const double exact_u = std::sin(x) * std::cos(y + 0.5 * m.h) * decay;
            const double exact_v = -std::cos(x + 0.5 * m.h) * std::sin(y) * decay;
            const double exact_p = 0.25 *
                (std::cos(2.0 * (x + 0.5 * m.h)) +
                 std::cos(2.0 * (y + 0.5 * m.h))) * decay * decay;
            velocity_error2 += (u[k] - exact_u) * (u[k] - exact_u) +
                               (v[k] - exact_v) * (v[k] - exact_v);
            pressure_error2 += (pressure[k] - mean_p - exact_p) *
                               (pressure[k] - mean_p - exact_p);
            max_div = std::max(max_div, std::abs(div[k]));
        }
    }
    const double cells = static_cast<double>(pressure.size());
    return {std::sqrt(velocity_error2 / cells),
            std::sqrt(pressure_error2 / cells), max_div};
}

// Save final cell-centered speed |(u,v)| as a binary PPM (P6) image.
// Face values are averaged to cell centers; fixed color scale 0 (blue) to 1 (red).
void write_speed_ppm(const Mesh& m, const Grid& u, const Grid& v,
                     const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open image file: " + filename);
    out << "P6\n" << m.n << ' ' << m.n << "\n255\n";
    for (int j = 0; j < m.n; ++j) {
        for (int i = 0; i < m.n; ++i) {
            const double uc = 0.5 * (m.at(u, i, j) + m.at(u, i + 1, j));
            const double vc = 0.5 * (m.at(v, i, j) + m.at(v, i, j + 1));
            const double s = std::clamp(std::hypot(uc, vc), 0.0, 1.0);
            const unsigned char pixel[3] = {
                static_cast<unsigned char>(255.0 * s),
                static_cast<unsigned char>(255.0 * (1.0 - std::abs(2.0 * s - 1.0))),
                static_cast<unsigned char>(255.0 * (1.0 - s))
            };
            out.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
        }
    }
    if (!out) throw std::runtime_error("failed to write image file: " + filename);
}

int main(int argc, char** argv) {
    try {
        std::string image_file;
        int numeric_argc = argc;
        if (argc >= 3 && std::string(argv[argc - 2]) == "--ppm") {
            image_file = argv[argc - 1];
            numeric_argc -= 2;
            if (image_file.empty()) throw std::invalid_argument("empty PPM filename");
        }
        if (numeric_argc > 7) {
            std::cerr << "Usage: " << argv[0]
                      << " [N=64] [steps=100] [dt=0.01] [nu=0.1]"
                         " [max_cg=1000] [cg_tol=1e-10] [--ppm image.ppm]\n";
            return 1;
        }
        Parameters args;
        if (numeric_argc > 1) args.n = parse_int(argv[1]);
        if (numeric_argc > 2) args.steps = parse_int(argv[2]);
        if (numeric_argc > 3) args.dt = parse_double(argv[3]);
        if (numeric_argc > 4) args.nu = parse_double(argv[4]);
        if (numeric_argc > 5) args.max_cg = parse_int(argv[5]);
        if (numeric_argc > 6) args.cg_tol = parse_double(argv[6]);
        if (args.n < 8 || args.n > 10000 || args.steps < 0 || args.dt <= 0.0
            || args.nu <= 0.0 || args.max_cg < 1 || args.cg_tol <= 0.0
            || args.cg_tol >= 1.0)
            throw std::invalid_argument("require 8 <= N <= 10000, steps >= 0,"
                                        " dt, nu > 0, max_cg >= 1, 0 < cg_tol < 1");

        const double pi = std::acos(-1.0);
        const double h = 2.0 * pi / args.n;
        // Sufficient conservative limits for the Taylor--Green initial speed <= 1.
        if (args.dt > std::min(0.5 * h, h * h / (4.0 * args.nu)))
            throw std::invalid_argument("dt too large: reduce dt for advection/diffusion stability");
        const Mesh m{args.n, h, 1.0 / h, 1.0 / (h * h)};
        const std::size_t cells = static_cast<std::size_t>(args.n) * args.n;
        Grid u(cells), v(cells), pressure(cells);
        Grid u_star(cells), v_star(cells), rhs(cells), r(cells), direction(cells), ap(cells);
        initialize(m, u, v, pressure);

        const auto start = std::chrono::steady_clock::now();
        int max_iterations_used = 0;
        for (int step = 0; step < args.steps; ++step) {
            predict_velocity(m, u, v, u_star, v_star, args.dt, args.nu);
            divergence(m, u_star, v_star, rhs);
            // A = -Delta_h, so A*p = -div(u*)/dt. Remove roundoff in mean(rhs).
            double mean_rhs = 0.0;
            for (double& value : rhs) {
                value /= -args.dt;
                mean_rhs += value;
            }
            mean_rhs /= static_cast<double>(cells);
            for (double& value : rhs) value -= mean_rhs;
            const int used = solve_pressure(m, rhs, pressure, r, direction, ap,
                                            args.max_cg, args.cg_tol);
            max_iterations_used = std::max(max_iterations_used, used);
            correct_velocity(m, u_star, v_star, pressure, u, v, args.dt);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double t = args.steps * args.dt;
        Grid div(cells);
        const Errors errors = measure_errors(m, u, v, pressure, div, args.nu, t);
        std::cout << std::setprecision(10)
                  << "N=" << args.n << " steps=" << args.steps << " dt=" << args.dt
                  << " nu=" << args.nu << " time=" << t << '\n'
                  << "compute_seconds=" << std::chrono::duration<double>(stop - start).count() << '\n'
                  << "velocity_rms_error=" << errors.velocity_rms
                  << " pressure_rms_error=" << errors.pressure_rms << '\n'
                  << "max_abs_divergence=" << errors.max_divergence
                  << " max_cg_iterations_used=" << max_iterations_used << '\n';
        if (!image_file.empty()) {
            write_speed_ppm(m, u, v, image_file);
            std::cout << "speed_image=" << image_file << " (PPM P6, speed 0..1)\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
