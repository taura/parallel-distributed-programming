// Serial 2D weakly-compressible SPH dam-break simulation.
// Each density/force evaluation scans every particle: no spatial search grid.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Particle {
    double x, y, vx, vy;
    double rho = 0.0, pressure = 0.0;
    double ax = 0.0, ay = 0.0;
};

constexpr double pi = 3.14159265358979323846;
constexpr double rho0 = 1000.0;
constexpr double sound_speed = 35.0;
constexpr double gravity = 9.81;
constexpr double alpha = 0.2;       // Monaghan artificial viscosity
constexpr double wall_damping = 0.5;
constexpr double wall = 0.02;

// 2D poly6 density kernel: integral over the disk of radius h is one.
double density_kernel(double r2, double h2, double coefficient) {
    if (r2 >= h2) return 0.0;
    const double d = h2 - r2;
    return coefficient * d * d * d;
}

void compute_density(std::vector<Particle>& p, double h, double mass) {
    const double h2 = h * h;
    const double coefficient = 4.0 / (pi * h2 * h2 * h2 * h2);
    for (std::size_t i = 0; i < p.size(); ++i) {
        double rho = 0.0;
        for (std::size_t j = 0; j < p.size(); ++j) {
            const double dx = p[i].x - p[j].x;
            const double dy = p[i].y - p[j].y;
            rho += mass * density_kernel(dx * dx + dy * dy, h2, coefficient);
        }
        p[i].rho = rho;
        // No tensile (negative) pressure near the free surface.
        p[i].pressure = sound_speed * sound_speed * std::max(0.0, rho - rho0);
    }
}

void compute_acceleration(std::vector<Particle>& p, double h, double mass) {
    const double h2 = h * h;
    const double grad_coefficient = -30.0 / (pi * h2 * h2 * h);
    for (std::size_t i = 0; i < p.size(); ++i) {
        double ax = 0.0, ay = -gravity;
        for (std::size_t j = 0; j < p.size(); ++j) {
            if (i == j) continue;
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double r2 = dx * dx + dy * dy;
            if (r2 >= h2) continue;
            if (r2 == 0.0) {
                // Give distinct coincident particles opposite finite directions.
                dx = i < j ? 0.125 * h : -0.125 * h;
                dy = 0.0;
                r2 = dx * dx;
            }

            const double r = std::sqrt(r2);
            // Gradient of the 2D spiky kernel points towards particle j.
            const double grad_factor = grad_coefficient * (h - r) * (h - r) / r;
            const double dv_dot_dr = (p[i].vx - p[j].vx) * dx
                                   + (p[i].vy - p[j].vy) * dy;
            double viscosity = 0.0;
            if (dv_dot_dr < 0.0) {
                const double rho_avg = 0.5 * (p[i].rho + p[j].rho);
                viscosity = -alpha * sound_speed * h * dv_dot_dr
                          / (rho_avg * (r2 + 0.01 * h2));
            }
            const double pair = -mass * (p[i].pressure / (p[i].rho * p[i].rho)
                                       + p[j].pressure / (p[j].rho * p[j].rho)
                                       + viscosity) * grad_factor;
            ax += pair * dx;
            ay += pair * dy;
        }
        p[i].ax = ax;
        p[i].ay = ay;
    }
}

void integrate(std::vector<Particle>& p, double dt) {
    for (Particle& q : p) {
        // Symplectic Euler; all accelerations were computed from the old state.
        q.vx += dt * q.ax;
        q.vy += dt * q.ay;
        q.x += dt * q.vx;
        q.y += dt * q.vy;
        if (q.x < wall) { q.x = 2.0 * wall - q.x; if (q.vx < 0) q.vx *= -wall_damping; }
        if (q.x > 1.0 - wall) { q.x = 2.0 * (1.0 - wall) - q.x; if (q.vx > 0) q.vx *= -wall_damping; }
        if (q.y < wall) { q.y = 2.0 * wall - q.y; if (q.vy < 0) q.vy *= -wall_damping; }
        if (q.y > 1.0 - wall) { q.y = 2.0 * (1.0 - wall) - q.y; if (q.vy > 0) q.vy *= -wall_damping; }
        q.x = std::clamp(q.x, wall, 1.0 - wall);
        q.y = std::clamp(q.y, wall, 1.0 - wall);
    }
}

// Velocity magnitude mapped from blue (stationary) through cyan/yellow to red
// (the chosen color-scale maximum). White indicates empty space, not zero speed.
void speed_color(double fraction, unsigned char color[3]) {
    constexpr unsigned char stops[4][3] = {
        {25, 45, 190}, {20, 210, 230}, {250, 220, 35}, {215, 30, 30}
    };
    const double scaled = std::min(1.0, std::max(0.0, fraction)) * 3.0;
    const int segment = std::min(2, static_cast<int>(scaled));
    const double t = scaled - segment;
    for (int channel = 0; channel < 3; ++channel)
        color[channel] = static_cast<unsigned char>(
            (1.0 - t) * stops[segment][channel] + t * stops[segment + 1][channel]);
}

void write_ppm(const std::string& filename, const std::vector<Particle>& p,
               double color_scale, double spacing) {
    constexpr int width = 600, plot_height = 600, height = 640;
    std::vector<unsigned char> rgb(3 * width * height, 255);
    // Draw antialiased filled disks rather than a jagged integer-pixel stencil.
    // Scale their radius with the initial particle spacing to avoid crowding.
    const double radius = std::clamp(0.42 * spacing * (width - 1), 2.1, 4.0);
    const int extent = static_cast<int>(std::ceil(radius + 1.0));
    for (const Particle& q : p) {
        const double cx = q.x * (width - 1);
        const double cy = (1.0 - q.y) * (plot_height - 1);
        const double speed = std::hypot(q.vx, q.vy);
        unsigned char color[3];
        speed_color(color_scale > 0.0 ? speed / color_scale : 0.0, color);
        for (int y = static_cast<int>(cy) - extent; y <= static_cast<int>(cy) + extent; ++y) {
            for (int x = static_cast<int>(cx) - extent; x <= static_cast<int>(cx) + extent; ++x) {
                if (x < 0 || x >= width || y < 0 || y >= plot_height) continue;
                const double distance = std::hypot(x - cx, y - cy);
                const double coverage = std::clamp(radius + 0.5 - distance, 0.0, 1.0);
                if (coverage == 0.0) continue;
                const std::size_t k = 3 * (static_cast<std::size_t>(y) * width + x);
                for (int channel = 0; channel < 3; ++channel)
                    rgb[k + channel] = static_cast<unsigned char>(
                        (1.0 - coverage) * rgb[k + channel] + coverage * color[channel]);
            }
        }
    }
    // Color bar: left = 0 m/s; right = color_scale m/s (printed in stdout).
    for (int y = 610; y < 628; ++y) {
        for (int x = 30; x < 570; ++x) {
            unsigned char color[3];
            speed_color((x - 30) / 539.0, color);
            const std::size_t k = 3 * (static_cast<std::size_t>(y) * width + x);
            for (int channel = 0; channel < 3; ++channel) rgb[k + channel] = color[channel];
        }
    }
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open image: " + filename);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (!out) throw std::runtime_error("cannot write image: " + filename);
}

std::string snapshot_name(const std::string& filename, int step) {
    std::ostringstream suffix;
    suffix << "_step" << std::setw(6) << std::setfill('0') << step;
    if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".ppm")
        return filename.substr(0, filename.size() - 4) + suffix.str() + ".ppm";
    return filename + suffix.str() + ".ppm";
}

int main(int argc, char** argv) {
    try {
        if (argc > 7) throw std::invalid_argument("too many arguments");
        const int nx = argc > 1 ? std::stoi(argv[1]) : 40;
        const int ny = argc > 2 ? std::stoi(argv[2]) : 72;
        const int steps = argc > 3 ? std::stoi(argv[3]) : 3000;
        const double dt = argc > 4 ? std::stod(argv[4]) : 0.0003;
        const std::string image = argc > 5 ? argv[5] : "";
        const int snapshot_interval = argc > 6 ? std::stoi(argv[6]) : 0;
        if (nx < 2 || ny < 2 || nx > 200 || ny > 200 || steps < 0
            || !std::isfinite(dt) || dt <= 0.0 || snapshot_interval < 0
            || (snapshot_interval > 0 && image.empty()))
            throw std::invalid_argument("need 2 <= nx,ny <= 200, steps >= 0, dt > 0, "
                                        "snapshot_interval >= 0 with image.ppm");

        // Dam of water in the left portion of a unit square. Initial spacing
        // is isotropic even if the requested nx:ny ratio changes.
        const double spacing = std::min(0.35 / nx, 0.70 / ny);
        const double h = 2.0 * spacing;
        const double mass = rho0 * spacing * spacing;
        std::vector<Particle> p;
        p.reserve(static_cast<std::size_t>(nx) * ny);
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix)
                p.push_back({0.10 + (ix + 0.5) * spacing,
                             0.12 + (iy + 0.5) * spacing, 0.0, 0.0});

        double initial_center_y = 0.0;
        for (const Particle& q : p) initial_center_y += q.y;
        initial_center_y /= p.size();

        // Fixed scale allows direct color comparison between snapshot frames.
        constexpr double snapshot_color_scale = 5.0; // m/s; faster speeds saturate to red
        int snapshot_count = 0;
        if (snapshot_interval > 0) {
            write_ppm(snapshot_name(image, 0), p, snapshot_color_scale, spacing);
            ++snapshot_count;
        }
        const auto start = std::chrono::steady_clock::now();
        auto image_time = std::chrono::steady_clock::duration::zero();
        for (int step = 0; step < steps; ++step) {
            compute_density(p, h, mass);
            compute_acceleration(p, h, mass);
            integrate(p, dt);
            if (snapshot_interval > 0
                && ((step + 1) % snapshot_interval == 0 || step + 1 == steps)) {
                const auto image_start = std::chrono::steady_clock::now();
                write_ppm(snapshot_name(image, step + 1), p, snapshot_color_scale, spacing);
                image_time += std::chrono::steady_clock::now() - image_start;
                ++snapshot_count;
            }
        }
        compute_density(p, h, mass); // Report density at final positions.
        const auto end = std::chrono::steady_clock::now();

        double center_x = 0.0, center_y = 0.0, mean_rho = 0.0, max_rho = 0.0;
        double kinetic_energy = 0.0, max_speed = 0.0;
        for (const Particle& q : p) {
            center_x += q.x;
            center_y += q.y;
            mean_rho += q.rho;
            max_rho = std::max(max_rho, q.rho);
            kinetic_energy += 0.5 * mass * (q.vx * q.vx + q.vy * q.vy);
            max_speed = std::max(max_speed, std::hypot(q.vx, q.vy));
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "particles=" << p.size() << " steps=" << steps
                  << " simulated_time=" << steps * dt << " s\n"
                  << "compute_time=" << std::chrono::duration<double>(end - start - image_time).count() << " s\n"
                  << "center_y_initial=" << initial_center_y
                  << " center_x_final=" << center_x / p.size()
                  << " center_y_final=" << center_y / p.size() << "\n"
                  << "mean_density=" << mean_rho / p.size()
                  << " max_density=" << max_rho
                  << " target_density=" << rho0 << "\n"
                  << "kinetic_energy=" << kinetic_energy << "\n"
                  << "max_speed=" << max_speed << " m/s\n";
        if (!image.empty()) {
            const double color_scale = snapshot_interval > 0 ? snapshot_color_scale : max_speed;
            write_ppm(image, p, color_scale, spacing);
            std::cout << "image=" << image
                      << " (color bar: blue=0, red=" << color_scale << " m/s)\n";
            if (snapshot_interval > 0)
                std::cout << "snapshots=" << snapshot_count << " interval=" << snapshot_interval
                          << " (step 0, multiples of interval, final step; same color scale)\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what()
                  << "\nUsage: sph [nx=40] [ny=72] [steps=3000] [dt=0.0003] "
                     "[image.ppm] [snapshot_interval=0]\n";
        return 1;
    }
}
