// Serial 2-D indoor radio coverage: direct and single-bounce specular paths.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec {
    double x, y;
};

Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y}; }
Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y}; }
Vec operator*(Vec a, double k) { return {a.x * k, a.y * k}; }
double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
double cross(Vec a, Vec b) { return a.x * b.y - a.y * b.x; }
double length(Vec a) { return std::sqrt(dot(a, a)); }

struct Wall {
    Vec a, b;
};

// Does a wall cut the *interior* of the path? Touching its endpoint is allowed.
bool cuts_path(Vec from, Vec to, const Wall& wall) {
    const Vec ray = to - from;
    const Vec edge = wall.b - wall.a;
    const double den = cross(ray, edge);
    if (std::abs(den) < 1e-12) return false;
    const double t = cross(wall.a - from, edge) / den;
    const double u = cross(wall.a - from, ray) / den;
    return t > 1e-9 && t < 1.0 - 1e-9 && u >= 0.0 && u <= 1.0;
}

bool visible(Vec a, Vec b, const std::vector<Wall>& walls, int ignore = -1) {
    for (std::size_t i = 0; i < walls.size(); ++i) {
        if (static_cast<int>(i) != ignore && cuts_path(a, b, walls[i])) return false;
    }
    return true;
}

// Mirror the transmitter in the wall's infinite line. The mirror-to-receiver
// line intersects the finite wall at the specular reflection point.
bool reflection_point(Vec tx, Vec rx, const Wall& wall, Vec& hit) {
    const Vec edge = wall.b - wall.a;
    const Vec normal = {-edge.y, edge.x};
    const double side_tx = dot(tx - wall.a, normal);
    const double side_rx = dot(rx - wall.a, normal);
    if (side_tx * side_rx <= 1e-12) return false;

    const Vec mirror = tx - normal * (2.0 * side_tx / dot(normal, normal));
    const Vec ray = rx - mirror;
    const double den = cross(ray, edge);
    if (std::abs(den) < 1e-12) return false;
    const double t = cross(wall.a - mirror, edge) / den;
    const double u = cross(wall.a - mirror, ray) / den;
    if (t <= 1e-9 || t >= 1.0 - 1e-9 || u <= 1e-9 || u >= 1.0 - 1e-9)
        return false;
    hit = wall.a + edge * u;
    return true;
}

struct Path {
    double distance;
    double coefficient;
};

// H(f) = sum_p gamma_p * lambda/(4*pi*d_p) * exp(-i*2*pi*f*d_p/c).
void add_path(const Path& path, double frequency, double amplitude_factor,
              double& real, double& imag) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double light_speed = 299792458.0;
    const double amplitude = path.coefficient * amplitude_factor / path.distance;
    const double phase = -2.0 * pi * frequency * path.distance / light_speed;
    real += amplitude * std::cos(phase);
    imag += amplitude * std::sin(phase);
}

// Fixed dB scale: blue = weak, red = strong, white = walls.
void paint_gain(std::vector<unsigned char>& image, int nx, int ny,
                int ix, int iy, double gain) {
    const double db = 10.0 * std::log10(std::max(gain, 1e-30));
    const double t = std::clamp((db + 100.0) / 60.0, 0.0, 1.0);
    const std::size_t p = 3 * (static_cast<std::size_t>(ny - 1 - iy) * nx + ix);
    image[p] = static_cast<unsigned char>(255.0 * t);
    image[p + 1] = static_cast<unsigned char>(255.0 * (1.0 - std::abs(2.0 * t - 1.0)));
    image[p + 2] = static_cast<unsigned char>(255.0 * (1.0 - t));
}

void mark(std::vector<unsigned char>& image, int nx, int ny, int x, int y,
          unsigned char red, unsigned char green, unsigned char blue) {
    if (x < 0 || x >= nx || y < 0 || y >= ny) return;
    const std::size_t p = 3 * (static_cast<std::size_t>(y) * nx + x);
    image[p] = red;
    image[p + 1] = green;
    image[p + 2] = blue;
}

void draw_layout(std::vector<unsigned char>& image, int nx, int ny,
                 double width, double height, const std::vector<Wall>& walls) {
    for (const Wall& wall : walls) {
        const double x0 = wall.a.x / width * nx - 0.5;
        const double y0 = (1.0 - wall.a.y / height) * ny - 0.5;
        const double x1 = wall.b.x / width * nx - 0.5;
        const double y1 = (1.0 - wall.b.y / height) * ny - 0.5;
        const int steps = std::max(1, static_cast<int>(
            2.0 * std::max(std::abs(x1 - x0), std::abs(y1 - y0))));
        for (int i = 0; i <= steps; ++i) {
            const double t = static_cast<double>(i) / steps;
            mark(image, nx, ny, static_cast<int>(std::lround(x0 + t * (x1 - x0))),
                 static_cast<int>(std::lround(y0 + t * (y1 - y0))), 255, 255, 255);
        }
    }
    // Draw the room boundary even where the geometric outer walls lie
    // exactly between the image's edge pixels and the outside of the image.
    for (int x = 0; x < nx; ++x) {
        mark(image, nx, ny, x, 0, 255, 255, 255);
        mark(image, nx, ny, x, ny - 1, 255, 255, 255);
    }
    for (int y = 0; y < ny; ++y) {
        mark(image, nx, ny, 0, y, 255, 255, 255);
        mark(image, nx, ny, nx - 1, y, 255, 255, 255);
    }
    // Black cross at the transmitter.
    const int tx_x = static_cast<int>(std::lround(3.0 / width * nx - 0.5));
    const int tx_y = static_cast<int>(std::lround(0.5 * ny - 0.5));
    for (int d = -2; d <= 2; ++d) {
        mark(image, nx, ny, tx_x + d, tx_y, 0, 0, 0);
        mark(image, nx, ny, tx_x, tx_y + d, 0, 0, 0);
    }
}

void write_ppm(const std::string& name, int nx, int ny,
               const std::vector<unsigned char>& image) {
    std::ofstream file(name, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open output file: " + name);
    file << "P6\n" << nx << ' ' << ny << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()),
               static_cast<std::streamsize>(image.size()));
    if (!file) throw std::runtime_error("failed writing output file: " + name);
}

int main(int argc, char** argv) {
    try {
        if (argc > 7) throw std::invalid_argument("too many arguments");
        const int nx = argc > 1 ? std::stoi(argv[1]) : 384;
        const int ny = argc > 2 ? std::stoi(argv[2]) : 256;
        const int interior = argc > 3 ? std::stoi(argv[3]) : 32;
        const unsigned seed = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 1;
        const int reflections = argc > 5 ? std::stoi(argv[5]) : 1;
        const std::string output = argc > 6 ? argv[6] : "coverage.ppm";
        if (nx < 2 || nx > 2048 || ny < 2 || ny > 2048 ||
            interior < 0 || interior > 120 || (reflections != 0 && reflections != 1))
            throw std::invalid_argument("image width/height=2..2048, walls=0..120, "
                                        "reflections=0 or 1");

        constexpr double width = 24.0, height = 16.0;  // metres
        constexpr double carrier = 2.4e9;
        constexpr double wavelength = 299792458.0 / carrier;
        constexpr double pi = 3.14159265358979323846;
        constexpr double wall_gamma = -0.65; // simplified constant reflection coefficient
        const double amplitude_factor = wavelength / (4.0 * pi);
        const Vec tx = {3.0, 8.0};
        std::vector<Wall> walls = {
            {{0, 0}, {width, 0}}, {{width, 0}, {width, height}},
            {{width, height}, {0, height}}, {{0, height}, {0, 0}}
        };
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> xpos(5.0, width - 2.0);
        std::uniform_real_distribution<double> ypos(2.0, height - 2.0);
        std::uniform_real_distribution<double> extent(1.0, 4.0);
        for (int i = 0; i < interior; ++i) {
            const double x = xpos(rng), y = ypos(rng), size = extent(rng);
            if (i % 2 == 0)
                walls.push_back({{x, y}, {std::min(x + size, width - 0.5), y}});
            else
                walls.push_back({{x, y}, {x, std::min(y + size, height - 0.5)}});
        }

        std::vector<unsigned char> image(static_cast<std::size_t>(nx) * ny * 3);
        double sum_gain = 0.0, max_free_space_magnitude_error = 0.0;
        double min_db = 0.0, max_db = -300.0;
        std::size_t los_points = 0, reflected_paths = 0;
        const std::size_t points = static_cast<std::size_t>(nx) * ny;

        const auto start = std::chrono::steady_clock::now();
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const Vec rx_center = {width * (ix + 0.5) / nx,
                                       height * (iy + 0.5) / ny};
                const double direct_distance = std::max(length(rx_center - tx), 0.05);
                double real = 0.0, imag = 0.0;
                if (visible(tx, rx_center, walls)) {
                    ++los_points;
                    add_path({direct_distance, 1.0}, carrier, amplitude_factor, real, imag);
                }
                if (reflections) {
                    for (std::size_t w = 0; w < walls.size(); ++w) {
                        Vec hit;
                        if (reflection_point(tx, rx_center, walls[w], hit) &&
                            visible(tx, hit, walls, static_cast<int>(w)) &&
                            visible(hit, rx_center, walls, static_cast<int>(w))) {
                            ++reflected_paths;
                            add_path({length(tx - hit) + length(rx_center - hit), wall_gamma},
                                     carrier, amplitude_factor, real, imag);
                        }
                    }
                }
                const double gain = real * real + imag * imag;
                sum_gain += gain;
                if (interior == 0 && reflections == 0) {
                    const double expected = amplitude_factor / direct_distance;
                    max_free_space_magnitude_error = std::max(
                        max_free_space_magnitude_error,
                        std::abs(std::hypot(real, imag) / expected - 1.0));
                }
                const double db = 10.0 * std::log10(std::max(gain, 1e-30));
                min_db = std::min(min_db, db);
                max_db = std::max(max_db, db);
                paint_gain(image, nx, ny, ix, iy, gain);
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        draw_layout(image, nx, ny, width, height, walls);
        write_ppm(output, nx, ny, image);

        std::cout << std::fixed << std::setprecision(6)
                  << "compute_seconds=" << seconds << "\n"
                  << "receiver_locations=" << points
                  << " walls=" << walls.size() << " reflected_paths=" << reflected_paths << "\n"
                  << "los_fraction=" << static_cast<double>(los_points) / points
                  << " mean_Pr_over_Pt=" << std::scientific
                  << sum_gain / points << std::fixed << "\n"
                  << "power_range_dB=" << min_db << " " << max_db << "\n";
        if (interior == 0 && reflections == 0)
            std::cout << "max_free_space_relative_magnitude_error=" << std::scientific
                      << max_free_space_magnitude_error << std::fixed << "\n";
        std::cout << "image=" << output << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n"
                  << "usage: ./app [width=384] [height=256] [interior_walls=32] "
                     "[seed=1] [reflections=1] [output=coverage.ppm]\n";
        return 1;
    }
}
