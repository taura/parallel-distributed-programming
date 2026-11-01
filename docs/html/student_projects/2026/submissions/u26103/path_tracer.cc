// Serial path tracer: sunlight and soft daylight through a living-room window.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec3 {
    double x, y, z;
    Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return *this * (1.0 / s); }
};

Vec3 multiply(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
Vec3 normalize(Vec3 a) { return a / std::sqrt(dot(a, a)); }

struct Ray { Vec3 origin, direction; };
struct Sphere { Vec3 center; double radius; Vec3 color; bool mirror; };
struct Box { Vec3 low, high, color; bool mirror = false; };
struct Hit {
    double t = std::numeric_limits<double>::infinity();
    Vec3 position{}, normal{}, color{};
    bool mirror = false;
    bool light = false;
};

// SplitMix64: each pixel gets its own reproducible random stream.
struct Random {
    std::uint64_t state;
    std::uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t x = state;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

constexpr double pi = 3.14159265358979323846;
constexpr double epsilon = 1e-4;
constexpr double window_z = -6.02;
constexpr double window_bottom = 1.35;
constexpr double window_top = 4.35;
constexpr double window_inner_x = 1.25;
constexpr double window_outer_x = 3.70;
constexpr double window_area = 2.0 * (window_outer_x - window_inner_x) *
                               (window_top - window_bottom);
const Vec3 sky_radiance{5.0, 7.0, 10.0};
const Vec3 sun_irradiance{14.0, 12.0, 9.0};
const Vec3 sun_direction = normalize(Vec3{-0.32, 0.38, -0.87});
const std::vector<Sphere> spheres = {
    // A ceramic vase and a small polished metal sculpture on its stand.
    {{ 0.15, 1.30, -1.65}, 0.28, {0.76, 0.19, 0.10}, false},
    {{ 2.25, 1.28, -3.25}, 0.38, {0.88, 0.91, 0.95}, true},
};
const std::vector<Box> boxes = {
    // Two actual openings in the back wall; no opaque wall covers the windows.
    {{-4.6, 0.00, -6.1}, { 4.6, 1.35, -5.9}, {0.80, 0.77, 0.69}},
    {{-4.6, 4.35, -6.1}, { 4.6, 5.80, -5.9}, {0.80, 0.77, 0.69}},
    {{-4.6, 1.35, -6.1}, {-3.70, 4.35, -5.9}, {0.80, 0.77, 0.69}},
    {{-1.25, 1.35, -6.1}, { 1.25, 4.35, -5.9}, {0.80, 0.77, 0.69}},
    {{ 3.70, 1.35, -6.1}, { 4.60, 4.35, -5.9}, {0.80, 0.77, 0.69}},
    {{-4.8, 0.00, -6.1}, {-4.6, 5.80,  5.5}, {0.72, 0.74, 0.70}},
    {{ 4.6, 0.00, -6.1}, { 4.8, 5.80,  5.5}, {0.78, 0.75, 0.69}},
    {{-4.8, 5.80, -6.1}, { 4.8, 6.00,  5.5}, {0.82, 0.81, 0.76}},
    // Window mullions, narrow rails, and sills cast shapes into the sunlight.
    {{-2.53, 1.35, -5.88}, {-2.42, 4.35, -5.80}, {0.82, 0.80, 0.73}},
    {{ 2.42, 1.35, -5.88}, { 2.53, 4.35, -5.80}, {0.82, 0.80, 0.73}},
    {{-3.70, 2.68, -5.88}, {-1.25, 2.76, -5.80}, {0.82, 0.80, 0.73}},
    {{ 1.25, 2.68, -5.88}, { 3.70, 2.76, -5.80}, {0.82, 0.80, 0.73}},
    {{-3.83, 1.29, -5.88}, {-1.12, 1.38, -5.66}, {0.58, 0.49, 0.37}},
    {{ 1.12, 1.29, -5.88}, { 3.83, 1.38, -5.66}, {0.58, 0.49, 0.37}},
    // Rug, coffee table and its four legs, and a small ceramic vase.
    {{-2.10, 0.00, -3.20}, {1.55, 0.025, 0.90}, {0.62, 0.37, 0.25}},
    {{-0.90, 0.90, -2.32}, {1.00, 1.02, -0.88}, {0.49, 0.28, 0.14}},
    {{-0.78, 0.00, -2.20}, {-0.61, 0.90, -2.03}, {0.39, 0.22, 0.12}},
    {{ 0.71, 0.00, -2.20}, { 0.88, 0.90, -2.03}, {0.39, 0.22, 0.12}},
    {{-0.78, 0.00, -1.17}, {-0.61, 0.90, -1.00}, {0.39, 0.22, 0.12}},
    {{ 0.71, 0.00, -1.17}, { 0.88, 0.90, -1.00}, {0.39, 0.22, 0.12}},
    // An upholstered sofa by the right wall: seat, back, and two arms.
    {{1.95, 0.38, -0.30}, {4.15, 0.76, 1.55}, {0.22, 0.42, 0.51}},
    {{3.82, 0.76, -0.30}, {4.15, 1.55, 1.55}, {0.20, 0.38, 0.47}},
    {{1.95, 0.76, -0.30}, {4.15, 1.13, 0.00}, {0.19, 0.36, 0.44}},
    {{1.95, 0.76, 1.25}, {4.15, 1.13, 1.55}, {0.19, 0.36, 0.44}},
    // Bookshelf on the right: planks and a few book-sized colored boxes.
    {{4.05, 0.00, -5.05}, {4.48, 3.28, -2.95}, {0.38, 0.24, 0.14}},
    {{3.86, 0.82, -5.08}, {4.48, 0.93, -2.92}, {0.63, 0.41, 0.23}},
    {{3.86, 1.63, -5.08}, {4.48, 1.74, -2.92}, {0.63, 0.41, 0.23}},
    {{3.86, 2.44, -5.08}, {4.48, 2.55, -2.92}, {0.63, 0.41, 0.23}},
    {{3.74, 1.74, -4.80}, {4.02, 2.43, -4.58}, {0.64, 0.17, 0.11}},
    {{3.74, 1.74, -4.47}, {4.02, 2.37, -4.24}, {0.16, 0.38, 0.48}},
    {{3.74, 1.74, -4.13}, {4.02, 2.38, -3.84}, {0.64, 0.52, 0.27}},
    {{3.74, 0.93, -4.62}, {4.02, 1.53, -4.35}, {0.20, 0.40, 0.26}},
    {{3.74, 0.93, -4.25}, {4.02, 1.51, -3.95}, {0.57, 0.32, 0.29}},
    // A low plinth carrying a mirror-finished decorative object.
    {{1.80, 0.00, -3.70}, {2.70, 0.90, -2.80}, {0.71, 0.65, 0.52}},
};

// A slab intersection, returning the normal of the visible box face.
void intersect_box(Ray ray, const Box& box, Hit& hit) {
    double near_t = -std::numeric_limits<double>::infinity();
    double far_t = std::numeric_limits<double>::infinity();
    Vec3 near_normal{}, far_normal{};
    const double origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const double direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    const double low[3] = {box.low.x, box.low.y, box.low.z};
    const double high[3] = {box.high.x, box.high.y, box.high.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(direction[axis]) < 1e-12) {
            if (origin[axis] < low[axis] || origin[axis] > high[axis]) return;
            continue;
        }
        double a = (low[axis] - origin[axis]) / direction[axis];
        double b = (high[axis] - origin[axis]) / direction[axis];
        Vec3 axis_normal = axis == 0 ? Vec3{1.0, 0.0, 0.0}
                         : axis == 1 ? Vec3{0.0, 1.0, 0.0} : Vec3{0.0, 0.0, 1.0};
        Vec3 n_a = -axis_normal;
        Vec3 n_b = axis_normal;
        if (a > b) {
            std::swap(a, b);
            std::swap(n_a, n_b);
        }
        if (a > near_t) { near_t = a; near_normal = n_a; }
        if (b < far_t) { far_t = b; far_normal = n_b; }
        if (near_t > far_t) return;
    }
    double t = near_t > epsilon ? near_t : far_t;
    if (t <= epsilon || t >= hit.t) return;
    hit.t = t;
    hit.position = ray.origin + ray.direction * t;
    hit.normal = near_t > epsilon ? near_normal : far_normal;
    hit.color = box.color;
    hit.mirror = box.mirror;
    hit.light = false;
}

Hit intersect(Ray ray, bool include_light = true) {
    Hit hit;
    for (const Box& box : boxes) intersect_box(ray, box, hit);
    for (const Sphere& sphere : spheres) {
        Vec3 oc = ray.origin - sphere.center;
        double b = dot(oc, ray.direction);
        double c = dot(oc, oc) - sphere.radius * sphere.radius;
        double discriminant = b * b - c;
        if (discriminant < 0.0) continue;
        double root = std::sqrt(discriminant);
        double t = -b - root;
        if (t <= epsilon) t = -b + root;
        if (t <= epsilon || t >= hit.t) continue;
        hit.t = t;
        hit.position = ray.origin + ray.direction * t;
        hit.normal = normalize(hit.position - sphere.center);
        hit.color = sphere.color;
        hit.mirror = sphere.mirror;
        hit.light = false;
    }

    if (std::abs(ray.direction.y) > 1e-12) {
        double t = -ray.origin.y / ray.direction.y;
        Vec3 p = ray.origin + ray.direction * t;
        if (t > epsilon && t < hit.t && std::abs(p.x) < 4.6 &&
            p.z > -6.1 && p.z < 5.5) {
            hit.t = t;
            hit.position = p;
            hit.normal = {0.0, 1.0, 0.0};
            double tile_x = p.x - std::floor(p.x);
            double tile_z = p.z - std::floor(p.z);
            bool grout = tile_x < 0.025 || tile_z < 0.025;
            hit.color = grout ? Vec3{0.39, 0.38, 0.34}
                              : Vec3{0.76, 0.75, 0.69};
            hit.mirror = false;
            hit.light = false;
        }
    }
    if (include_light && std::abs(ray.direction.z) > 1e-12) {
        double t = (window_z - ray.origin.z) / ray.direction.z;
        if (t > epsilon && t < hit.t) {
            Vec3 p = ray.origin + ray.direction * t;
            if (p.y >= window_bottom && p.y <= window_top &&
                std::abs(p.x) >= window_inner_x && std::abs(p.x) <= window_outer_x) {
                hit.t = t;
                hit.position = p;
                hit.normal = {0.0, 0.0, 1.0};
                hit.light = true;
            }
        }
    }
    return hit;
}

// Daylight sampled from one of the two window openings (an area light).
Vec3 window_light(const Hit& hit, Random& random) {
    double window_x = window_inner_x +
                      (window_outer_x - window_inner_x) * random.uniform();
    if (random.uniform() < 0.5) window_x = -window_x;
    Vec3 light_point{window_x,
                     window_bottom + (window_top - window_bottom) * random.uniform(),
                     window_z};
    Vec3 delta = light_point - hit.position;
    double distance2 = dot(delta, delta);
    double distance = std::sqrt(distance2);
    Vec3 direction = delta / distance;
    double surface_cos = dot(hit.normal, direction);
    double light_cos = -direction.z; // window normal points into the room (+z)
    if (surface_cos <= 0.0 || light_cos <= 0.0) return {0.0, 0.0, 0.0};
    Ray shadow{hit.position + hit.normal * epsilon, direction};
    if (intersect(shadow, false).t < distance - epsilon) return {0.0, 0.0, 0.0};
    return multiply(hit.color, sky_radiance) *
           (window_area * surface_cos * light_cos / (pi * distance2));
}

// An approximately directional sun; slight angular width softens shadow edges.
Vec3 sun_light(const Hit& hit, Random& random) {
    Vec3 tangent = normalize(cross(sun_direction, Vec3{0.0, 1.0, 0.0}));
    Vec3 bitangent = cross(sun_direction, tangent);
    double angle = 2.0 * pi * random.uniform();
    double radius = 0.025 * std::sqrt(random.uniform());
    Vec3 direction = normalize(sun_direction +
                               tangent * (radius * std::cos(angle)) +
                               bitangent * (radius * std::sin(angle)));
    double surface_cos = dot(hit.normal, direction);
    if (surface_cos <= 0.0) return {0.0, 0.0, 0.0};
    Ray shadow{hit.position + hit.normal * epsilon, direction};
    if (std::isfinite(intersect(shadow, false).t)) return {0.0, 0.0, 0.0};
    return multiply(hit.color, sun_irradiance) * (surface_cos / pi);
}

// Cosine-weighted hemisphere: diffuse throughput becomes just the surface color.
Vec3 diffuse_direction(Vec3 normal, Random& random) {
    double angle = 2.0 * pi * random.uniform();
    double r = std::sqrt(random.uniform());
    double h = std::sqrt(1.0 - r * r);
    Vec3 helper = std::abs(normal.y) < 0.9 ? Vec3{0.0, 1.0, 0.0}
                                            : Vec3{1.0, 0.0, 0.0};
    Vec3 tangent = normalize(cross(helper, normal));
    Vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * (r * std::cos(angle)) +
                     bitangent * (r * std::sin(angle)) + normal * h);
}

Vec3 trace(Ray ray, int max_bounces, double sun_strength, Random& random) {
    Vec3 radiance{0.0, 0.0, 0.0};
    Vec3 throughput{1.0, 1.0, 1.0};
    bool previous_was_mirror = true;
    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        Hit hit = intersect(ray);
        if (!std::isfinite(hit.t)) break; // black background
        if (hit.light) {
            // Diffuse-to-window paths are already counted by window_light().
            if (previous_was_mirror && dot(hit.normal, -ray.direction) > 0.0)
                radiance = radiance + multiply(throughput, sky_radiance);
            break;
        }
        if (hit.mirror) {
            ray = {hit.position + hit.normal * epsilon,
                   ray.direction - hit.normal * (2.0 * dot(ray.direction, hit.normal))};
            throughput = multiply(throughput, hit.color);
            previous_was_mirror = true;
        } else {
            radiance = radiance + multiply(throughput,
                           window_light(hit, random) + sun_light(hit, random) * sun_strength);
            ray = {hit.position + hit.normal * epsilon,
                   diffuse_direction(hit.normal, random)};
            throughput = multiply(throughput, hit.color);
            previous_was_mirror = false;
        }
    }
    return radiance;
}

int parameter(const char* text, int lower, int upper) {
    std::string s(text);
    std::size_t end = 0;
    long value = std::stol(s, &end);
    if (end != s.size() || value < lower || value > upper)
        throw std::invalid_argument("parameter out of range: " + s);
    return static_cast<int>(value);
}

double sunlight_parameter(const char* text) {
    std::string s(text);
    std::size_t end = 0;
    double value = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(value) || value < 0.0 || value > 10.0)
        throw std::invalid_argument("sun strength out of range: " + s);
    return value;
}

unsigned char channel(double linear) {
    double mapped = std::max(0.0, linear) / (1.0 + std::max(0.0, linear));
    return static_cast<unsigned char>(255.0 * std::sqrt(mapped) + 0.5);
}

int main(int argc, char** argv) {
    try {
        if (argc > 8) throw std::invalid_argument("too many arguments");
        int width = argc > 1 ? parameter(argv[1], 1, 4096) : 512;
        int height = argc > 2 ? parameter(argv[2], 1, 4096) : 512;
        int samples = argc > 3 ? parameter(argv[3], 1, 4096) : 64;
        int bounces = argc > 4 ? parameter(argv[4], 1, 32) : 4;
        int seed = argc > 5 ? parameter(argv[5], 0, 2147483647) : 42;
        std::string filename = argc > 6 ? argv[6] : "image.ppm";
        double sun_strength = argc > 7 ? sunlight_parameter(argv[7]) : 1.0;

        Vec3 camera{0.35, 2.25, 4.8};
        Vec3 forward = normalize(Vec3{0.0, 1.75, -2.2} - camera);
        Vec3 right = normalize(cross(forward, {0.0, 1.0, 0.0}));
        Vec3 up = cross(right, forward);
        double scale = std::tan(58.0 * pi / 360.0);
        std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
        double mean_luminance = 0.0;
        double center_luminance = 0.0;
        std::size_t lit_pixels = 0;

        auto begin = std::chrono::steady_clock::now();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Random random{static_cast<std::uint64_t>(seed) ^
                              (static_cast<std::uint64_t>(y) * width + x + 1) *
                                  0x9e3779b97f4a7c15ULL};
                Vec3 sum{0.0, 0.0, 0.0};
                for (int s = 0; s < samples; ++s) {
                    double px = (2.0 * (x + random.uniform()) / width - 1.0) *
                                scale * width / height;
                    double py = (1.0 - 2.0 * (y + random.uniform()) / height) * scale;
                    Ray ray{camera, normalize(forward + right * px + up * py)};
                    sum = sum + trace(ray, bounces, sun_strength, random);
                }
                Vec3 color = sum / samples;
                double luminance = 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;
                mean_luminance += luminance;
                if (x == width / 2 && y == height / 2) center_luminance = luminance;
                if (luminance > 0.0) ++lit_pixels;
                std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3;
                pixels[offset] = channel(color.x);
                pixels[offset + 1] = channel(color.y);
                pixels[offset + 2] = channel(color.z);
            }
        }
        auto end = std::chrono::steady_clock::now();

        std::ofstream output(filename, std::ios::binary);
        if (!output) throw std::runtime_error("cannot open output: " + filename);
        output << "P6\n" << width << ' ' << height << "\n255\n";
        output.write(reinterpret_cast<const char*>(pixels.data()),
                     static_cast<std::streamsize>(pixels.size()));
        if (!output) throw std::runtime_error("cannot write output: " + filename);

        std::cout << std::fixed << std::setprecision(6)
                  << "render_seconds " << std::chrono::duration<double>(end - begin).count()
                  << "\nmean_linear_luminance " << mean_luminance / (width * height)
                  << "\ncenter_linear_luminance " << center_luminance
                  << "\nlit_pixels " << lit_pixels << '/' << width * height
                  << "\nimage " << filename << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what()
                  << "\nusage: ./path_tracer [width height samples bounces seed output.ppm sun_strength]\n";
        return 1;
    }
}
