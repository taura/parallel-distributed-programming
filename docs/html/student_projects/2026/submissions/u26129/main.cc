// Serial ray tracer: C++17 and the standard library only.
// Each pixel is independent; this baseline deliberately uses no parallel APIs.
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

constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-4;

struct Vec3 {
    double x, y, z;
    Vec3 operator+(Vec3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return *this * (1.0/s); }
};
double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
Vec3 unit(Vec3 a) { return a / std::sqrt(dot(a,a)); }
Vec3 reflect(Vec3 d, Vec3 n) { return d - n*(2.0*dot(d,n)); }

struct Ray { Vec3 origin, direction; };
struct Material { Vec3 color; double reflection; };
struct Sphere { Vec3 center; double radius; Material material; };
struct Hit { double t; Vec3 point, normal; Material material; };
struct Scene { std::vector<Sphere> spheres; bool floor = true; };
struct Stats { std::uint64_t traced_rays = 0, shadow_rays = 0; };

// Solve |origin + t*direction - center|^2 = radius^2.
bool hit_sphere(const Sphere& sphere, const Ray& ray,
                double t_min, double t_max, Hit& hit) {
    const Vec3 oc = ray.origin - sphere.center;
    const double a = dot(ray.direction, ray.direction);
    const double half_b = dot(oc, ray.direction);
    const double c = dot(oc, oc) - sphere.radius*sphere.radius;
    const double discriminant = half_b*half_b - a*c;
    if (discriminant < 0.0) return false;
    const double root = std::sqrt(discriminant);
    double t = (-half_b-root)/a;
    if (t < t_min || t > t_max) {
        t = (-half_b+root)/a;
        if (t < t_min || t > t_max) return false;
    }
    hit = {t, ray.origin + ray.direction*t, {}, sphere.material};
    hit.normal = (hit.point-sphere.center)/sphere.radius;
    return true;
}

bool intersect(const Scene& scene, const Ray& ray,
               double t_min, double t_max, Hit& hit) {
    bool found = false;
    Hit candidate{};
    for (const Sphere& sphere : scene.spheres) {
        if (hit_sphere(sphere, ray, t_min, t_max, candidate)) {
            found = true;
            t_max = candidate.t;
            hit = candidate;
        }
    }
    // Infinite checkerboard plane at y=0.
    if (scene.floor && std::abs(ray.direction.y) > 1e-12) {
        const double t = -ray.origin.y/ray.direction.y;
        if (t >= t_min && t <= t_max) {
            const Vec3 p = ray.origin + ray.direction*t;
            const double parity = std::fmod(std::floor(p.x)+std::floor(p.z), 2.0);
            const Vec3 color = std::abs(parity) < 0.5
                ? Vec3{0.70, 0.73, 0.78} : Vec3{0.12, 0.16, 0.22};
            hit = {t, p, {0,1,0}, {color, 0.12}};
            found = true;
        }
    }
    return found;
}

Vec3 trace(const Scene& scene, const Ray& ray, int depth,
           int max_depth, Stats& stats) {
    ++stats.traced_rays;
    Hit hit{};
    if (!intersect(scene, ray, EPS, std::numeric_limits<double>::infinity(), hit)) {
        const double t = 0.5*(ray.direction.y+1.0);
        return Vec3{0.90,0.94,1.00}*(1.0-t) + Vec3{0.30,0.52,0.80}*t;
    }
    // Face the normal toward the incoming ray (also handles inside-sphere rays).
    if (dot(hit.normal, ray.direction) > 0.0) hit.normal = hit.normal*(-1.0);
    const Vec3 light_position{-3.0,7.0,5.0};
    const Vec3 shadow_origin = hit.point + hit.normal*EPS;
    const Vec3 to_light = light_position-shadow_origin;
    const double light_distance = std::sqrt(dot(to_light,to_light));
    const Vec3 light_direction = to_light/light_distance;
    const double diffuse = std::max(0.0, dot(hit.normal,light_direction));
    Vec3 color = hit.material.color*0.10; // Small constant ambient term.
    if (diffuse > 0.0) {
        Hit blocker{};
        ++stats.shadow_rays;
        const bool shadow = intersect(scene, {shadow_origin,light_direction},
                                      EPS, light_distance-EPS, blocker);
        if (!shadow) {
            color = color + hit.material.color*(0.90*diffuse);
            const Vec3 reflected_light = reflect(light_direction*(-1.0), hit.normal);
            const double specular = std::pow(std::max(0.0,
                dot(reflected_light, ray.direction*(-1.0))), 64.0);
            color = color + Vec3{1,1,1}*(0.35*specular);
        }
    }
    if (depth < max_depth && hit.material.reflection > 0.0) {
        const Ray reflected{shadow_origin, unit(reflect(ray.direction,hit.normal))};
        const Vec3 bounced = trace(scene, reflected, depth+1, max_depth, stats);
        const double r = hit.material.reflection;
        color = color*(1.0-r) + bounced*r;
    }
    // At the depth limit, use the local shading above without another bounce.
    return color;
}

// Per-pixel random state makes samples independent of future scheduling order.
// All integer arithmetic is unsigned, so overflow is defined modulo 2^64.
double random01(std::uint64_t& state) {
    std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z = (z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    z ^= z>>31;
    return static_cast<double>(z>>11)*0x1.0p-53;
}

struct Options {
    int width = 640, height = 360, samples = 16, max_depth = 3;
    std::uint64_t seed = 42;
    std::string output = "image.ppm";
    bool self_test = false, help = false;
};

std::uint64_t number(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer: " + text);
    std::size_t pos = 0;
    const auto n = std::stoull(text, &pos);
    if (pos != text.size()) throw std::runtime_error("Invalid integer: " + text);
    return n;
}
int bounded(const std::string& value, int lo, int hi, const std::string& flag) {
    const auto n = number(value);
    if (n < static_cast<unsigned>(lo) || n > static_cast<unsigned>(hi))
        throw std::runtime_error(flag + " must be in [" + std::to_string(lo)
                                 + ", " + std::to_string(hi) + "]");
    return static_cast<int>(n);
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i=1; i<argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") { o.help = true; continue; }
        if (flag == "--self-test") { o.self_test = true; continue; }
        if (flag != "--width" && flag != "--height" && flag != "--samples"
            && flag != "--max-depth" && flag != "--seed" && flag != "--output")
            throw std::runtime_error("Unknown option: " + flag);
        if (++i >= argc) throw std::runtime_error("Missing value for " + flag);
        const std::string value = argv[i];
        if (flag == "--width") o.width = bounded(value,1,16384,flag);
        else if (flag == "--height") o.height = bounded(value,1,16384,flag);
        else if (flag == "--samples") o.samples = bounded(value,1,1000000,flag);
        else if (flag == "--max-depth") o.max_depth = bounded(value,0,16,flag);
        else if (flag == "--seed") o.seed = number(value);
        else o.output = value;
    }
    if (static_cast<std::uint64_t>(o.width)*o.height > 100000000)
        throw std::runtime_error("Image must contain at most 100,000,000 pixels");
    return o;
}

void self_test() {
    int checks = 0;
    auto check = [&checks](bool ok, const char* label) {
        if (!ok) throw std::runtime_error(std::string("Self-test failed: ")+label);
        ++checks;
    };
    auto close = [](double a, double b) { return std::abs(a-b)<1e-10; };
    const Sphere sphere{{0,0,0},1,{{1,0,0},0}};
    Hit h{};
    check(hit_sphere(sphere,{{0,0,3},{0,0,-1}},EPS,100,h)
          && close(h.t,2) && close(h.normal.z,1), "front sphere hit");
    check(!hit_sphere(sphere,{{0,0,3},{0,1,0}},EPS,100,h), "sphere miss");
    check(!hit_sphere(sphere,{{0,0,3},{0,0,1}},EPS,100,h), "sphere behind ray");
    check(hit_sphere(sphere,{{0,0,0},{1,0,0}},EPS,100,h)
          && close(h.t,1), "exit from inside sphere");
    check(hit_sphere(sphere,{{1,0,3},{0,0,-1}},EPS,100,h)
          && close(h.t,3), "tangent ray");
    Scene plane{{},true};
    check(intersect(plane,{{0,2,0},{0,-1,0}},EPS,100,h)
          && close(h.t,2) && close(h.point.y,0), "floor hit");
    check(!intersect(plane,{{0,2,0},{1,0,0}},EPS,100,h), "parallel to floor");
    const Vec3 r = reflect(unit({1,-1,0}),{0,1,0});
    check(close(r.x,std::sqrt(0.5)) && close(r.y,std::sqrt(0.5)), "reflection law");
    Scene blocker{{{{0,0,5},1,{{1,1,1},0}}},false};
    check(!intersect(blocker,{{0,0,0},{0,0,1}},EPS,3,h), "blocker beyond light");
    check(intersect(blocker,{{0,0,0},{0,0,1}},EPS,6,h), "blocker before light");
    Scene ordered{{{{0,0,8},1,{{1,0,0},0}},{{0,0,4},1,{{0,1,0},0}}},false};
    check(intersect(ordered,{{0,0,0},{0,0,1}},EPS,100,h)
          && close(h.t,3), "nearest hit");
    std::uint64_t a=42, b=42;
    const double u=random01(a), v=random01(b);
    check(u==v && u>=0 && u<1, "repeatable random samples");
    std::cout << "Self-test: " << checks << " checks passed\n";
}

unsigned char channel(double x) {
    if (!std::isfinite(x)) throw std::runtime_error("Non-finite pixel value");
    return static_cast<unsigned char>(255.0*std::sqrt(std::clamp(x,0.0,1.0))+0.5);
}

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc,argv);
        if (o.help) {
            std::cout << "Serial C++17 ray tracer\n"
                "  --width N       image width (default 640)\n"
                "  --height N      image height (default 360)\n"
                "  --samples N     samples per pixel (default 16)\n"
                "  --max-depth N   maximum reflection bounces, 0..16 (default 3)\n"
                "  --seed N        unsigned 64-bit seed (default 42)\n"
                "  --output PATH   binary PPM output (default image.ppm)\n"
                "  --self-test     run geometry checks and exit\n";
            return 0;
        }
        if (o.self_test) { self_test(); return 0; }
        const Scene scene{{
            {{-1.6,1.0,-0.6},1.0,{{0.04,0.60,0.65},0.25}},
            {{1.0,0.75,0.7},0.75,{{0.90,0.12,0.07},0.18}},
            {{0.6,1.2,-2.2},1.2,{{0.30,0.35,0.42},0.85}},
            {{-0.9,0.42,1.8},0.42,{{0.86,0.89,0.94},0.70}},
            {{2.5,0.55,-1.1},0.55,{{0.95,0.56,0.08},0.30}}
        },true};
        const Vec3 eye{6.0,3.4,7.0}, target{0.0,0.9,-0.7};
        const Vec3 forward = unit(target-eye);
        const Vec3 right = unit(cross(forward,{0,1,0}));
        const Vec3 up = cross(right,forward);
        const double scale = std::tan(45.0*PI/360.0); // Vertical FOV = 45 degrees.
        const double aspect = static_cast<double>(o.width)/o.height;
        const auto pixel_count = static_cast<std::size_t>(o.width)*o.height;
        std::vector<unsigned char> image(pixel_count*3);
        Stats stats;
        const auto start = std::chrono::steady_clock::now();
        for (int y=0; y<o.height; ++y) {
            for (int x=0; x<o.width; ++x) {
                const auto index = static_cast<std::size_t>(y)*o.width+x;
                std::uint64_t state = o.seed ^
                    (static_cast<std::uint64_t>(index)*UINT64_C(0xd2b74407b1ce6e93));
                Vec3 sum{0,0,0};
                for (int sample=0; sample<o.samples; ++sample) {
                    const double jitter_x = random01(state);
                    const double jitter_y = random01(state);
                    const double sx = (2.0*(x+jitter_x)/o.width-1.0)*aspect*scale;
                    const double sy = (1.0-2.0*(y+jitter_y)/o.height)*scale;
                    const Ray ray{eye,unit(forward+right*sx+up*sy)};
                    sum = sum + trace(scene,ray,0,o.max_depth,stats);
                }
                const Vec3 average = sum/static_cast<double>(o.samples);
                image[3*index] = channel(average.x);
                image[3*index+1] = channel(average.y);
                image[3*index+2] = channel(average.z);
            }
        }
        const auto rendered = std::chrono::steady_clock::now();
        std::ofstream out(o.output,std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output: "+o.output);
        out << "P6\n" << o.width << ' ' << o.height << "\n255\n";
        out.write(reinterpret_cast<const char*>(image.data()),
                  static_cast<std::streamsize>(image.size()));
        out.close();
        if (!out) throw std::runtime_error("Failed writing output: "+o.output);
        const auto written = std::chrono::steady_clock::now();
        // FNV-1a of RGB bytes: useful for same-build repeatability checks.
        std::uint64_t checksum = UINT64_C(14695981039346656037);
        for (unsigned char byte : image) { checksum ^= byte; checksum *= UINT64_C(1099511628211); }
        const double render_s = std::chrono::duration<double>(rendered-start).count();
        const double write_s = std::chrono::duration<double>(written-rendered).count();
        const auto primary = static_cast<std::uint64_t>(pixel_count)*o.samples;
        std::cout << "image=" << o.output << "\nwidth=" << o.width
                  << " height=" << o.height << " samples=" << o.samples
                  << " max_depth=" << o.max_depth << " seed=" << o.seed
                  << "\nprimary_rays=" << primary << " traced_rays=" << stats.traced_rays
                  << " shadow_rays=" << stats.shadow_rays
                  << std::fixed << std::setprecision(6)
                  << "\nrender_seconds=" << render_s << "\nwrite_seconds=" << write_s
                  << "\nprimary_Mrays_per_second=" << (render_s>0 ? primary/render_s/1e6 : 0)
                  << "\npixel_checksum_fnv1a64=" << std::hex << checksum << std::dec << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
