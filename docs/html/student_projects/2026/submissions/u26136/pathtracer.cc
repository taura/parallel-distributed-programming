#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-6;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  Vec3() = default;
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
  Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
  Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
  Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
  Vec3 operator*(const Vec3& b) const { return Vec3(x * b.x, y * b.y, z * b.z); }

  Vec3& operator+=(const Vec3& b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
};

Vec3 operator*(double s, const Vec3& v) { return v * s; }

double Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

double Length(const Vec3& v) { return std::sqrt(Dot(v, v)); }

Vec3 Normalize(const Vec3& v) {
  const double len = Length(v);
  if (len <= 0.0) return Vec3(0.0, 0.0, 0.0);
  return v / len;
}

Vec3 Clamp01(const Vec3& v) {
  return Vec3(std::clamp(v.x, 0.0, 1.0), std::clamp(v.y, 0.0, 1.0), std::clamp(v.z, 0.0, 1.0));
}

struct Ray {
  Vec3 o;
  Vec3 d;
};

enum class MaterialType {
  kDiffuse,
  kMirror,
  kLight,
};

struct Material {
  MaterialType type = MaterialType::kDiffuse;
  Vec3 color = Vec3(1.0, 1.0, 1.0);
  Vec3 emission = Vec3(0.0, 0.0, 0.0);
};

struct Sphere {
  Vec3 c;
  double r = 1.0;
  int mat_id = 0;
};

struct Triangle {
  Vec3 v0;
  Vec3 v1;
  Vec3 v2;
  int mat_id = 0;
};

struct Hit {
  bool hit = false;
  double t = std::numeric_limits<double>::infinity();
  Vec3 p;
  Vec3 n;
  int mat_id = -1;
};

class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed ? seed : 1ull) {}

  uint64_t NextU64() {
    uint64_t x = state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state_ = x;
    return x * 2685821657736338717ull;
  }

  double Next01() {
    const uint64_t bits = NextU64() >> 11;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
  }

 private:
  uint64_t state_;
};

uint64_t MixSeed(uint64_t a, uint64_t b) {
  uint64_t x = a + 0x9e3779b97f4a7c15ull + (b << 6) + (b >> 2);
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

bool IntersectSphere(const Ray& ray, const Sphere& s, Hit& hit) {
  const Vec3 oc = ray.o - s.c;
  const double b = Dot(oc, ray.d);
  const double c = Dot(oc, oc) - s.r * s.r;
  const double disc = b * b - c;
  if (disc < 0.0) return false;
  const double sd = std::sqrt(disc);
  double t = -b - sd;
  if (t <= kEps) t = -b + sd;
  if (t <= kEps || t >= hit.t) return false;
  hit.hit = true;
  hit.t = t;
  hit.p = ray.o + ray.d * t;
  hit.n = Normalize(hit.p - s.c);
  hit.mat_id = s.mat_id;
  return true;
}

bool IntersectTriangle(const Ray& ray, const Triangle& tri, Hit& hit) {
  const Vec3 e1 = tri.v1 - tri.v0;
  const Vec3 e2 = tri.v2 - tri.v0;
  const Vec3 pvec = Cross(ray.d, e2);
  const double det = Dot(e1, pvec);
  if (std::abs(det) < 1e-10) return false;
  const double inv_det = 1.0 / det;
  const Vec3 tvec = ray.o - tri.v0;
  const double u = Dot(tvec, pvec) * inv_det;
  if (u < 0.0 || u > 1.0) return false;
  const Vec3 qvec = Cross(tvec, e1);
  const double v = Dot(ray.d, qvec) * inv_det;
  if (v < 0.0 || u + v > 1.0) return false;
  const double t = Dot(e2, qvec) * inv_det;
  if (t <= kEps || t >= hit.t) return false;

  Vec3 n = Normalize(Cross(e1, e2));
  if (Dot(n, ray.d) > 0.0) n = n * -1.0;

  hit.hit = true;
  hit.t = t;
  hit.p = ray.o + ray.d * t;
  hit.n = n;
  hit.mat_id = tri.mat_id;
  return true;
}

struct Scene {
  std::vector<Material> materials;
  std::vector<Sphere> spheres;
  std::vector<Triangle> triangles;
};

void AddQuad(std::vector<Triangle>& tris, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d,
             int mat_id) {
  tris.push_back(Triangle{a, b, c, mat_id});
  tris.push_back(Triangle{a, c, d, mat_id});
}

Scene MakeCornellScene() {
  Scene scene;
  scene.materials.push_back(Material{MaterialType::kDiffuse, Vec3(0.75, 0.75, 0.75), Vec3(0.0, 0.0, 0.0)});  // 0 white
  scene.materials.push_back(Material{MaterialType::kDiffuse, Vec3(0.12, 0.45, 0.15), Vec3(0.0, 0.0, 0.0)});  // 1 green
  scene.materials.push_back(Material{MaterialType::kDiffuse, Vec3(0.65, 0.05, 0.05), Vec3(0.0, 0.0, 0.0)});  // 2 red
  scene.materials.push_back(Material{MaterialType::kLight, Vec3(1.0, 1.0, 1.0), Vec3(17.0, 17.0, 17.0)});    // 3 light
  scene.materials.push_back(Material{MaterialType::kMirror, Vec3(0.92, 0.92, 0.92), Vec3(0.0, 0.0, 0.0)});   // 4 mirror
  scene.materials.push_back(Material{MaterialType::kDiffuse, Vec3(0.2, 0.35, 0.8), Vec3(0.0, 0.0, 0.0)});    // 5 blue
  scene.materials.push_back(Material{MaterialType::kDiffuse, Vec3(0.85, 0.75, 0.2), Vec3(0.0, 0.0, 0.0)});   // 6 yellow

  const Vec3 A(0.0, 0.0, 0.0);
  const Vec3 B(1.0, 0.0, 0.0);
  const Vec3 C(1.0, 1.0, 0.0);
  const Vec3 D(0.0, 1.0, 0.0);
  const Vec3 E(0.0, 0.0, 1.0);
  const Vec3 F(1.0, 0.0, 1.0);
  const Vec3 G(1.0, 1.0, 1.0);
  const Vec3 H(0.0, 1.0, 1.0);

  AddQuad(scene.triangles, A, B, F, E, 0);  // floor
  AddQuad(scene.triangles, D, H, G, C, 0);  // ceiling
  AddQuad(scene.triangles, A, E, H, D, 2);  // left red
  AddQuad(scene.triangles, B, C, G, F, 1);  // right green
  AddQuad(scene.triangles, E, F, G, H, 0);  // back wall

  const double ly = 0.999;
  AddQuad(scene.triangles, Vec3(0.34, ly, 0.34), Vec3(0.66, ly, 0.34), Vec3(0.66, ly, 0.66),
          Vec3(0.34, ly, 0.66), 3);

  scene.spheres.push_back(Sphere{Vec3(0.32, 0.20, 0.55), 0.20, 4});
  scene.spheres.push_back(Sphere{Vec3(0.68, 0.16, 0.30), 0.16, 5});

  scene.triangles.push_back(Triangle{Vec3(0.50, 0.00, 0.70), Vec3(0.80, 0.00, 0.88), Vec3(0.67, 0.42, 0.84), 6});
  scene.triangles.push_back(Triangle{Vec3(0.50, 0.00, 0.70), Vec3(0.67, 0.42, 0.84), Vec3(0.32, 0.38, 0.80), 6});
  scene.triangles.push_back(Triangle{Vec3(0.50, 0.00, 0.70), Vec3(0.32, 0.38, 0.80), Vec3(0.20, 0.00, 0.86), 6});
  scene.triangles.push_back(Triangle{Vec3(0.20, 0.00, 0.86), Vec3(0.80, 0.00, 0.88), Vec3(0.67, 0.42, 0.84), 6});
  scene.triangles.push_back(Triangle{Vec3(0.20, 0.00, 0.86), Vec3(0.67, 0.42, 0.84), Vec3(0.32, 0.38, 0.80), 6});

  return scene;
}

bool IntersectScene(const Scene& scene, const Ray& ray, Hit& hit) {
  bool any = false;
  for (const auto& s : scene.spheres) {
    any |= IntersectSphere(ray, s, hit);
  }
  for (const auto& tri : scene.triangles) {
    any |= IntersectTriangle(ray, tri, hit);
  }
  return any;
}

Vec3 BuildOrthonormal(const Vec3& n, const Vec3& local) {
  Vec3 t = (std::abs(n.z) < 0.999) ? Normalize(Cross(Vec3(0.0, 0.0, 1.0), n)) : Normalize(Cross(Vec3(0.0, 1.0, 0.0), n));
  Vec3 b = Cross(n, t);
  return t * local.x + b * local.y + n * local.z;
}

Vec3 SampleCosineHemisphere(Rng& rng) {
  const double u1 = rng.Next01();
  const double u2 = rng.Next01();
  const double r = std::sqrt(u1);
  const double phi = 2.0 * kPi * u2;
  const double x = r * std::cos(phi);
  const double y = r * std::sin(phi);
  const double z = std::sqrt(std::max(0.0, 1.0 - u1));
  return Vec3(x, y, z);
}

Vec3 TracePath(const Scene& scene, Ray ray, Rng& rng, int max_depth) {
  Vec3 radiance(0.0, 0.0, 0.0);
  Vec3 throughput(1.0, 1.0, 1.0);

  for (int depth = 0; depth < max_depth; ++depth) {
    Hit hit;
    if (!IntersectScene(scene, ray, hit)) break;

    const Material& mat = scene.materials[hit.mat_id];
    radiance += throughput * mat.emission;

    if (mat.type == MaterialType::kLight) break;

    ray.o = hit.p + hit.n * 1e-4;

    if (mat.type == MaterialType::kMirror) {
      ray.d = Normalize(ray.d - hit.n * (2.0 * Dot(ray.d, hit.n)));
      throughput = throughput * mat.color;
    } else {
      const Vec3 local = SampleCosineHemisphere(rng);
      ray.d = Normalize(BuildOrthonormal(hit.n, local));
      throughput = throughput * mat.color;
    }

    if (depth >= 3) {
      const double p = std::clamp(std::max({throughput.x, throughput.y, throughput.z}), 0.05, 0.95);
      if (rng.Next01() > p) break;
      throughput = throughput / p;
    }
  }

  return radiance;
}

struct Args {
  int width = 512;
  int height = 512;
  int spp = 64;
  int max_depth = 8;
  uint64_t seed = 42;
  std::string out = "image.ppm";
};

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t idx = 0;
    const int v = std::stoi(s, &idx);
    if (idx != s.size()) return false;
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseU64(const std::string& s, uint64_t& out) {
  try {
    size_t idx = 0;
    const unsigned long long v = std::stoull(s, &idx);
    if (idx != s.size()) return false;
    out = static_cast<uint64_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseArgs(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto require_value = [&](std::string& value) -> bool {
      if (i + 1 >= argc) return false;
      value = argv[++i];
      return true;
    };

    std::string value;
    if (key == "--width" && require_value(value)) {
      if (!ParseInt(value, args.width) || args.width <= 0) return false;
    } else if (key == "--height" && require_value(value)) {
      if (!ParseInt(value, args.height) || args.height <= 0) return false;
    } else if (key == "--spp" && require_value(value)) {
      if (!ParseInt(value, args.spp) || args.spp <= 0) return false;
    } else if (key == "--max-depth" && require_value(value)) {
      if (!ParseInt(value, args.max_depth) || args.max_depth <= 0) return false;
    } else if (key == "--seed" && require_value(value)) {
      if (!ParseU64(value, args.seed)) return false;
    } else if (key == "--output" && require_value(value)) {
      args.out = value;
    } else if (key == "--help") {
      return false;
    } else {
      return false;
    }
  }
  return true;
}

void WritePPM(const std::string& path, int width, int height, const std::vector<Vec3>& img) {
  std::ofstream ofs(path, std::ios::binary);
  ofs << "P6\n" << width << " " << height << "\n255\n";
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      Vec3 c = Clamp01(img[y * width + x]);
      c = Vec3(std::pow(c.x, 1.0 / 2.2), std::pow(c.y, 1.0 / 2.2), std::pow(c.z, 1.0 / 2.2));
      const unsigned char r = static_cast<unsigned char>(std::clamp(c.x * 255.0 + 0.5, 0.0, 255.0));
      const unsigned char g = static_cast<unsigned char>(std::clamp(c.y * 255.0 + 0.5, 0.0, 255.0));
      const unsigned char b = static_cast<unsigned char>(std::clamp(c.z * 255.0 + 0.5, 0.0, 255.0));
      ofs.write(reinterpret_cast<const char*>(&r), 1);
      ofs.write(reinterpret_cast<const char*>(&g), 1);
      ofs.write(reinterpret_cast<const char*>(&b), 1);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, args)) {
    std::cerr << "Usage: " << argv[0]
              << " [--width W] [--height H] [--spp S] [--max-depth D] [--seed N] [--output image.ppm]\n";
    return 1;
  }

  const Scene scene = MakeCornellScene();
  std::vector<Vec3> image(static_cast<size_t>(args.width) * static_cast<size_t>(args.height), Vec3(0.0, 0.0, 0.0));

  const Vec3 cam_pos(0.5, 0.5, -1.7);
  const Vec3 cam_target(0.5, 0.5, 0.5);
  const Vec3 cam_up(0.0, 1.0, 0.0);
  const double fov_deg = 40.0;

  const Vec3 forward = Normalize(cam_target - cam_pos);
  const Vec3 right = Normalize(Cross(forward, cam_up));
  const Vec3 up = Normalize(Cross(right, forward));
  const double aspect = static_cast<double>(args.width) / static_cast<double>(args.height);
  const double scale = std::tan(0.5 * fov_deg * kPi / 180.0);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int y = 0; y < args.height; ++y) {
    for (int x = 0; x < args.width; ++x) {
      Vec3 sum(0.0, 0.0, 0.0);
      const uint64_t pixel_seed = MixSeed(args.seed, static_cast<uint64_t>(y) * static_cast<uint64_t>(args.width) + static_cast<uint64_t>(x));
      Rng rng(pixel_seed);

      for (int s = 0; s < args.spp; ++s) {
        const double jx = rng.Next01();
        const double jy = rng.Next01();
        const double ndc_x = ((static_cast<double>(x) + jx) / static_cast<double>(args.width)) * 2.0 - 1.0;
        const double ndc_y = 1.0 - ((static_cast<double>(y) + jy) / static_cast<double>(args.height)) * 2.0;
        const Vec3 dir = Normalize(forward + right * (ndc_x * aspect * scale) + up * (ndc_y * scale));
        sum += TracePath(scene, Ray{cam_pos, dir}, rng, args.max_depth);
      }

      image[y * args.width + x] = sum / static_cast<double>(args.spp);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double checksum = 0.0;
  for (const auto& c : image) checksum += c.x + c.y + c.z;

  WritePPM(args.out, args.width, args.height, image);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "render_ms " << elapsed_ms << "\n";
  std::cout << "checksum " << std::setprecision(10) << checksum << "\n";
  std::cout << "output " << args.out << "\n";
  return 0;
}
