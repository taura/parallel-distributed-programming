#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  Vec3() = default;
  Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}

  Vec3 operator-() const { return Vec3(-x, -y, -z); }
  Vec3 &operator+=(const Vec3 &o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  Vec3 &operator*=(double s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  Vec3 &operator/=(double s) {
    return (*this) *= (1.0 / s);
  }
};

static inline Vec3 operator+(Vec3 a, const Vec3 &b) { return a += b; }
static inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline Vec3 operator*(Vec3 a, double s) { return a *= s; }
static inline Vec3 operator*(double s, Vec3 a) { return a *= s; }
static inline Vec3 operator*(const Vec3 &a, const Vec3 &b) {
  return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline Vec3 operator/(Vec3 a, double s) { return a /= s; }

static inline double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline double length2(const Vec3 &v) { return dot(v, v); }
static inline double length(const Vec3 &v) { return std::sqrt(length2(v)); }
static inline Vec3 normalize(const Vec3 &v) { return v / length(v); }

struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t nextU64() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
  }

  double next01() {
    const uint64_t bits = nextU64() >> 11U;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
  }
};

static inline Vec3 randomInUnitSphere(Rng &rng) {
  while (true) {
    Vec3 p(2.0 * rng.next01() - 1.0, 2.0 * rng.next01() - 1.0,
           2.0 * rng.next01() - 1.0);
    if (length2(p) < 1.0) {
      return p;
    }
  }
}

static inline Vec3 randomUnitVector(Rng &rng) {
  return normalize(randomInUnitSphere(rng));
}

static inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2.0 * dot(v, n) * n;
}

static inline Vec3 refract(const Vec3 &uv, const Vec3 &n, double etaiOverEtat) {
  const double cosTheta = std::min(dot(-uv, n), 1.0);
  const Vec3 rOutPerp = etaiOverEtat * (uv + cosTheta * n);
  const Vec3 rOutParallel = -std::sqrt(std::fabs(1.0 - length2(rOutPerp))) * n;
  return rOutPerp + rOutParallel;
}

static inline double schlick(double cosine, double refIdx) {
  double r0 = (1.0 - refIdx) / (1.0 + refIdx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
}

struct Ray {
  Vec3 origin;
  Vec3 dir;
  Vec3 at(double t) const { return origin + t * dir; }
};

enum class MaterialType { Lambertian, Metal, Dielectric };

struct Material {
  MaterialType type = MaterialType::Lambertian;
  Vec3 albedo = Vec3(0.8, 0.8, 0.8);
  double fuzz = 0.0;
  double refIdx = 1.5;
};

struct Sphere {
  Vec3 center;
  double radius = 1.0;
  Material material;
};

struct Hit {
  double t = 0.0;
  Vec3 p;
  Vec3 normal;
  bool frontFace = false;
  const Material *material = nullptr;
};

static inline void setFaceNormal(const Ray &r, const Vec3 &outwardNormal,
                                 Hit &hit) {
  hit.frontFace = dot(r.dir, outwardNormal) < 0.0;
  hit.normal = hit.frontFace ? outwardNormal : -outwardNormal;
}

static bool hitSphere(const Sphere &s, const Ray &r, double tMin, double tMax,
                      Hit &hit) {
  const Vec3 oc = r.origin - s.center;
  const double a = length2(r.dir);
  const double halfB = dot(oc, r.dir);
  const double c = length2(oc) - s.radius * s.radius;
  const double discriminant = halfB * halfB - a * c;
  if (discriminant < 0.0) {
    return false;
  }

  const double sqrtd = std::sqrt(discriminant);
  double root = (-halfB - sqrtd) / a;
  if (root < tMin || root > tMax) {
    root = (-halfB + sqrtd) / a;
    if (root < tMin || root > tMax) {
      return false;
    }
  }

  hit.t = root;
  hit.p = r.at(root);
  const Vec3 outwardNormal = (hit.p - s.center) / s.radius;
  setFaceNormal(r, outwardNormal, hit);
  hit.material = &s.material;
  return true;
}

static bool hitWorld(const std::vector<Sphere> &world, const Ray &r, double tMin,
                     double tMax, Hit &closestHit) {
  Hit temp;
  bool hitAnything = false;
  double closest = tMax;
  for (const Sphere &s : world) {
    if (hitSphere(s, r, tMin, closest, temp)) {
      hitAnything = true;
      closest = temp.t;
      closestHit = temp;
    }
  }
  return hitAnything;
}

struct Camera {
  Vec3 origin;
  Vec3 horizontal;
  Vec3 vertical;
  Vec3 lowerLeft;

  Camera(double vfovDeg, double aspect) {
    const double theta = vfovDeg * (3.14159265358979323846 / 180.0);
    const double h = std::tan(theta / 2.0);
    const double viewportHeight = 2.0 * h;
    const double viewportWidth = aspect * viewportHeight;

    origin = Vec3(0.0, 0.0, 0.0);
    horizontal = Vec3(viewportWidth, 0.0, 0.0);
    vertical = Vec3(0.0, viewportHeight, 0.0);
    lowerLeft = origin - horizontal / 2.0 - vertical / 2.0 - Vec3(0.0, 0.0, 1.0);
  }

  Ray getRay(double u, double v) const {
    return Ray{origin, lowerLeft + u * horizontal + v * vertical - origin};
  }
};

static Vec3 background(const Vec3 &dir) {
  const Vec3 unit = normalize(dir);
  const double t = 0.5 * (unit.y + 1.0);
  return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
}

static Vec3 trace(const Ray &initialRay, const std::vector<Sphere> &world,
                  int maxDepth, Rng &rng) {
  Ray ray = initialRay;
  Vec3 attenuation(1.0, 1.0, 1.0);

  for (int depth = 0; depth < maxDepth; ++depth) {
    Hit hit;
    if (!hitWorld(world, ray, 1e-4, std::numeric_limits<double>::infinity(), hit)) {
      return attenuation * background(ray.dir);
    }

    const Material &m = *hit.material;
    if (m.type == MaterialType::Lambertian) {
      Vec3 scatterDir = hit.normal + randomUnitVector(rng);
      if (length2(scatterDir) < 1e-12) {
        scatterDir = hit.normal;
      }
      attenuation = attenuation * m.albedo;
      ray = Ray{hit.p, scatterDir};
      continue;
    }

    if (m.type == MaterialType::Metal) {
      Vec3 reflected = reflect(normalize(ray.dir), hit.normal);
      reflected = reflected + m.fuzz * randomInUnitSphere(rng);
      if (dot(reflected, hit.normal) <= 0.0) {
        return Vec3(0.0, 0.0, 0.0);
      }
      attenuation = attenuation * m.albedo;
      ray = Ray{hit.p, reflected};
      continue;
    }

    attenuation = attenuation * Vec3(1.0, 1.0, 1.0);
    const double refractionRatio = hit.frontFace ? (1.0 / m.refIdx) : m.refIdx;
    const Vec3 unitDir = normalize(ray.dir);
    const double cosTheta = std::min(dot(-unitDir, hit.normal), 1.0);
    const double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
    const bool cannotRefract = refractionRatio * sinTheta > 1.0;

    Vec3 dir;
    if (cannotRefract || schlick(cosTheta, refractionRatio) > rng.next01()) {
      dir = reflect(unitDir, hit.normal);
    } else {
      dir = refract(unitDir, hit.normal, refractionRatio);
    }
    ray = Ray{hit.p, dir};
  }

  return Vec3(0.0, 0.0, 0.0);
}

static uint64_t mix64(uint64_t x) {
  x ^= x >> 33U;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33U;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33U;
  return x;
}

static int toByte(double x) {
  x = std::clamp(x, 0.0, 0.999);
  return static_cast<int>(256.0 * x);
}

struct Params {
  int width = 960;
  int height = 540;
  int spp = 64;
  int maxDepth = 12;
  int smallSpheres = 120;
  uint64_t seed = 12345;
  std::string output = "raytrace.ppm";
};

static void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " [--width N] [--height N] [--spp N] [--max-depth N]"
               " [--small-spheres N] [--seed N] [--output FILE]"
            << '\n';
}

static bool parseInt(const char *text, int &value) {
  try {
    const std::string s(text);
    size_t pos = 0;
    const long long v = std::stoll(s, &pos, 10);
    if (pos != s.size() || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max()) {
      return false;
    }
    value = static_cast<int>(v);
    return true;
  } catch (...) {
    return false;
  }
}

static bool parseU64(const char *text, uint64_t &value) {
  try {
    const std::string s(text);
    size_t pos = 0;
    const unsigned long long v = std::stoull(s, &pos, 10);
    if (pos != s.size()) {
      return false;
    }
    value = static_cast<uint64_t>(v);
    return true;
  } catch (...) {
    return false;
  }
}

static bool parseArgs(int argc, char **argv, Params &p) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&](const std::string &name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--width") {
      const char *v = needValue(arg);
      if (!v || !parseInt(v, p.width)) {
        return false;
      }
    } else if (arg == "--height") {
      const char *v = needValue(arg);
      if (!v || !parseInt(v, p.height)) {
        return false;
      }
    } else if (arg == "--spp") {
      const char *v = needValue(arg);
      if (!v || !parseInt(v, p.spp)) {
        return false;
      }
    } else if (arg == "--max-depth") {
      const char *v = needValue(arg);
      if (!v || !parseInt(v, p.maxDepth)) {
        return false;
      }
    } else if (arg == "--small-spheres") {
      const char *v = needValue(arg);
      if (!v || !parseInt(v, p.smallSpheres)) {
        return false;
      }
    } else if (arg == "--seed") {
      const char *v = needValue(arg);
      if (!v || !parseU64(v, p.seed)) {
        return false;
      }
    } else if (arg == "--output") {
      const char *v = needValue(arg);
      if (!v) {
        return false;
      }
      p.output = v;
    } else if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      return false;
    }
  }

  if (p.width <= 0 || p.height <= 0 || p.spp <= 0 || p.maxDepth <= 0 ||
      p.smallSpheres < 0) {
    std::cerr << "All size/depth/sample parameters must be positive (small-spheres >= 0).\n";
    return false;
  }

  return true;
}

static std::vector<Sphere> makeScene(int smallSpheres, uint64_t seed) {
  std::vector<Sphere> world;
  world.reserve(static_cast<size_t>(smallSpheres) + 8);

  world.push_back(Sphere{Vec3(0.0, -1000.0, -1.0), 1000.0,
                         Material{MaterialType::Lambertian, Vec3(0.5, 0.5, 0.5),
                                  0.0, 1.0}});

  world.push_back(Sphere{Vec3(0.0, 1.0, -3.0), 1.0,
                         Material{MaterialType::Dielectric, Vec3(1.0, 1.0, 1.0),
                                  0.0, 1.5}});
  world.push_back(Sphere{Vec3(-2.0, 1.0, -4.0), 1.0,
                         Material{MaterialType::Lambertian, Vec3(0.7, 0.2, 0.2),
                                  0.0, 1.0}});
  world.push_back(Sphere{Vec3(2.0, 1.0, -4.0), 1.0,
                         Material{MaterialType::Metal, Vec3(0.8, 0.8, 0.9), 0.05,
                                  1.0}});

  Rng rng(seed ^ 0xa5a5a5a5a5a5a5a5ULL);
  for (int i = 0; i < smallSpheres; ++i) {
    const double x = -8.0 + 16.0 * rng.next01();
    const double z = -10.0 - 10.0 * rng.next01();
    const double r = 0.12 + 0.18 * rng.next01();
    const Vec3 center(x, r, z);

    if (length(center - Vec3(0.0, 1.0, -3.0)) < 1.3 ||
        length(center - Vec3(-2.0, 1.0, -4.0)) < 1.3 ||
        length(center - Vec3(2.0, 1.0, -4.0)) < 1.3) {
      continue;
    }

    const double choose = rng.next01();
    if (choose < 0.7) {
      Vec3 albedo(0.2 + 0.8 * rng.next01(), 0.2 + 0.8 * rng.next01(),
                  0.2 + 0.8 * rng.next01());
      world.push_back(
          Sphere{center, r, Material{MaterialType::Lambertian, albedo, 0.0, 1.0}});
    } else if (choose < 0.9) {
      Vec3 albedo(0.5 + 0.5 * rng.next01(), 0.5 + 0.5 * rng.next01(),
                  0.5 + 0.5 * rng.next01());
      const double fuzz = 0.02 + 0.28 * rng.next01();
      world.push_back(
          Sphere{center, r, Material{MaterialType::Metal, albedo, fuzz, 1.0}});
    } else {
      world.push_back(Sphere{center, r,
                             Material{MaterialType::Dielectric,
                                      Vec3(1.0, 1.0, 1.0), 0.0, 1.5}});
    }
  }

  return world;
}

int main(int argc, char **argv) {
  Params p;
  if (!parseArgs(argc, argv, p)) {
    printUsage(argv[0]);
    return 1;
  }

  const double aspect = static_cast<double>(p.width) / static_cast<double>(p.height);
  const Camera cam(35.0, aspect);
  const std::vector<Sphere> world = makeScene(p.smallSpheres, p.seed);

  std::vector<unsigned char> image(static_cast<size_t>(p.width) * p.height * 3);
  uint64_t checksum = 0;
  double luminanceSum = 0.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int y = p.height - 1; y >= 0; --y) {
    for (int x = 0; x < p.width; ++x) {
      Vec3 color(0.0, 0.0, 0.0);

      const uint64_t pixelId = static_cast<uint64_t>(y) * static_cast<uint64_t>(p.width) +
                               static_cast<uint64_t>(x);
      Rng rng(mix64(p.seed ^ (pixelId + 1ULL) * 0x9e3779b97f4a7c15ULL));

      for (int s = 0; s < p.spp; ++s) {
        const double u =
            (static_cast<double>(x) + rng.next01()) / static_cast<double>(p.width - 1);
        const double v =
            (static_cast<double>(y) + rng.next01()) / static_cast<double>(p.height - 1);
        const Ray r = cam.getRay(u, v);
        color += trace(r, world, p.maxDepth, rng);
      }

      color /= static_cast<double>(p.spp);
      luminanceSum += 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z;

      const double r = std::sqrt(std::max(0.0, color.x));
      const double g = std::sqrt(std::max(0.0, color.y));
      const double b = std::sqrt(std::max(0.0, color.z));

      const int ir = toByte(r);
      const int ig = toByte(g);
      const int ib = toByte(b);

      const int outY = p.height - 1 - y;
      const size_t idx = (static_cast<size_t>(outY) * p.width + static_cast<size_t>(x)) * 3;
      image[idx + 0] = static_cast<unsigned char>(ir);
      image[idx + 1] = static_cast<unsigned char>(ig);
      image[idx + 2] = static_cast<unsigned char>(ib);

      uint64_t px = (static_cast<uint64_t>(ir) << 16U) |
                    (static_cast<uint64_t>(ig) << 8U) | static_cast<uint64_t>(ib);
      checksum ^= mix64(px + pixelId * 0x517cc1b727220a95ULL);
      checksum = (checksum << 1U) | (checksum >> 63U);
    }
  }
  const auto t1 = std::chrono::steady_clock::now();

  std::ofstream ofs(p.output, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output file: " << p.output << '\n';
    return 2;
  }
  ofs << "P6\n" << p.width << " " << p.height << "\n255\n";
  ofs.write(reinterpret_cast<const char *>(image.data()),
            static_cast<std::streamsize>(image.size()));
  if (!ofs) {
    std::cerr << "Failed to write output file: " << p.output << '\n';
    return 3;
  }

  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double avgLum = luminanceSum / static_cast<double>(p.width * p.height);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "render_ms=" << ms << '\n';
  std::cout << "avg_luminance=" << avgLum << '\n';
  std::cout << "checksum=0x" << std::hex << checksum << std::dec << '\n';
  std::cout << "spheres=" << world.size() << '\n';
  std::cout << "output=" << p.output << '\n';

  return 0;
}
