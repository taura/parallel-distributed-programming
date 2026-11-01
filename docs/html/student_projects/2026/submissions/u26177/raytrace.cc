#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec3 {
  double x;
  double y;
  double z;

  Vec3() : x(0.0), y(0.0), z(0.0) {}
  Vec3(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}

  Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
  Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
  Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
  Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
};

static Vec3 mul(const Vec3& a, const Vec3& b) {
  return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static double length(const Vec3& v) {
  return std::sqrt(dot(v, v));
}

static Vec3 normalize(const Vec3& v) {
  const double len = length(v);
  if (len == 0.0) {
    return Vec3(0.0, 0.0, 0.0);
  }
  return v / len;
}

static Vec3 reflect(const Vec3& v, const Vec3& n) {
  return v - n * (2.0 * dot(v, n));
}

struct Ray {
  Vec3 origin;
  Vec3 dir;
};

struct Material {
  Vec3 color;
  double ambient;
  double diffuse;
  double specular;
  double shininess;
  double reflectivity;
  double ior = 0.0;  // Path mode: >1 denotes a solid, clear dielectric in air.
  double metallic = 0.0;  // GGX mode: 0 = dielectric, 1 = conductor.
  double roughness = 0.4;
};

struct Sphere {
  Vec3 center;
  double radius;
  int material_id;
};

struct Box {
  Vec3 lower;
  Vec3 upper;
  int material_id;
};

struct Ellipsoid {
  Vec3 center;
  Vec3 radii;
  int material_id;
  double tilt = 0.0;  // Rotation about world Z, in radians.
};

struct Triangle {
  Vec3 a, b, c;
  int material_id;
  Vec3 na, nb, nc;  // Zero normals select flat shading; otherwise interpolate.
};

struct Light {
  Vec3 position;
  Vec3 color;
  double intensity;
  double radius;  // Half-width of a horizontal square light.
};

struct Hit {
  bool hit;
  double t;
  Vec3 point;
  Vec3 normal;
  int material_id;
};

static Hit intersect_sphere(const Ray& ray, const Sphere& s) {
  const Vec3 oc = ray.origin - s.center;
  const double a = dot(ray.dir, ray.dir);
  const double b = 2.0 * dot(oc, ray.dir);
  const double c = dot(oc, oc) - s.radius * s.radius;
  const double disc = b * b - 4.0 * a * c;

  if (disc < 0.0) {
    return {false, 0.0, Vec3(), Vec3(), -1};
  }

  const double sqrt_disc = std::sqrt(disc);
  const double t1 = (-b - sqrt_disc) / (2.0 * a);
  const double t2 = (-b + sqrt_disc) / (2.0 * a);

  double t = std::numeric_limits<double>::max();
  if (t1 > 1e-6) {
    t = t1;
  } else if (t2 > 1e-6) {
    t = t2;
  } else {
    return {false, 0.0, Vec3(), Vec3(), -1};
  }

  const Vec3 p = ray.origin + ray.dir * t;
  const Vec3 n = normalize(p - s.center);
  return {true, t, p, n, s.material_id};
}

static Hit intersect_ellipsoid(const Ray& ray, const Ellipsoid& e) {
  const double c = std::cos(e.tilt);
  const double s = std::sin(e.tilt);
  const auto to_local = [&](const Vec3& v) {
    return Vec3((c * v.x + s * v.y) / e.radii.x,
                (-s * v.x + c * v.y) / e.radii.y, v.z / e.radii.z);
  };
  // Transform to a unit sphere. Do not normalize the transformed direction:
  // retaining its length preserves the world ray's distance parameter t.
  const Ray local{to_local(ray.origin - e.center), to_local(ray.dir)};
  const Hit h = intersect_sphere(local, {Vec3(), 1.0, e.material_id});
  if (!h.hit) {
    return h;
  }
  // Surface normals transform by the inverse transpose, not like positions.
  const Vec3 n(h.normal.x / e.radii.x, h.normal.y / e.radii.y, h.normal.z / e.radii.z);
  return {true, h.t, ray.origin + ray.dir * h.t,
          normalize(Vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z)), e.material_id};
}

static Hit intersect_box(const Ray& ray, const Box& box) {
  // Intersect the three axis-aligned slabs. Track both entry and exit normals
  // so rays originating inside a box also return the correct surface.
  const double origins[] = {ray.origin.x, ray.origin.y, ray.origin.z};
  const double directions[] = {ray.dir.x, ray.dir.y, ray.dir.z};
  const double lower[] = {box.lower.x, box.lower.y, box.lower.z};
  const double upper[] = {box.upper.x, box.upper.y, box.upper.z};
  const Vec3 axes[] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
  const Hit miss{false, 0.0, Vec3(), Vec3(), -1};
  double enter = -std::numeric_limits<double>::infinity();
  double exit = std::numeric_limits<double>::infinity();
  Vec3 enter_normal, exit_normal;
  for (int axis = 0; axis < 3; ++axis) {
    if (directions[axis] == 0.0) {
      if (origins[axis] < lower[axis] || origins[axis] > upper[axis]) {
        return miss;
      }
      continue;
    }
    double near = (lower[axis] - origins[axis]) / directions[axis];
    double far = (upper[axis] - origins[axis]) / directions[axis];
    Vec3 near_normal = axes[axis] * -1.0;
    Vec3 far_normal = axes[axis];
    if (near > far) {
      std::swap(near, far);
      std::swap(near_normal, far_normal);
    }
    if (near > enter) {
      enter = near;
      enter_normal = near_normal;
    }
    if (far < exit) {
      exit = far;
      exit_normal = far_normal;
    }
    if (enter > exit) {
      return miss;
    }
  }
  if (exit <= 1e-6 || !std::isfinite(exit)) {
    return miss;
  }
  const bool outside = enter > 1e-6;
  const double t = outside ? enter : exit;
  return {true, t, ray.origin + ray.dir * t, outside ? enter_normal : exit_normal, box.material_id};
}

static Hit intersect_triangle(const Ray& ray, const Triangle& tri) {
  // Moller-Trumbore: solve for distance and barycentric coordinates.
  const Hit miss{false, 0.0, Vec3(), Vec3(), -1};
  const Vec3 e1 = tri.b - tri.a;
  const Vec3 e2 = tri.c - tri.a;
  const Vec3 p = cross(ray.dir, e2);
  const double det = dot(e1, p);
  if (std::abs(det) < 1e-12) {
    return miss;
  }
  const Vec3 s = ray.origin - tri.a;
  const double u = dot(s, p) / det;
  if (u < -1e-10 || u > 1.0 + 1e-10) {
    return miss;
  }
  const Vec3 q = cross(s, e1);
  const double v = dot(ray.dir, q) / det;
  if (v < -1e-10 || u + v > 1.0 + 1e-10) {
    return miss;
  }
  const double t = dot(e2, q) / det;
  if (t <= 1e-6) {
    return miss;
  }
  const Vec3 geometric_normal = normalize(cross(e1, e2));
  Vec3 n = normalize(tri.na * (1.0 - u - v) + tri.nb * u + tri.nc * v);
  if (dot(n, n) == 0.0) {
    n = geometric_normal;
  }
  // Cloth and decorative mesh surfaces are opaque and two-sided.
  if (dot(n, ray.dir) > 0.0) {
    n = n * -1.0;
  }
  return {true, t, ray.origin + ray.dir * t, n, tri.material_id};
}

static Hit intersect_scene(const Ray& ray, const std::vector<Sphere>& spheres,
                           const std::vector<Box>& boxes,
                           const std::vector<Ellipsoid>& ellipsoids = {},
                           const std::vector<Triangle>& triangles = {}) {
  Hit best{false, std::numeric_limits<double>::max(), Vec3(), Vec3(), -1};
  for (const Sphere& s : spheres) {
    const Hit h = intersect_sphere(ray, s);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }
  for (const Box& box : boxes) {
    const Hit h = intersect_box(ray, box);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }
  for (const Ellipsoid& e : ellipsoids) {
    const Hit h = intersect_ellipsoid(ray, e);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }
  for (const Triangle& tri : triangles) {
    const Hit h = intersect_triangle(ray, tri);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }
  // Infinite ground plane, y = -2, using material 3.
  if (std::abs(ray.dir.y) > 1e-12) {
    const double t = (-2.0 - ray.origin.y) / ray.dir.y;
    if (t > 1e-6 && t < best.t) {
      best = {true, t, ray.origin + ray.dir * t, Vec3(0.0, 1.0, 0.0), 3};
    }
  }
  return best;
}

static bool occluded(const Vec3& p, const Vec3& ldir, double max_dist,
                     const std::vector<Sphere>& spheres, const std::vector<Box>& boxes,
                     const std::vector<Ellipsoid>& ellipsoids = {},
                     const std::vector<Triangle>& triangles = {}) {
  const Ray shadow{p + ldir * 1e-4, ldir};
  const Hit h = intersect_scene(shadow, spheres, boxes, ellipsoids, triangles);
  return h.hit && h.t < max_dist;
}

static Vec3 sky_color(const Vec3& dir) {
  const double t = 0.5 * (dir.y + 1.0);
  return Vec3(0.78, 0.88, 1.0) * (1.0 - t) + Vec3(0.20, 0.33, 0.62) * t;
}

static Material surface_material(const Hit& hit, const std::vector<Material>& materials) {
  Material m = materials[hit.material_id];
  if (hit.material_id == 3) {
    const double tile = std::floor(hit.point.x) + std::floor(hit.point.z);
    if (std::fmod(tile, 2.0) != 0.0) {
      m.color = Vec3(0.12, 0.16, 0.22);
    }
  }
  return m;
}

static Vec3 shade_direct(const Hit& hit,
                         const Ray& ray,
                         const std::vector<Sphere>& spheres,
                         const std::vector<Box>& boxes,
                         const std::vector<Ellipsoid>& ellipsoids,
                         const std::vector<Triangle>& triangles,
                         const std::vector<Material>& materials,
                         const std::vector<Light>& lights,
                         int light_samples,
                         std::mt19937& rng) {
  const Material m = surface_material(hit, materials);
  Vec3 c = m.color * m.ambient;
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  for (const Light& light : lights) {
    // Average point-light contributions over the emitter: a simple soft-shadow
    // model with fixed total intensity (not a physically based area-light model).
    const int samples = light.radius == 0.0 ? 1 : light_samples;
    for (int sample = 0; sample < samples; ++sample) {
      const double dx = light.radius * (2.0 * unif(rng) - 1.0);
      const double dz = light.radius * (2.0 * unif(rng) - 1.0);
      const Vec3 to_light_vec = light.position + Vec3(dx, 0.0, dz) - hit.point;
      const double dist_to_light = length(to_light_vec);
      if (dist_to_light <= 1e-4) {
        continue;
      }
      const Vec3 to_light = to_light_vec / dist_to_light;
      const double ndotl = dot(hit.normal, to_light);
      if (ndotl <= 0.0 || occluded(hit.point, to_light, dist_to_light - 1e-4, spheres, boxes, ellipsoids, triangles)) {
        continue;
      }

      const Vec3 diffuse = mul(m.color * (m.diffuse * ndotl), light.color) * light.intensity;
      const Vec3 view_dir = normalize(ray.origin - hit.point);
      const Vec3 reflect_dir = reflect(to_light * -1.0, hit.normal);
      const double spec = std::pow(std::max(0.0, dot(view_dir, reflect_dir)), m.shininess);
      const Vec3 specular = light.color * (m.specular * spec * light.intensity);
      c = c + (diffuse + specular) / static_cast<double>(samples);
    }
  }

  return c;
}

static Vec3 trace_ray(const Ray& input_ray,
                      const std::vector<Sphere>& spheres,
                      const std::vector<Box>& boxes,
                      const std::vector<Ellipsoid>& ellipsoids,
                      const std::vector<Triangle>& triangles,
                      const std::vector<Material>& materials,
                      const std::vector<Light>& lights,
                      int max_depth,
                      int light_samples,
                      std::mt19937& rng,
                      std::uint64_t& primary_hits,
                      bool count_primary) {
  Vec3 throughput(1.0, 1.0, 1.0);
  Vec3 radiance(0.0, 0.0, 0.0);
  Ray ray = input_ray;

  for (int depth = 0; depth <= max_depth; ++depth) {
    const Hit hit = intersect_scene(ray, spheres, boxes, ellipsoids, triangles);
    if (!hit.hit) {
      radiance = radiance + mul(throughput, sky_color(ray.dir));
      break;
    }

    if (count_primary && depth == 0) {
      ++primary_hits;
    }

    const Material& m = materials[hit.material_id];
    const Vec3 local = shade_direct(hit, ray, spheres, boxes, ellipsoids, triangles, materials, lights, light_samples, rng);
    // At the depth limit the remaining reflected contribution is truncated.
    radiance = radiance + mul(throughput, local) * (1.0 - m.reflectivity);

    if (m.reflectivity <= 0.0 || depth == max_depth) {
      break;
    }

    throughput = throughput * m.reflectivity;
    const Vec3 new_dir = normalize(reflect(ray.dir, hit.normal));
    ray = Ray{hit.point + new_dir * 1e-4, new_dir};
  }

  return radiance;
}

#include "path_trace.h"

static uint8_t to_u8(double x) {
  if (x < 0.0) {
    x = 0.0;
  }
  if (x > 1.0) {
    x = 1.0;
  }
  return static_cast<uint8_t>(x * 255.0 + 0.5);
}

static Vec3 tonemap_gamma(Vec3 c) {
  c.x = c.x / (1.0 + c.x);
  c.y = c.y / (1.0 + c.y);
  c.z = c.z / (1.0 + c.z);
  c.x = std::sqrt(c.x);
  c.y = std::sqrt(c.y);
  c.z = std::sqrt(c.z);
  return c;
}

static int parse_int(const std::string& text) {
  size_t used = 0;
  const int value = std::stoi(text, &used);
  if (used != text.size()) {
    throw std::invalid_argument("Invalid integer: " + text);
  }
  return value;
}

static void make_gallery(std::vector<Sphere>& spheres, std::vector<Box>& boxes,
                         std::vector<Material>& materials, int objects, std::uint32_t seed) {
  const int stone = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.78, 0.72, 0.60), 0.10, 0.85, 0.12, 20.0, 0.03});
  const int dark = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.07, 0.10, 0.15), 0.08, 0.75, 0.5, 60.0, 0.18});
  const int gold = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.95, 0.58, 0.13), 0.08, 0.8, 0.6, 90.0, 0.35, 0.0, 1.0, 0.24});
  const int wall = static_cast<int>(materials.size());
  materials.push_back({Vec3(0.30, 0.47, 0.56), 0.12, 0.85, 0.1, 16.0, 0.0});

  // Open-front gallery: back wall, framed mirror and two rows of columns.
  boxes.push_back({Vec3(-6.7, -2.0, -13.0), Vec3(6.7, 4.4, -12.7), wall});
  boxes.push_back({Vec3(-2.9, -0.9, -12.65), Vec3(2.9, 3.6, -12.4), gold});
  boxes.push_back({Vec3(-2.65, -0.65, -12.39), Vec3(2.65, 3.35, -12.3), 4});
  for (double z : {-10.5, -5.5}) {
    for (double x : {-6.0, 6.0}) {
      boxes.push_back({Vec3(x - 0.65, -2.0, z - 0.65), Vec3(x + 0.65, -1.6, z + 0.65), dark});
      boxes.push_back({Vec3(x - 0.30, -1.6, z - 0.30), Vec3(x + 0.30, 3.7, z + 0.30), stone});
      boxes.push_back({Vec3(x - 0.55, 3.7, z - 0.55), Vec3(x + 0.55, 4.05, z + 0.55), gold});
    }
    boxes.push_back({Vec3(-6.55, 4.05, z - 0.35), Vec3(6.55, 4.45, z + 0.35), stone});
  }

  // Three stacked slabs form a stepped base for a floating spiral sculpture.
  for (int step = 0; step < 3; ++step) {
    const double half = 2.7 - 0.35 * step;
    boxes.push_back({Vec3(-half, -2.0 + 0.18 * step, -7.5 - half),
                     Vec3(half, -1.82 + 0.18 * step, -7.5 + half), step == 1 ? gold : dark});
  }
  spheres = {
      {Vec3(0.0, 0.04, -7.5), 1.5, 4},  // central mirror sphere
      {Vec3(-4.0, -0.4, -4.0), 0.8, 0},
      {Vec3(4.0, -0.25, -5.0), 0.95, 1},
      {Vec3(0.0, -0.6, -1.6), 0.6, 2},
  };
  for (size_t i = 1; i < spheres.size(); ++i) {
    const Sphere& s = spheres[i];
    const double half = s.radius + 0.2;
    boxes.push_back({Vec3(s.center.x - half, -2.0, s.center.z - half),
                     Vec3(s.center.x + half, s.center.y - s.radius, s.center.z + half), stone});
  }

  // Side installations: descending stacks with gold caps.
  for (double side : {-1.0, 1.0}) {
    for (int i = 0; i < 3; ++i) {
      const double x = side * (3.6 + 0.7 * i);
      const double top = 1.7 - 0.7 * i;
      boxes.push_back({Vec3(x - 0.28, -2.0, -11.3), Vec3(x + 0.28, top, -10.5), stone});
      boxes.push_back({Vec3(x - 0.32, top, -11.34), Vec3(x + 0.32, top + 0.12, -10.46), gold});
    }
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  const double pi = 3.14159265358979323846;
  const int beads = objects - 4;
  // Radius decreases at higher counts so the beads stay distinct.
  const double bead_radius = std::min(0.23, 10.0 / std::max(1, beads));
  for (int i = 0; i < beads; ++i) {
    const double t = (i + 0.5) / std::max(1, beads);
    const double angle = 4.0 * pi * t;
    const double red = 0.2 + 0.75 * unif(rng);
    const double green = 0.2 + 0.75 * unif(rng);
    const double blue = 0.2 + 0.75 * unif(rng);
    materials.push_back({Vec3(red, green, blue), 0.08, 0.8, 0.55, 70.0, 0.25});
    spheres.push_back({Vec3(2.15 * std::cos(angle), -0.9 + 3.6 * t,
                            -7.5 + 2.15 * std::sin(angle)),
                       bead_radius, static_cast<int>(materials.size() - 1)});
  }
}

#include "character_scene.h"

static void check_geometry() {
  const Box box{Vec3(-1, -1, -1), Vec3(1, 1, 1), 0};
  const Vec3 axes[] = {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)};
  for (const Vec3& axis : axes) {
    for (double sign : {-1.0, 1.0}) {
      const Vec3 normal = axis * sign;
      const Hit outside = intersect_box({normal * 3.0, normal * -1.0}, box);
      const Hit inside = intersect_box({Vec3(), normal}, box);
      if (!outside.hit || std::abs(outside.t - 2.0) > 1e-12 || length(outside.normal - normal) > 1e-12 ||
          !inside.hit || std::abs(inside.t - 1.0) > 1e-12 || length(inside.normal - normal) > 1e-12) {
        throw std::runtime_error("Box entry/exit distance or normal check failed");
      }
    }
  }
  if (intersect_box({Vec3(2, 0, 3), Vec3(0, 0, -1)}, box).hit ||
      intersect_box({Vec3(0, 0, 3), Vec3(0, 0, 1)}, box).hit ||
      !intersect_box({Vec3(1, 0, 3), Vec3(0, 0, -1)}, box).hit) {
    throw std::runtime_error("Box parallel/boundary/behind check failed");
  }
  const std::vector<Sphere> spheres = {{Vec3(0, 0, -5), 1.0, 1}};
  const std::vector<Box> boxes = {{Vec3(-1, -1, -3), Vec3(1, 1, -2), 0}};
  const Ray ray{Vec3(), Vec3(0, 0, -1)};
  const Hit nearest = intersect_scene(ray, spheres, boxes);
  if (!nearest.hit || std::abs(nearest.t - 2.0) > 1e-12 || nearest.material_id != 0 ||
      occluded(ray.origin, ray.dir, 1.0, spheres, boxes) ||
      !occluded(ray.origin, ray.dir, 3.0, spheres, boxes)) {
    throw std::runtime_error("Nearest-hit/shadow-distance check failed");
  }
  // Nonuniform scaling and rotation must preserve t and transform normals.
  const Ellipsoid e{Vec3(0, 0, -5), Vec3(2, 1, 0.5), 0};
  const Hit front = intersect_ellipsoid(ray, e);
  const Hit inside = intersect_ellipsoid({e.center, Vec3(1, 0, 0)}, e);
  const double pi = 3.14159265358979323846;
  Ellipsoid rotated = e;
  rotated.tilt = pi / 2.0;
  const Hit top = intersect_ellipsoid({Vec3(0, 4, -5), Vec3(0, -1, 0)}, rotated);
  const double q = std::sqrt(0.5);
  const Vec3 surface = e.center + Vec3(2 * q, q, 0);
  const Vec3 expected_normal = normalize(Vec3(q / 2, q, 0));
  const Hit oblique = intersect_ellipsoid({surface + expected_normal * 3.0, expected_normal * -1.0}, e);
  if (!front.hit || std::abs(front.t - 4.5) > 1e-10 ||
      !inside.hit || std::abs(inside.t - 2.0) > 1e-10 ||
      !top.hit || std::abs(top.t - 2.0) > 1e-10 || length(top.normal - Vec3(0, 1, 0)) > 1e-10 ||
      !oblique.hit || std::abs(oblique.t - 3.0) > 1e-10 || length(oblique.normal - expected_normal) > 1e-10 ||
      intersect_ellipsoid({Vec3(3, 0, 0), Vec3(0, 0, -1)}, e).hit ||
      occluded(ray.origin, ray.dir, 4.0, {}, {}, {e}) ||
      !occluded(ray.origin, ray.dir, 5.0, {}, {}, {e})) {
    throw std::runtime_error("Ellipsoid distance/normal/rotation/shadow check failed");
  }
  const Triangle tri{Vec3(-1, -1, -2), Vec3(1, -1, -2), Vec3(0, 1, -2), 0, {}, {}, {}};
  const Hit triangle_front = intersect_triangle(ray, tri);
  const Hit triangle_back = intersect_triangle({Vec3(0, 0, -4), Vec3(0, 0, 1)}, tri);
  Triangle smooth = tri;
  smooth.na = smooth.nb = Vec3(0, 0, 1);
  smooth.nc = normalize(Vec3(0, 1, 1));
  const Hit interpolated = intersect_triangle(ray, smooth);
  const Hit mixed = intersect_scene(ray, spheres, {}, {e}, {tri});
  if (!triangle_front.hit || std::abs(triangle_front.t - 2.0) > 1e-10 ||
      !triangle_back.hit || triangle_back.normal.z > -0.99 ||
      !interpolated.hit || length(interpolated.normal - normalize(smooth.na + smooth.nc)) > 1e-10 ||
      intersect_triangle({Vec3(2, 0, 0), ray.dir}, tri).hit ||
      intersect_triangle({Vec3(), Vec3(1, 0, 0)}, tri).hit ||
      intersect_triangle({Vec3(), Vec3(0, 0, 1)}, tri).hit ||
      intersect_triangle(ray, {tri.a, tri.a, tri.c, 0, {}, {}, {}}).hit ||
      !intersect_triangle({Vec3(0, -1, 0), ray.dir}, tri).hit ||
      !mixed.hit || std::abs(mixed.t - 2.0) > 1e-10 ||
      occluded(ray.origin, ray.dir, 1.0, {}, {}, {}, {tri}) ||
      !occluded(ray.origin, ray.dir, 3.0, {}, {}, {}, {tri})) {
    throw std::runtime_error("Triangle barycentric/normal/two-sided/shadow check failed");
  }
  std::cout << "Geometry checks passed (boxes, ellipsoids, triangles, closest hits, shadows)\n";
}

struct SampleMoments {
  int count = 0;
  Vec3 mean, m2;

  void add(const Vec3& value) {
    ++count;
    const Vec3 delta = value - mean;
    mean = mean + delta / static_cast<double>(count);
    m2 = m2 + mul(delta, value - mean);
  }

  bool stable(double tolerance) const {
    if (count < 2) return false;
    const auto channel_stable = [&](double average, double squared_deviations) {
      const double error = 1.96 * std::sqrt(std::max(0.0, squared_deviations) /
                                           (static_cast<double>(count) * (count - 1)));
      return error <= tolerance * std::max(std::abs(average), 0.05);
    };
    return channel_stable(mean.x, m2.x) && channel_stable(mean.y, m2.y) && channel_stable(mean.z, m2.z);
  }
};

static void check_sampling() {
  SampleMoments constant, varying;
  for (int i = 0; i < 16; ++i) {
    constant.add(Vec3(0.5, 0.25, 0.1));
    varying.add(Vec3(i % 2 == 0 ? 0 : 2, 0.25, 0.1));
  }
  if (!constant.stable(0.01) || varying.stable(0.1) ||
      std::abs(varying.mean.x - 1.0) > 1e-12 || std::abs(varying.m2.x - 16.0) > 1e-12) {
    throw std::runtime_error("Adaptive RGB mean/variance/stopping check failed");
  }
  std::cout << "Adaptive sampling checks passed: constant and varying RGB samples\n";
}

static void write_ppm(const std::string& filename, int width, int height, const std::vector<uint8_t>& pixels) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) throw std::runtime_error("Cannot open output file: " + filename);
  ofs << "P6\n" << width << " " << height << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  ofs.close();
  if (!ofs) throw std::runtime_error("Failed to write image: " + filename);
}

int main(int argc, char** argv) try {
  int width = 960;
  int height = 540;
  int spp = 4;
  int max_depth = 2;
  bool depth_specified = false;
  PathOptions path_options;
  std::string integrator = "whitted";
  double adaptive_tolerance = 0.0;
  int min_spp = 16;
  std::string samples_output;
  int objects = 64;
  int detail = 2;
  int light_samples = 4;
  double light_radius = 1.0;
  std::uint32_t seed = 42;
  std::string output = "image.ppm";
  std::string scene = "gallery";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::cout << "Usage: ./raytrace [--w W] [--h H] [--spp N] [--max-depth D]\n"
                   "  [--objects N] [--light-samples N] [--light-radius R]\n"
                   "  [--seed S] [--out FILE.ppm] [--scene gallery|spheres|character|transport]\n"
                   "  [--integrator whitted|path] [--nee 0|1] [--rr-depth N] [--env X]\n"
                   "  [--bsdf ideal|ggx] [--roughness R] [--adaptive T] [--min-spp N] [--samples-out FILE.ppm]\n"
                   "Defaults: 960x540, spp=4, depth=2, objects=64, light-samples=4, radius=1, seed=42\n"
                   "Default scene: gallery. objects counts spheres only, excluding boxes and ground.\n"
                   "In character scene, objects sets the number of stage decoration beads.\n"
                   "--detail 1..8: character mesh detail (default 2); higher values add triangles.\n"
                   "Path defaults: max-depth=8, nee=1 (MIS), rr-depth=3, env=0.15.\n"
                   "BSDF default: ideal. GGX roughness: per material, or override with 0.02..1.\n"
                   "Adaptive default: 0 (off). If >0, spp is a cap, min-spp>=8 (default 16).\n"
                   "depth=0: direct lighting only; radius=0: hard shadows.\n"
                   "--check-geometry, --check-transport or --check-advanced runs checks and exits.\n";
      return 0;
    }
    if (arg == "--check-geometry") {
      check_geometry();
      return 0;
    }
    if (arg == "--check-transport") {
      check_transport();
      return 0;
    }
    if (arg == "--check-advanced") {
      check_microfacet();
      check_ggx_transport();
      check_sampling();
      return 0;
    }
    if (i + 1 == argc) {
      throw std::invalid_argument("Missing value for " + arg);
    }
    const std::string value = argv[++i];
    if (arg == "--w") {
      width = parse_int(value);
    } else if (arg == "--h") {
      height = parse_int(value);
    } else if (arg == "--spp") {
      spp = parse_int(value);
    } else if (arg == "--max-depth") {
      max_depth = parse_int(value);
      depth_specified = true;
    } else if (arg == "--integrator") {
      integrator = value;
    } else if (arg == "--bsdf") {
      if (value != "ideal" && value != "ggx") throw std::invalid_argument("bsdf must be ideal or ggx");
      path_options.ggx = value == "ggx";
    } else if (arg == "--roughness") {
      size_t used = 0;
      path_options.roughness = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(path_options.roughness) ||
          path_options.roughness < 0.02 || path_options.roughness > 1.0) {
        throw std::invalid_argument("roughness must be in [0.02, 1]");
      }
    } else if (arg == "--adaptive") {
      size_t used = 0;
      adaptive_tolerance = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(adaptive_tolerance) || adaptive_tolerance < 0.0 || adaptive_tolerance > 1.0) {
        throw std::invalid_argument("adaptive must be in [0, 1], with 0 disabling early stopping");
      }
    } else if (arg == "--min-spp") {
      min_spp = parse_int(value);
    } else if (arg == "--samples-out") {
      samples_output = value;
    } else if (arg == "--nee") {
      const int nee = parse_int(value);
      if (nee != 0 && nee != 1) {
        throw std::invalid_argument("nee must be 0 or 1");
      }
      path_options.nee = nee != 0;
    } else if (arg == "--rr-depth") {
      path_options.rr_depth = parse_int(value);
    } else if (arg == "--env") {
      size_t used = 0;
      path_options.environment = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(path_options.environment) || path_options.environment < 0.0) {
        throw std::invalid_argument("env must be finite and nonnegative");
      }
    } else if (arg == "--objects") {
      objects = parse_int(value);
    } else if (arg == "--detail") {
      detail = parse_int(value);
    } else if (arg == "--light-samples") {
      light_samples = parse_int(value);
    } else if (arg == "--light-radius") {
      size_t used = 0;
      light_radius = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(light_radius) || light_radius < 0.0) {
        throw std::invalid_argument("light-radius must be finite and >= 0");
      }
    } else if (arg == "--seed") {
      size_t used = 0;
      const auto parsed = std::stoull(value, &used);
      if (value.empty() || value.front() == '-' || used != value.size() ||
          parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("seed must be a 32-bit unsigned integer");
      }
      seed = static_cast<std::uint32_t>(parsed);
    } else if (arg == "--out") {
      output = value;
    } else if (arg == "--scene") {
      scene = value;
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }

  if (scene != "gallery" && scene != "spheres" && scene != "character" && scene != "transport") {
    throw std::invalid_argument("scene must be gallery, spheres, character or transport");
  }
  if (integrator != "whitted" && integrator != "path") {
    throw std::invalid_argument("integrator must be whitted or path");
  }
  if ((path_options.ggx || adaptive_tolerance > 0.0) && integrator != "path") {
    throw std::invalid_argument("GGX and adaptive sampling require --integrator path");
  }
  if (min_spp < 8 || (adaptive_tolerance > 0.0 && min_spp > spp)) {
    throw std::invalid_argument("Require min-spp>=8 and, for adaptive sampling, min-spp<=spp");
  }
  if (!samples_output.empty() && samples_output == output) {
    throw std::invalid_argument("samples-out and out must name different files");
  }
  if (path_options.rr_depth < 0) {
    throw std::invalid_argument("rr-depth must be nonnegative");
  }
  if (integrator == "path" && !depth_specified) {
    max_depth = 8;
  }
  if (detail < 1 || detail > 8) {
    throw std::invalid_argument("detail must be between 1 and 8");
  }

  if (width <= 0 || height <= 0 || spp <= 0 || max_depth < 0 ||
      max_depth == std::numeric_limits<int>::max() || objects < 4 || light_samples <= 0) {
    std::cerr << "Require width,height,spp,light-samples>0, objects>=4, 0<=depth<INT_MAX\n";
    return 1;
  }

  std::vector<Material> materials = {
      {Vec3(0.95, 0.24, 0.24), 0.08, 0.85, 0.25, 24.0, 0.15},
      {Vec3(0.25, 0.85, 0.35), 0.08, 0.88, 0.35, 40.0, 0.20},
      {Vec3(0.30, 0.45, 0.95), 0.08, 0.82, 0.50, 80.0, 0.45},
      {Vec3(0.90, 0.86, 0.65), 0.07, 0.92, 0.10, 8.0, 0.00},
      {Vec3(0.92, 0.92, 0.95), 0.02, 0.20, 0.95, 150.0, 0.85, 0.0, 1.0, 0.18},
  };

  std::vector<Sphere> spheres = {
      {Vec3(-1.2, -1.0, -4.2), 1.0, 0},
      {Vec3(1.2, -0.8, -5.2), 1.2, 1},
      {Vec3(0.1, -1.42, -3.2), 0.58, 2},
      {Vec3(2.8, -1.0, -6.8), 1.0, 4},
  };

  // Non-overlapping background spheres on a jittered grid. Scene randomness
  // is separate from rendering, so changing spp never changes the scene.
  std::mt19937 scene_rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  const int columns = std::max(1, static_cast<int>(std::ceil(std::sqrt(objects - 4.0))));
  for (int i = 0; scene == "spheres" && i < objects - 4; ++i) {
    const double radius = 0.25 + 0.2 * unif(scene_rng);
    const double x = (i % columns - (columns - 1) * 0.5) * 1.2 + 0.1 * (unif(scene_rng) - 0.5);
    const double z = -9.0 - (i / columns) * 1.2;
    const double red = 0.15 + 0.8 * unif(scene_rng);
    const double green = 0.15 + 0.8 * unif(scene_rng);
    const double blue = 0.15 + 0.8 * unif(scene_rng);
    const double reflection = 0.1 + 0.7 * unif(scene_rng);
    materials.push_back({Vec3(red, green, blue), 0.06, 0.8, 0.5, 60.0, reflection});
    spheres.push_back({Vec3(x, radius - 2.0, z), radius, static_cast<int>(materials.size() - 1)});
  }

  std::vector<Box> boxes;
  std::vector<Ellipsoid> ellipsoids;
  std::vector<Triangle> triangles;
  std::vector<Light> lights = {
      {Vec3(5.5, 6.2, -1.0), Vec3(1.00, 0.98, 0.95), 1.00, light_radius},
      {Vec3(-4.5, 3.0, -2.8), Vec3(0.70, 0.80, 1.00), 0.42, light_radius},
  };

  Vec3 camera(0.0, 0.15, 0.5);
  Vec3 target = camera + Vec3(0.0, 0.0, -1.0);
  double fov_deg = 58.0;
  if (scene == "gallery") {
    make_gallery(spheres, boxes, materials, objects, seed);
    camera = Vec3(7.0, 4.8, 8.0);
    target = Vec3(0.0, 0.7, -6.5);
    fov_deg = 44.0;
    lights = {
        {Vec3(-3.5, 7.5, -1.0), Vec3(1.0, 0.87, 0.68), 0.95, light_radius},
        {Vec3(5.0, 5.5, -7.0), Vec3(0.60, 0.78, 1.0), 0.75, light_radius},
        {Vec3(0.0, 3.6, -11.5), Vec3(1.0, 0.64, 0.36), 0.35, light_radius * 0.5},
    };
  }
  if (scene == "character") {
    spheres.clear();
    make_character(spheres, boxes, ellipsoids, triangles, materials, objects, detail);
    camera = Vec3(1.8, 2.0, 8.5);
    target = Vec3(0.0, 0.55, -5.0);
    fov_deg = 30.0;
    lights = {
        {Vec3(-3.5, 6.5, 0.0), Vec3(1.0, 0.88, 0.80), 0.85, light_radius},
        {Vec3(4.0, 3.5, 0.0), Vec3(0.72, 0.85, 1.0), 0.55, light_radius},
        {Vec3(0.0, 5.0, -7.0), Vec3(0.65, 1.0, 0.96), 0.60, light_radius},
    };
  }
  if (scene == "transport") {
    make_transport_scene(spheres, boxes, materials, lights, light_radius);
    camera = Vec3(0.0, 0.65, 4.0);
    target = Vec3(0.0, 0.45, -4.0);
    fov_deg = 48.0;
  }
  path_options.max_depth = max_depth;
  path_options.light_samples = light_samples;
  const Vec3 forward = normalize(target - camera);
  const Vec3 right = normalize(cross(forward, Vec3(0.0, 1.0, 0.0)));
  const Vec3 up = cross(right, forward);
  const double pi = 3.14159265358979323846;
  const double aspect = static_cast<double>(width) / static_cast<double>(height);
  const double scale = std::tan(0.5 * fov_deg * pi / 180.0);

  if (static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / 3 / static_cast<size_t>(height)) {
    throw std::invalid_argument("Image dimensions overflow the buffer size");
  }
  const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<uint8_t> pixels(pixel_count * 3);
  std::vector<uint8_t> sample_pixels(samples_output.empty() ? 0 : pixel_count * 3);

  const auto t0 = std::chrono::steady_clock::now();

  std::uint64_t primary_hits = 0;
  double luminance_sum = 0.0;
  PathStats path_stats;
  std::uint64_t total_samples = 0, stopped_pixels = 0;
  int minimum_samples = spp, maximum_samples = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Independent per-pixel RNG state also makes future pixel parallelism easy.
      std::seed_seq pixel_seed{seed, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)};
      std::mt19937 rng(pixel_seed);
      Vec3 acc(0.0, 0.0, 0.0);
      SampleMoments moments;
      int used_samples = 0;

      for (int s = 0; s < spp; ++s) {
        const double jx = unif(rng);
        const double jy = unif(rng);

        const double px = (2.0 * (x + jx) / static_cast<double>(width) - 1.0) * aspect * scale;
        const double py = (1.0 - 2.0 * (y + jy) / static_cast<double>(height)) * scale;
        const Ray ray{camera, normalize(forward + right * px + up * py)};

        const bool count_primary = (s == 0);
        Vec3 sample;
        if (integrator == "path") {
          sample = trace_path(ray, spheres, boxes, ellipsoids, triangles, materials,
                              lights, path_options, rng, path_stats, primary_hits, count_primary);
        } else {
          sample = trace_ray(ray, spheres, boxes, ellipsoids, triangles, materials, lights, max_depth,
                             light_samples, rng, primary_hits, count_primary);
        }
        acc = acc + sample;
        ++used_samples;
        if (adaptive_tolerance > 0.0) {
          moments.add(sample);
          // Check in batches, on linear RGB values before tone mapping.
          if (used_samples >= min_spp && (used_samples - min_spp) % 8 == 0 && moments.stable(adaptive_tolerance)) break;
        }
      }

      total_samples += used_samples;
      minimum_samples = std::min(minimum_samples, used_samples);
      maximum_samples = std::max(maximum_samples, used_samples);
      if (used_samples < spp) ++stopped_pixels;
      Vec3 c = acc / static_cast<double>(used_samples);
      if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z) ||
          c.x < 0.0 || c.y < 0.0 || c.z < 0.0) {
        throw std::runtime_error("Invalid radiance at pixel " + std::to_string(x) + "," + std::to_string(y));
      }
      luminance_sum += 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;

      c = tonemap_gamma(c);
      const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
      pixels[idx + 0] = to_u8(c.x);
      pixels[idx + 1] = to_u8(c.y);
      pixels[idx + 2] = to_u8(c.z);
      if (!sample_pixels.empty()) {
        const double fraction = static_cast<double>(used_samples) / spp;
        sample_pixels[idx] = to_u8(fraction);
        sample_pixels[idx + 1] = to_u8(0.2 + 0.6 * (1.0 - std::abs(2.0 * fraction - 1.0)));
        sample_pixels[idx + 2] = to_u8(1.0 - fraction);
      }
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  write_ppm(output, width, height, pixels);
  if (!samples_output.empty()) write_ppm(samples_output, width, height, sample_pixels);

  const double avg_luminance = luminance_sum / static_cast<double>(pixel_count);
  std::cout << "Rendered: " << width << "x" << height << "\n";
  std::cout << "Integrator: " << integrator << "\n";
  if (integrator == "path") {
    std::cout << "BSDF: " << (path_options.ggx ? "GGX + VNDF" : "ideal diffuse/mirror")
              << ", roughness override (-1=material): " << path_options.roughness << "\n";
    std::cout << "NEE+MIS: " << path_options.nee << ", RR depth: " << path_options.rr_depth
              << ", environment: " << path_options.environment << "\n";
    std::cout << "Path segments: " << path_stats.segments << ", shadow rays: " << path_stats.shadows << "\n";
    std::cout << "Diffuse/mirror/refraction events: " << path_stats.diffuse << "/"
              << path_stats.mirror << "/" << path_stats.refraction << "\n";
    std::cout << "GGX glossy samples: " << path_stats.glossy << "\n";
    std::cout << "Roulette terminations: " << path_stats.roulette << "\n";
    std::cout << "Mean segments per sample: " << path_stats.segments / static_cast<double>(total_samples) << "\n";
  }
  std::cout << "Scene: " << scene << ", boxes: " << boxes.size() << ", lights: " << lights.size() << "\n";
  std::cout << "Spheres (excluding ground): " << spheres.size() << "\n";
  std::cout << "Ellipsoids: " << ellipsoids.size() << "\n";
  std::cout << "Triangles: " << triangles.size() << ", mesh detail: " << detail << "\n";
  std::cout << "Light samples: " << light_samples << ", radius: " << light_radius << "\n";
  std::cout << "Seed: " << seed << "\n";
  std::cout << (adaptive_tolerance > 0.0 ? "Samples per pixel (cap): " : "Samples per pixel: ") << spp << "\n";
  std::cout << "Actual samples min/mean/max: " << minimum_samples << "/"
            << total_samples / static_cast<double>(pixel_count) << "/" << maximum_samples << "\n";
  std::cout << "Adaptive tolerance: " << adaptive_tolerance << ", pixels stopped early: " << stopped_pixels << "\n";
  std::cout << "Total camera samples: " << total_samples << "\n";
  std::cout << "Max depth: " << max_depth << "\n";
  std::cout << "Primary hits (first sample per pixel): " << primary_hits << "\n";
  std::cout << "Average luminance: " << avg_luminance << "\n";
  std::cout << "Render time (ms): " << ms << "\n";
  std::cout << "Output: " << output << "\n";
  if (!samples_output.empty()) std::cout << "Sample-count heatmap: " << samples_output << "\n";

  return 0;
} catch (const std::exception& e) {
  std::cerr << "Error: " << e.what() << "\n";
  return 1;
}
