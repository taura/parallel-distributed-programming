#include "scene.h"
#include <cstdio>
#include <cstdlib>
#include <array>
#include <map>

static const double kPi = 3.14159265358979323846;

Ray Camera::generate_ray(int px, int py, int width, int height, double jx, double jy) const {
  Vec3 f = normalize(look_at - eye);
  Vec3 r = normalize(cross(f, up));
  Vec3 u = cross(r, f);
  double h = std::tan(vfov_deg * kPi / 360.0);
  double aspect = (double)width / height;
  double sx = (2.0 * (px + jx) / width - 1.0) * aspect * h;
  double sy = (1.0 - 2.0 * (py + jy) / height) * h;
  Ray ray;
  ray.o = eye;
  ray.d = normalize(f + r * sx + u * sy);
  return ray;
}

int Scene::add_material(const Material& m) {
  materials.push_back(m);
  return (int)materials.size() - 1;
}

bool Scene::intersect(const Ray& r, Hit& hit) const {
  bool found = bvh.intersect(r, triangles, hit);
  for (const Sphere& s : spheres) found |= ::intersect(r, s, hit);
  return found;
}

void add_quad(Scene& s, Vec3 a, Vec3 b, Vec3 c, Vec3 d, int mat) {
  s.triangles.push_back({a, b, c, mat});
  s.triangles.push_back({a, c, d, mat});
}

// Start from an icosahedron, split every triangle into 4 `subdiv` times and
// push the vertices onto the sphere.
void add_icosphere(Scene& s, Vec3 center, double r, int subdiv, int mat) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<Vec3> v = {
      {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
      {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (Vec3& p : v) p = normalize(p);
  std::vector<std::array<int, 3>> f = {
      {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
      {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
      {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  for (int level = 0; level < subdiv; level++) {
    std::map<std::pair<int, int>, int> mid_cache;
    auto midpoint = [&](int a, int b) {
      std::pair<int, int> key(std::min(a, b), std::max(a, b));
      auto it = mid_cache.find(key);
      if (it != mid_cache.end()) return it->second;
      v.push_back(normalize(v[a] + v[b]));
      return mid_cache[key] = (int)v.size() - 1;
    };
    std::vector<std::array<int, 3>> nf;
    nf.reserve(f.size() * 4);
    for (auto& tri : f) {
      int a = midpoint(tri[0], tri[1]), b = midpoint(tri[1], tri[2]), c = midpoint(tri[2], tri[0]);
      nf.push_back({tri[0], a, c});
      nf.push_back({tri[1], b, a});
      nf.push_back({tri[2], c, b});
      nf.push_back({a, b, c});
    }
    f.swap(nf);
  }
  for (auto& tri : f) {
    Triangle T{center + v[tri[0]] * r, center + v[tri[1]] * r, center + v[tri[2]] * r, mat};
    // make sure the winding gives an outward normal (needed for glass)
    Vec3 n = cross(T.v1 - T.v0, T.v2 - T.v0);
    if (dot(n, T.v0 - center) < 0) std::swap(T.v1, T.v2);
    s.triangles.push_back(T);
  }
}

// Cornell box: x in [-1,1], y in [0,2], z in [-1,1], open towards +z (camera side)
static void add_cornell_box(Scene& s) {
  int white = s.add_material({MaterialType::Diffuse, Vec3(0.73, 0.73, 0.73), Vec3(), 1.0});
  int red = s.add_material({MaterialType::Diffuse, Vec3(0.65, 0.05, 0.05), Vec3(), 1.0});
  int green = s.add_material({MaterialType::Diffuse, Vec3(0.12, 0.45, 0.15), Vec3(), 1.0});
  int light = s.add_material({MaterialType::Light, Vec3(), Vec3(5.0, 5.0, 5.0), 1.0});
  add_quad(s, {-1, 0, 1}, {1, 0, 1}, {1, 0, -1}, {-1, 0, -1}, white);     // floor
  add_quad(s, {-1, 2, -1}, {1, 2, -1}, {1, 2, 1}, {-1, 2, 1}, white);     // ceiling
  add_quad(s, {-1, 0, -1}, {1, 0, -1}, {1, 2, -1}, {-1, 2, -1}, white);   // back
  add_quad(s, {-1, 0, 1}, {-1, 0, -1}, {-1, 2, -1}, {-1, 2, 1}, red);     // left
  add_quad(s, {1, 0, -1}, {1, 0, 1}, {1, 2, 1}, {1, 2, -1}, green);       // right
  add_quad(s, {-0.5, 1.99, -0.5}, {0.5, 1.99, -0.5}, {0.5, 1.99, 0.5}, {-0.5, 1.99, 0.5}, light);
  s.background = Vec3();
  s.camera = {Vec3(0, 1, 3.4), Vec3(0, 1, 0), Vec3(0, 1, 0), 40.0};
}

Scene build_scene(int scene_id, int detail) {
  Scene s;
  if (scene_id == 0 || scene_id == 1) {
    add_cornell_box(s);
    int diffuse = s.add_material({MaterialType::Diffuse, Vec3(0.73, 0.73, 0.73), Vec3(), 1.0});
    int mirror = s.add_material({MaterialType::Mirror, Vec3(0.95, 0.95, 0.95), Vec3(), 1.0});
    int glass = s.add_material({MaterialType::Glass, Vec3(1, 1, 1), Vec3(), 1.5});
    Vec3 c[3] = {Vec3(-0.5, 0.4, -0.45), Vec3(0.5, 0.4, -0.3), Vec3(0.0, 0.3, 0.45)};
    double r[3] = {0.4, 0.4, 0.3};
    int m[3] = {diffuse, mirror, glass};
    for (int i = 0; i < 3; i++) {
      if (scene_id == 0) s.spheres.push_back({c[i], r[i], m[i]});
      else add_icosphere(s, c[i], r[i], detail, m[i]);
    }
  } else if (scene_id == 2) {
    int white = s.add_material({MaterialType::Diffuse, Vec3(1, 1, 1), Vec3(), 1.0});
    add_icosphere(s, Vec3(0, 0, 0), 1.0, detail, white);
    s.background = Vec3(1, 1, 1);
    s.camera = {Vec3(0, 0, 4), Vec3(0, 0, 0), Vec3(0, 1, 0), 40.0};
  } else {
    fprintf(stderr, "unknown scene id %d\n", scene_id);
    exit(1);
  }
  s.bvh.build(s.triangles, 4);
  return s;
}
