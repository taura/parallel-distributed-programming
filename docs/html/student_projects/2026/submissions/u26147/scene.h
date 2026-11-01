// scene.h : procedurally generated scene (no external model files)
#pragma once
#include "geometry.h"
#include "material.h"
#include "bvh.h"
#include <vector>

struct Camera {
  Vec3 eye, look_at, up;
  double vfov_deg;
  // Primary ray through pixel (px, py) with sub-pixel jitter (jx, jy) in [0,1).
  // py = 0 is the top row.
  Ray generate_ray(int px, int py, int width, int height, double jx, double jy) const;
};

struct Scene {
  std::vector<Material> materials;
  std::vector<Triangle> triangles;
  std::vector<Sphere> spheres;  // few analytic spheres, intersected linearly
  BVH bvh;
  Camera camera;
  Vec3 background;  // radiance of rays escaping the scene

  int add_material(const Material& m);
  bool intersect(const Ray& r, Hit& hit) const;  // BVH triangles + spheres
};

// Scene builders. `detail` controls triangle count so the workload scales.
//   0 : Cornell box + analytic spheres (sanity check, known appearance)
//   1 : Cornell box + icosphere meshes, 20 * 4^detail triangles each (BVH workload)
//   2 : furnace test (white diffuse icosphere in uniform white environment)
Scene build_scene(int scene_id, int detail);

// Helpers for building geometry
void add_quad(Scene& s, Vec3 a, Vec3 b, Vec3 c, Vec3 d, int mat);  // a,b,c,d counter-clockwise
void add_icosphere(Scene& s, Vec3 center, double r, int subdiv, int mat);
