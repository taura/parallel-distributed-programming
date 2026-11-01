// geometry.h : rays, primitives, bounding boxes and ray-primitive intersection
#pragma once
#include "vec3.h"
#include <limits>

constexpr double kEps = 1e-6;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Ray {
  Vec3 o;  // origin
  Vec3 d;  // direction (normalized)
};

struct Hit {
  double t = kInf;  // distance along the ray
  Vec3 p;           // hit position
  Vec3 n;           // geometric normal (normalized, outward for closed objects)
  int material = -1;
};

struct Triangle {
  Vec3 v0, v1, v2;  // counter-clockwise seen from outside
  int material;
};

struct Sphere {
  Vec3 center;
  double radius;
  int material;
};

struct AABB {
  Vec3 lo = Vec3(kInf, kInf, kInf);
  Vec3 hi = Vec3(-kInf, -kInf, -kInf);
  void expand(const Vec3& p);
  void expand(const AABB& b);
  Vec3 centroid() const;
  double surface_area() const;
  // slab test; returns true if the ray hits the box within (t_min, t_max)
  bool intersect(const Ray& r, double t_min, double t_max) const;
};

AABB bounds(const Triangle& tri);
AABB bounds(const Sphere& s);

// Return true and fill `hit` if the primitive is hit closer than hit.t
bool intersect(const Ray& r, const Triangle& tri, Hit& hit);
bool intersect(const Ray& r, const Sphere& s, Hit& hit);
