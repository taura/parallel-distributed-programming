#include "geometry.h"

void AABB::expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
void AABB::expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
Vec3 AABB::centroid() const { return (lo + hi) * 0.5; }

double AABB::surface_area() const {
  Vec3 e = hi - lo;
  return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
}

bool AABB::intersect(const Ray& r, double t_min, double t_max) const {
  for (int a = 0; a < 3; a++) {
    double inv = 1.0 / r.d[a];
    double t0 = (lo[a] - r.o[a]) * inv;
    double t1 = (hi[a] - r.o[a]) * inv;
    if (inv < 0.0) std::swap(t0, t1);
    t_min = t0 > t_min ? t0 : t_min;
    t_max = t1 < t_max ? t1 : t_max;
    if (t_max < t_min) return false;
  }
  return true;
}

AABB bounds(const Triangle& tri) {
  AABB b;
  b.expand(tri.v0);
  b.expand(tri.v1);
  b.expand(tri.v2);
  return b;
}

AABB bounds(const Sphere& s) {
  Vec3 r(s.radius, s.radius, s.radius);
  AABB b;
  b.lo = s.center - r;
  b.hi = s.center + r;
  return b;
}

// Moller-Trumbore
bool intersect(const Ray& r, const Triangle& tri, Hit& hit) {
  Vec3 e1 = tri.v1 - tri.v0;
  Vec3 e2 = tri.v2 - tri.v0;
  Vec3 pv = cross(r.d, e2);
  double det = dot(e1, pv);
  if (std::fabs(det) < 1e-12) return false;  // ray parallel to the triangle
  double inv_det = 1.0 / det;
  Vec3 tv = r.o - tri.v0;
  double u = dot(tv, pv) * inv_det;
  if (u < 0.0 || u > 1.0) return false;
  Vec3 qv = cross(tv, e1);
  double v = dot(r.d, qv) * inv_det;
  if (v < 0.0 || u + v > 1.0) return false;
  double t = dot(e2, qv) * inv_det;
  if (t <= kEps || t >= hit.t) return false;
  hit.t = t;
  hit.p = r.o + r.d * t;
  hit.n = normalize(cross(e1, e2));
  hit.material = tri.material;
  return true;
}

// solve |o + t d - c|^2 = R^2 (d is normalized)
bool intersect(const Ray& r, const Sphere& s, Hit& hit) {
  Vec3 oc = r.o - s.center;
  double b = dot(oc, r.d);
  double c = dot(oc, oc) - s.radius * s.radius;
  double disc = b * b - c;
  if (disc < 0.0) return false;
  double sq = std::sqrt(disc);
  double t = -b - sq;
  if (t <= kEps) t = -b + sq;  // origin inside the sphere
  if (t <= kEps || t >= hit.t) return false;
  hit.t = t;
  hit.p = r.o + r.d * t;
  hit.n = (hit.p - s.center) / s.radius;
  hit.material = s.material;
  return true;
}
