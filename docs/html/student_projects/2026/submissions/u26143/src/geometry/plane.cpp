#include "geometry/plane.h"

#include <cmath>

#include "math/math_utils.h"

Plane::Plane(const Vec3& point, const Vec3& normal, const Material* material)
    : point_(point), normal_(normalize(normal)), material_(material) {
    // Build stable tangent/bitangent basis from normal.
    const Vec3 helper = (std::fabs(normal_.y) < 0.999) ? Vec3(0.0, 1.0, 0.0)
                                                       : Vec3(1.0, 0.0, 0.0);
    tangent_ = normalize(cross(helper, normal_));
    bitangent_ = cross(normal_, tangent_);
}

bool Plane::intersect(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    const double denom = dot(normal_, ray.direction);
    if (std::fabs(denom) < math::kEpsilon) {
        return false; // Parallel (or nearly parallel) ray.
    }

    const double t = dot(point_ - ray.origin, normal_) / denom;
    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.point = ray.at(t);
    rec.set_face_normal(ray.direction, normal_);
    rec.material = material_;

    // Procedural planar coordinates for checkerboard usage later.
    const Vec3 rel = rec.point - point_;
    rec.u = dot(rel, tangent_);
    rec.v = dot(rel, bitangent_);

    return true;
}