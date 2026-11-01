#include "geometry/sphere.h"

#include <cmath>

#include "math/math_utils.h"

Sphere::Sphere(const Vec3& center, double radius, const Material* material)
    : center_(center), radius_(radius), material_(material) {}

bool Sphere::intersect(const Ray& ray, double t_min, double t_max, HitRecord& rec) const {
    // Solve ||(o + t d) - c||^2 = r^2
    const Vec3 oc = ray.origin - center_;
    const double a = dot(ray.direction, ray.direction);
    const double half_b = dot(oc, ray.direction);
    const double c = dot(oc, oc) - radius_ * radius_;

    const double discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return false;
    }

    const double sqrt_d = std::sqrt(discriminant);

    // Try nearest root first.
    double root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.point = ray.at(rec.t);

    const Vec3 outward_normal = (rec.point - center_) / radius_;
    rec.set_face_normal(ray.direction, outward_normal);
    rec.material = material_;

    // Spherical UV (optional; useful for future texturing).
    const double theta = std::acos(-outward_normal.y);
    const double phi = std::atan2(-outward_normal.z, outward_normal.x) + math::kPi;
    rec.u = phi / (2.0 * math::kPi);
    rec.v = theta / math::kPi;

    return true;
}