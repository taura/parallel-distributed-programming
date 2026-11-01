#pragma once

#include "geometry/hittable.h"

// Infinite plane defined by point + normal.
// Also stores local tangent basis for procedural UV generation.
class Plane : public Hittable {
public:
    Plane(const Vec3& point, const Vec3& normal, const Material* material);

    bool intersect(
        const Ray& ray,
        double t_min,
        double t_max,
        HitRecord& rec
    ) const override;

private:
    Vec3 point_;
    Vec3 normal_;
    Vec3 tangent_;
    Vec3 bitangent_;
    const Material* material_;
};